"""FastAPI backend: authentication, multi-omics upload, drug response prediction, history."""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ai_model.preprocess import parse_long_csv

from .auth import ROLES, create_token, current_user, ensure_admin, hash_password, require_roles, verify_password
from .config import settings
from .db import Experiment, Prediction, Sample, SessionLocal, User, get_db, init_db
from .inference import make_inference
from .monitoring import metrics, setup_logging
from .storage import make_storage

setup_logging()
log = logging.getLogger("api")
state = {}


@asynccontextmanager
async def lifespan(app):
    init_db()
    with SessionLocal() as db:
        ensure_admin(db)
    state["storage"] = make_storage()
    try:
        state["inference"] = make_inference()
        info = state["inference"].info()
        with SessionLocal() as db:
            if not db.query(Experiment).filter_by(model_version=info["version"]).first():
                db.add(Experiment(model_version=info["version"], metrics=info.get("metrics")))
                db.commit()
        log.info("model ready: %s", info["version"])
    except Exception:
        log.exception("model not loaded; prediction endpoints will return 503")
        state["inference"] = None
    yield


app = FastAPI(title="AI Drug Response Prediction API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    if request.url.path.startswith("/api"):
        log.info("request", extra={"extra_fields": {
            "method": request.method, "path": request.url.path, "status": response.status_code, "ms": round(ms, 1)}})
        if response.status_code >= 500:
            metrics.put("ServerErrors", 1)
    return response


def inference():
    if state.get("inference") is None:
        raise HTTPException(503, "Model is not loaded. Check MODEL_DIR / MODEL_S3_URI / SAGEMAKER_ENDPOINT.")
    return state["inference"]


# ---------------------------------------------------------------- schemas
class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class NewUser(Credentials):
    role: str = "researcher"


class PredictRequest(BaseModel):
    sample_id: int
    drug_ids: Optional[List[str]] = None
    top_k: int = Field(10, ge=1, le=50)


def user_out(u):
    return {"id": u.id, "email": u.email, "role": u.role}


def sample_out(s):
    return {"id": s.id, "name": s.name, "cancer_type": s.cancer_type, "feature_counts": s.feature_counts,
            "created_at": s.created_at.isoformat() + "Z", "owner_id": s.owner_id}


def prediction_out(p):
    return {"id": p.id, "sample_id": p.sample_id, "drug_id": p.drug_id, "drug_name": p.drug_name,
            "predicted_ln_ic50": p.predicted_ln_ic50, "predicted_ic50_um": p.predicted_ic50_um,
            "sensitivity": p.sensitivity, "sensitivity_score": p.sensitivity_score,
            "explanation": p.explanation, "model_version": p.model_version,
            "created_at": p.created_at.isoformat() + "Z"}


def owned_sample(db, sample_id, user):
    s = db.get(Sample, sample_id)
    if s is None or (user.role != "admin" and s.owner_id != user.id):
        raise HTTPException(404, "Sample not found")
    return s


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.get("inference") is not None}


@app.post("/api/auth/signup")
def signup(body: Credentials, db: Session = Depends(get_db)):
    if not settings.allow_signup:
        raise HTTPException(403, "Self sign-up is disabled; ask an administrator")
    email = body.email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(409, "Email already registered")
    user = User(email=email, password_hash=hash_password(body.password), role="researcher")
    db.add(user)
    db.commit()
    return {"token": create_token(user), "user": user_out(user)}


@app.post("/api/auth/login")
def login(body: Credentials, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.strip().lower()).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return {"token": create_token(user), "user": user_out(user)}


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return user_out(user)


@app.get("/api/users")
def list_users(db: Session = Depends(get_db), _: User = Depends(require_roles("admin"))):
    return [user_out(u) for u in db.query(User).order_by(User.id)]


@app.post("/api/users")
def create_user(body: NewUser, db: Session = Depends(get_db), _: User = Depends(require_roles("admin"))):
    if body.role not in ROLES:
        raise HTTPException(400, "role must be one of %s" % (ROLES,))
    email = body.email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(409, "Email already registered")
    user = User(email=email, password_hash=hash_password(body.password), role=body.role)
    db.add(user)
    db.commit()
    return user_out(user)


@app.get("/api/drugs")
def drugs(_: User = Depends(current_user)):
    return sorted(inference().list_drugs(), key=lambda d: d["drug_name"].lower())


@app.get("/api/model")
def model_info(_: User = Depends(current_user)):
    return inference().info()


@app.post("/api/samples")
async def upload_sample(
    file: UploadFile = File(...),
    name: str = Form(""),
    cancer_type: str = Form(""),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "File larger than %d MB" % settings.max_upload_mb)
    try:
        parsed = parse_long_csv(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as e:
        raise HTTPException(400, "Could not parse CSV: %s" % e)
    except Exception:
        raise HTTPException(400, "Could not parse CSV: expected columns modality,feature,value")
    if not parsed:
        raise HTTPException(400, "CSV contains no rows")

    storage = state["storage"]
    batch = uuid.uuid4().hex
    storage.put("uploads/user-%d/%s/%s" % (user.id, batch, os.path.basename(file.filename or "upload.csv")), raw)
    created = []
    for sid, sample in parsed.items():
        label = name or os.path.splitext(file.filename or "sample")[0]
        if len(parsed) > 1:
            label = "%s / %s" % (label, sid)
        key = "processed/user-%d/%s/%s.json" % (user.id, batch, uuid.uuid4().hex)
        storage.put(key, json.dumps(sample).encode())
        s = Sample(name=label[:255], cancer_type=cancer_type or None, owner_id=user.id, storage_key=key,
                   feature_counts={m: len(v) for m, v in sample.items()})
        db.add(s)
        created.append(s)
    db.commit()
    metrics.put("SamplesUploaded", len(created))
    return [sample_out(s) for s in created]


@app.get("/api/samples")
def list_samples(db: Session = Depends(get_db), user: User = Depends(current_user)):
    q = db.query(Sample)
    if user.role != "admin":
        q = q.filter(Sample.owner_id == user.id)
    return [sample_out(s) for s in q.order_by(Sample.id.desc()).limit(200)]


@app.post("/api/predict")
def predict(body: PredictRequest, db: Session = Depends(get_db), user: User = Depends(current_user)):
    sample_row = owned_sample(db, body.sample_id, user)
    sample = json.loads(state["storage"].get(sample_row.storage_key))
    start = time.time()
    try:
        result = inference().predict(sample, drug_ids=body.drug_ids, explain=True, top_k=body.top_k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    latency = (time.time() - start) * 1000
    metrics.put("PredictionLatency", latency, unit="Milliseconds")
    metrics.put("Predictions", len(result["predictions"]))

    rows = []
    for r in result["predictions"]:
        p = Prediction(
            sample_id=sample_row.id, user_id=user.id, drug_id=r["drug_id"], drug_name=r["drug_name"],
            predicted_ln_ic50=r["predicted_ln_ic50"], predicted_ic50_um=r["predicted_ic50_um"],
            sensitivity=r["sensitivity"], sensitivity_score=r["sensitivity_score"],
            explanation={"modality_attention": r.get("modality_attention"), "top_features": r.get("top_features")},
            model_version=result["model_version"],
        )
        db.add(p)
        rows.append(p)
    db.commit()
    log.info("prediction", extra={"extra_fields": {"sample_id": sample_row.id, "n_drugs": len(rows),
                                                   "latency_ms": round(latency, 1)}})
    return {"sample": sample_out(sample_row), "model_version": result["model_version"],
            "coverage": result["coverage"], "predictions": [prediction_out(p) for p in rows]}


@app.get("/api/predictions")
def list_predictions(sample_id: Optional[int] = None, limit: int = 200,
                     db: Session = Depends(get_db), user: User = Depends(current_user)):
    q = db.query(Prediction)
    if user.role != "admin":
        q = q.filter(Prediction.user_id == user.id)
    if sample_id is not None:
        q = q.filter(Prediction.sample_id == sample_id)
    return [prediction_out(p) for p in q.order_by(Prediction.id.desc()).limit(min(limit, 1000))]


@app.get("/api/stats")
def stats(db: Session = Depends(get_db), user: User = Depends(current_user)):
    q = db.query(Prediction.drug_name, Prediction.sensitivity, func.count(Prediction.id))
    if user.role != "admin":
        q = q.filter(Prediction.user_id == user.id)
    by_drug = {}
    for drug, sens, n in q.group_by(Prediction.drug_name, Prediction.sensitivity):
        by_drug.setdefault(drug, {"Sensitive": 0, "Resistant": 0})[sens] = n
    samples = db.query(func.count(Sample.id))
    if user.role != "admin":
        samples = samples.filter(Sample.owner_id == user.id)
    return {"samples": samples.scalar(), "predictions": sum(sum(v.values()) for v in by_drug.values()),
            "by_drug": by_drug}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


_frontend = settings.frontend_dir or os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")

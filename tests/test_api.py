import importlib
import os

import pytest


@pytest.fixture(scope="module")
def client(trained, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    os.environ.update({
        "DATABASE_URL": "sqlite:///%s" % (tmp / "test.db"), "STORAGE_DIR": str(tmp / "storage"),
        "MODEL_DIR": str(trained["model"]), "ADMIN_EMAIL": "admin@test.org", "ADMIN_PASSWORD": "adminpass123",
        "INFERENCE_MODE": "local", "STORAGE_BACKEND": "local",
    })
    from fastapi.testclient import TestClient

    from backend.app import auth, config, db, inference, main, monitoring, storage
    for mod in (config, db, auth, storage, inference, monitoring, main):  # pick up the env vars set above
        importlib.reload(mod)
    with TestClient(main.app) as c:
        yield c


def auth(token):
    return {"Authorization": "Bearer " + token}


def test_full_flow(client, trained):
    assert client.get("/health").json()["model_loaded"] is True
    r = client.post("/api/auth/signup", json={"email": "Res@Test.org", "password": "password123"})
    assert r.status_code == 200
    token = r.json()["token"]
    assert client.post("/api/auth/login", json={"email": "res@test.org", "password": "wrong-password"}).status_code == 401

    with open(trained["sample"], "rb") as f:
        r = client.post("/api/samples", headers=auth(token), files={"file": ("p.csv", f, "text/csv")},
                        data={"name": "Patient A", "cancer_type": "Lung"})
    assert r.status_code == 200, r.text
    sample_id = r.json()[0]["id"]

    drugs = client.get("/api/drugs", headers=auth(token)).json()
    r = client.post("/api/predict", headers=auth(token),
                    json={"sample_id": sample_id, "drug_ids": [d["drug_id"] for d in drugs[:3]]})
    assert r.status_code == 200, r.text
    assert len(r.json()["predictions"]) == 3
    assert len(client.get("/api/predictions", headers=auth(token)).json()) == 3
    assert client.get("/api/stats", headers=auth(token)).json()["predictions"] == 3


def test_access_control(client, trained):
    a = client.post("/api/auth/signup", json={"email": "a@test.org", "password": "password123"}).json()["token"]
    b = client.post("/api/auth/signup", json={"email": "b@test.org", "password": "password123"}).json()["token"]
    with open(trained["sample"], "rb") as f:
        sid = client.post("/api/samples", headers=auth(a), files={"file": ("p.csv", f, "text/csv")}).json()[0]["id"]
    # another researcher cannot see or use the sample
    assert client.post("/api/predict", headers=auth(b), json={"sample_id": sid}).status_code == 404
    assert client.get("/api/users", headers=auth(a)).status_code == 403
    assert client.get("/api/drugs").status_code == 401
    admin = client.post("/api/auth/login", json={"email": "admin@test.org", "password": "adminpass123"}).json()["token"]
    assert client.get("/api/users", headers=auth(admin)).status_code == 200
    assert client.post("/api/predict", headers=auth(admin), json={"sample_id": sid}).status_code == 200


def test_bad_upload(client):
    token = client.post("/api/auth/signup", json={"email": "c@test.org", "password": "password123"}).json()["token"]
    r = client.post("/api/samples", headers=auth(token), files={"file": ("x.csv", b"gene,value\nEGFR,1\n", "text/csv")})
    assert r.status_code == 400

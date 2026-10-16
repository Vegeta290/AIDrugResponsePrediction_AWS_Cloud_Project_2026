"""SQLAlchemy models. Works with SQLite locally and Amazon RDS PostgreSQL in the cloud."""
import datetime as dt

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


def now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="researcher")  # admin | researcher | clinician
    created_at = Column(DateTime, default=now)


class Sample(Base):
    """An uploaded patient / cell-line multi-omics profile."""
    __tablename__ = "samples"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    cancer_type = Column(String(128))
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    storage_key = Column(String(512), nullable=False)  # raw upload in S3 / local storage
    feature_counts = Column(JSON)  # {modality: n_features}
    created_at = Column(DateTime, default=now)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey("samples.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    drug_id = Column(String(64), nullable=False)
    drug_name = Column(String(255))
    predicted_ln_ic50 = Column(Float)
    predicted_ic50_um = Column(Float)
    sensitivity = Column(String(16))
    sensitivity_score = Column(Float)
    explanation = Column(JSON)  # modality attention + top features
    model_version = Column(String(64))
    created_at = Column(DateTime, default=now, index=True)


class Experiment(Base):
    """Model registry / experiment log: one row per model version seen by the API."""
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True)
    model_version = Column(String(64), unique=True)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=now)


# Flat reporting views for Amazon QuickSight dashboards (PostgreSQL only).
REPORTING_VIEWS = [
    """CREATE OR REPLACE VIEW v_prediction_report AS
       SELECT p.id, p.created_at, p.drug_id, p.drug_name, p.predicted_ic50_um, p.predicted_ln_ic50,
              p.sensitivity, p.sensitivity_score, p.model_version,
              s.id AS sample_id, s.name AS sample_name, s.cancer_type, u.role AS requested_by_role
       FROM predictions p JOIN samples s ON s.id = p.sample_id JOIN users u ON u.id = p.user_id""",
    """CREATE OR REPLACE VIEW v_drug_summary AS
       SELECT drug_name, COUNT(*) AS n_predictions,
              SUM(CASE WHEN sensitivity = 'Sensitive' THEN 1 ELSE 0 END) AS n_sensitive,
              AVG(predicted_ln_ic50) AS mean_ln_ic50
       FROM predictions GROUP BY drug_name""",
]


def init_db():
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        from sqlalchemy import text

        with engine.begin() as conn:
            for sql in REPORTING_VIEWS:
                conn.execute(text(sql))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

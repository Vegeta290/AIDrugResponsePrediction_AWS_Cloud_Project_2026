"""Runtime configuration from environment variables (set by docker / EC2 setup script)."""
import os


def _bool(name, default="false"):
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


class Settings:
    # Local default is SQLite; on AWS this points at Amazon RDS PostgreSQL.
    database_url = os.environ.get("DATABASE_URL", "sqlite:///./drugresponse.db")

    # "local" = files under storage_dir, "s3" = Amazon S3 bucket.
    storage_backend = os.environ.get("STORAGE_BACKEND", "local")
    storage_dir = os.environ.get("STORAGE_DIR", "./storage")
    s3_bucket = os.environ.get("S3_BUCKET", "")
    aws_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))

    # "local" = model loaded in this process (EC2), "sagemaker" = SageMaker endpoint.
    inference_mode = os.environ.get("INFERENCE_MODE", "local")
    model_dir = os.environ.get("MODEL_DIR", "./models/latest")
    model_s3_uri = os.environ.get("MODEL_S3_URI", "")  # s3://.../model.tar.gz, downloaded at startup
    sagemaker_endpoint = os.environ.get("SAGEMAKER_ENDPOINT", "")

    jwt_secret = os.environ.get("JWT_SECRET", "dev-only-change-me")
    jwt_hours = int(os.environ.get("JWT_HOURS", "12"))
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@example.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    allow_signup = _bool("ALLOW_SIGNUP", "true")

    cloudwatch_metrics = _bool("CLOUDWATCH_METRICS")
    metrics_namespace = os.environ.get("METRICS_NAMESPACE", "DrugResponse")
    frontend_dir = os.environ.get("FRONTEND_DIR", "")
    max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "20"))


settings = Settings()

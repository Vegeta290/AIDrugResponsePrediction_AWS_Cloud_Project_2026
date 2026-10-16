"""Object storage abstraction: local disk for development, Amazon S3 in the cloud."""
import os

from .config import settings


class LocalStorage:
    def __init__(self, base):
        self.base = os.path.abspath(base)

    def put(self, key, data):
        path = os.path.join(self.base, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return key

    def get(self, key):
        with open(os.path.join(self.base, key), "rb") as f:
            return f.read()


class S3Storage:
    def __init__(self, bucket, region):
        import boto3

        self.bucket = bucket
        self.s3 = boto3.client("s3", region_name=region)

    def put(self, key, data):
        # SSE-S3 encryption at rest; the bucket policy also enforces it.
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ServerSideEncryption="AES256")
        return key

    def get(self, key):
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()


def make_storage():
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET")
        return S3Storage(settings.s3_bucket, settings.aws_region)
    return LocalStorage(settings.storage_dir)

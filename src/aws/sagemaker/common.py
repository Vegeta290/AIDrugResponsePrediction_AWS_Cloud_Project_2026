"""Helpers shared by the SageMaker scripts."""
import io
import json
import os
import tarfile

import boto3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AI_MODEL_DIR = os.path.join(ROOT, "src", "ai_model")
STATE_FILE = os.path.join(ROOT, "results", "last_training_job.json")
FRAMEWORK_VERSION = "2.1"
PY_VERSION = "py310"


def stack_outputs(stack, region):
    cf = boto3.client("cloudformation", region_name=region)
    outputs = cf.describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}


def upload_source(bucket, region):
    """Tar src/ai_model to s3://bucket/code/sourcedir.tar.gz (also used by the retraining Lambda)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in sorted(os.listdir(AI_MODEL_DIR)):
            if name.endswith((".py", ".txt")):
                tar.add(os.path.join(AI_MODEL_DIR, name), arcname=name)
    buf.seek(0)
    boto3.client("s3", region_name=region).upload_fileobj(buf, bucket, "code/sourcedir.tar.gz")
    return "s3://%s/code/sourcedir.tar.gz" % bucket


def save_state(data):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)

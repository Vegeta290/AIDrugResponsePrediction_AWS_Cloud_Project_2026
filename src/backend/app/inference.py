"""Inference service: in-process model (EC2) or Amazon SageMaker endpoint."""
import json
import logging
import os
import tarfile
import tempfile

from .config import settings

log = logging.getLogger("inference")


def _download_model(s3_uri, target_dir):
    import boto3

    bucket, key = s3_uri[len("s3://"):].split("/", 1)
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        boto3.client("s3", region_name=settings.aws_region).download_fileobj(bucket, key, tmp)
        tmp.flush()
        with tarfile.open(tmp.name) as tar:
            for member in tar.getmembers():  # refuse path traversal entries
                if member.name.startswith("/") or ".." in member.name.split("/"):
                    raise ValueError("unsafe path in model archive: %s" % member.name)
            tar.extractall(target_dir)
    log.info("downloaded model %s -> %s", s3_uri, target_dir)


class LocalInference:
    def __init__(self):
        from ai_model.predictor import DrugResponsePredictor

        model_dir = settings.model_dir
        if settings.model_s3_uri and not os.path.exists(os.path.join(model_dir, "model.pt")):
            _download_model(settings.model_s3_uri, model_dir)
        self.predictor = DrugResponsePredictor(model_dir)

    def predict(self, sample, drug_ids=None, explain=True, top_k=10):
        return self.predictor.predict(sample, drug_ids=drug_ids, explain=explain, top_k=top_k)

    def list_drugs(self):
        return self.predictor.list_drugs()

    def info(self):
        return self.predictor.info()


class SageMakerInference:
    def __init__(self):
        import boto3

        if not settings.sagemaker_endpoint:
            raise RuntimeError("INFERENCE_MODE=sagemaker requires SAGEMAKER_ENDPOINT")
        self.client = boto3.client("sagemaker-runtime", region_name=settings.aws_region)
        self._info = None

    def _call(self, payload):
        resp = self.client.invoke_endpoint(
            EndpointName=settings.sagemaker_endpoint, ContentType="application/json",
            Accept="application/json", Body=json.dumps(payload),
        )
        return json.loads(resp["Body"].read())

    def predict(self, sample, drug_ids=None, explain=True, top_k=10):
        return self._call({"sample": sample, "drug_ids": drug_ids, "explain": explain, "top_k": top_k})

    def list_drugs(self):
        return self._call({"action": "list_drugs"})["drugs"]

    def info(self):
        if self._info is None:
            self._info = self._call({"action": "info"})
        return self._info


def make_inference():
    return SageMakerInference() if settings.inference_mode == "sagemaker" else LocalInference()

import importlib.util
import json
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class FakeBody:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class FakeAWS:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.calls = []

    def client(self, name, **kw):
        fake = self

        class C:
            def get_object(self, Bucket, Key):
                return {"Body": FakeBody(fake.objects[Key])}

            def put_object(self, Bucket, Key, Body, **kw):
                fake.objects[Key] = Body

            def invoke_endpoint(self, **kw):
                return {"Body": FakeBody(json.dumps({"predictions": [{"drug_name": "X"}]}).encode())}

            def put_metric_data(self, **kw):
                fake.calls.append(("metric", kw))

            def create_training_job(self, **kw):
                fake.calls.append(("train", kw))

        return C()


def load_handler(fake, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=fake.client))
    spec = importlib.util.spec_from_file_location("handler", os.path.join(ROOT, "src/aws/lambda_preprocess/handler.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def event(key):
    return {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": key}}}]}


def test_incoming_csv(monkeypatch):
    csv = b"sample_id,modality,feature,value\nP1,expression,EGFR,7.2\nP1,mutation,KRAS,2\nP2,cnv,EGFR,NA\nP2,bogus,X,1\n"
    fake = FakeAWS({"incoming/batch1.csv": csv})
    monkeypatch.setenv("ENDPOINT_NAME", "ep")
    h = load_handler(fake, monkeypatch)
    assert h.lambda_handler(event("incoming/batch1.csv"), None)["results"][0] == {"samples": 1, "errors": 1}
    assert json.loads(fake.objects["processed/batch1/P1.json"]) == {"expression": {"EGFR": 7.2}, "mutation": {"KRAS": 1.0}}
    report = json.loads(fake.objects["predictions/batch1.json"])
    assert report["predictions"]["P1"]["predictions"][0]["drug_name"] == "X"


def test_retrain_trigger(monkeypatch):
    fake = FakeAWS({"triggers/retrain.json": b'{"hyperparameters": {"epochs": 5}}'})
    for k, v in {"AWS_REGION": "ap-south-1", "SAGEMAKER_ROLE_ARN": "arn:role", "TRAINING_IMAGE": "img"}.items():
        monkeypatch.setenv(k, v)
    h = load_handler(fake, monkeypatch)
    h.lambda_handler(event("triggers/retrain.json"), None)
    job = [c for c in fake.calls if c[0] == "train"][0][1]
    assert json.loads(job["HyperParameters"]["epochs"]) == "5"
    assert job["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] == "s3://b/datasets/canonical/"


def test_free_tier_disables_sagemaker(monkeypatch):
    fake = FakeAWS({"triggers/retrain.json": b"{}", "incoming/b.csv": b"sample_id,modality,feature,value\nP1,cnv,EGFR,1\n"})
    monkeypatch.setenv("ENABLE_SAGEMAKER", "false")
    monkeypatch.setenv("ENDPOINT_NAME", "ep")
    h = load_handler(fake, monkeypatch)
    assert h.lambda_handler(event("triggers/retrain.json"), None)["results"][0]["training_job"] is None
    h.lambda_handler(event("incoming/b.csv"), None)
    assert not [c for c in fake.calls if c[0] == "train"]
    assert json.loads(fake.objects["predictions/b.json"])["predictions"] == {}

"""Amazon SageMaker inference handlers (PyTorch serving container).

Request (application/json):
    {"sample": {"expression": {"EGFR": 7.1, ...}, "mutation": {...}},
     "drug_ids": ["1032", "1006"],      # optional, default = all drugs
     "explain": true, "top_k": 10}
    {"action": "list_drugs"}  |  {"action": "info"}
"""
import json

try:
    from .predictor import DrugResponsePredictor
except ImportError:
    from predictor import DrugResponsePredictor


def model_fn(model_dir):
    return DrugResponsePredictor(model_dir)


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError("Unsupported content type: %s" % content_type)
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return json.loads(body)


def predict_fn(data, predictor):
    action = data.get("action", "predict")
    if action == "list_drugs":
        return {"drugs": predictor.list_drugs()}
    if action == "info":
        return predictor.info()
    return predictor.predict(
        data["sample"], drug_ids=data.get("drug_ids"),
        explain=data.get("explain", True), top_k=int(data.get("top_k", 10)),
    )


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"

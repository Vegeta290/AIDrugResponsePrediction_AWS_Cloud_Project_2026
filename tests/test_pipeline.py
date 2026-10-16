import json

import numpy as np
import pytest

from ai_model import inference
from ai_model.drug_features import ATOM_DIM, smiles_to_graph
from ai_model.predictor import DrugResponsePredictor
from ai_model.preprocess import parse_long_csv


def test_training_artifacts(trained):
    for name in ("model.pt", "preprocessor.json", "config.json", "drugs.json", "metrics.json"):
        assert (trained["model"] / name).exists()
    metrics = json.loads((trained["model"] / "metrics.json").read_text())
    assert {"rmse", "pearson", "roc_auc", "precision", "recall"} <= set(metrics["test"])
    assert (trained["results"] / "test_predictions.csv").exists()


def test_smiles_graph_shapes():
    x, a, m = smiles_to_graph("C1=C(C(=O)NC(=O)N1)F")  # 5-fluorouracil, 9 heavy atoms
    assert x.shape[1] == ATOM_DIM and a.shape[0] == a.shape[1]
    assert m.sum() == 9
    assert np.allclose(a, a.T)


def test_predictor_with_missing_modalities(trained):
    p = DrugResponsePredictor(str(trained["model"]))
    sample = parse_long_csv(trained["sample"].read_text())["sample"]
    full = p.predict(sample)
    assert len(full["predictions"]) == len(p.drugs)
    first = full["predictions"][0]
    assert first["sensitivity"] in ("Sensitive", "Resistant")
    assert abs(sum(first["modality_attention"].values()) - 1) < 1e-3
    assert first["modality_attention"]["proteomics"] == 0  # absent modality is masked out
    only_expr = p.predict({"expression": sample["expression"]}, drug_ids=[p.drugs[0]["drug_id"]], explain=False)
    assert len(only_expr["predictions"]) == 1


def test_predictor_rejects_unrelated_features(trained):
    p = DrugResponsePredictor(str(trained["model"]))
    with pytest.raises(ValueError):
        p.predict({"expression": {"NOT_A_GENE": 1.0}})


def test_sagemaker_handlers(trained):
    model = inference.model_fn(str(trained["model"]))
    sample = parse_long_csv(trained["sample"].read_text())["sample"]
    body = json.dumps({"sample": sample, "top_k": 3})
    out, ctype = inference.output_fn(inference.predict_fn(inference.input_fn(body, "application/json"), model))
    assert ctype == "application/json" and json.loads(out)["predictions"]
    drugs = inference.predict_fn({"action": "list_drugs"}, model)["drugs"]
    assert drugs and "drug_name" in drugs[0]


def test_parse_long_csv_validation():
    with pytest.raises(ValueError):
        parse_long_csv("modality,feature\nexpression,EGFR\n")
    with pytest.raises(ValueError):
        parse_long_csv("modality,feature,value\nmetabolome,X,1\n")
    multi = parse_long_csv("sample_id,modality,feature,value\nA,expression,EGFR,1\nB,cnv,EGFR,0.2\n")
    assert set(multi) == {"A", "B"}

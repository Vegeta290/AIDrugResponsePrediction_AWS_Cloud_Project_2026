"""Validate a cell-line-trained model on TCGA patients (objective 4 in the proposal).

TCGA has no IC50 values, so validation uses clinical drug response:
responders (Complete/Partial Response) should get lower predicted ln(IC50)
than non-responders (Stable/Progressive Disease) for the same drug.

Inputs (canonical layout, rows = patients):
    <tcga-dir>/expression.csv, cnv.csv, mutation.csv   (any subset)
    <tcga-dir>/clinical_response.csv   patient_id, drug_name, response
       response in {Complete Response, Partial Response, Stable Disease, Clinical Progressive Disease}
       or short codes CR/PR/SD/PD.

Usage:
    PYTHONPATH=src python -m ai_model.validate_tcga --model-dir models/latest --tcga-dir data/tcga
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from .predictor import DrugResponsePredictor
    from .preprocess import load_omics
except ImportError:
    from predictor import DrugResponsePredictor
    from preprocess import load_omics

RESPONDER = {"complete response": 1, "partial response": 1, "cr": 1, "pr": 1,
             "stable disease": 0, "clinical progressive disease": 0, "progressive disease": 0, "sd": 0, "pd": 0}


def recenter_to_cell_lines(df, preproc, modality):
    """Remove the tumour-vs-cell-line batch shift: z-score within the patient cohort,
    then map onto the cell-line training distribution so the model's scaler yields the same z."""
    cols = [c for c in preproc.features[modality] if c in df.columns]
    sub = df[cols]
    z = (sub - sub.mean()) / sub.std().replace(0, 1)
    idx = {f: i for i, f in enumerate(preproc.features[modality])}
    mean = np.array([preproc.mean[modality][idx[c]] for c in cols])
    std = np.array([preproc.std[modality][idx[c]] for c in cols])
    return z * std + mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/latest")
    p.add_argument("--tcga-dir", required=True)
    p.add_argument("--output", default="results/tcga_validation.json")
    p.add_argument("--no-recenter", action="store_true", help="skip cohort re-centring (not recommended)")
    p.add_argument("--min-patients", type=int, default=10)
    args = p.parse_args()

    predictor = DrugResponsePredictor(args.model_dir)
    omics = load_omics(args.tcga_dir)
    if not args.no_recenter:
        for m in list(omics):
            if m in predictor.preproc.features and m != "mutation":
                omics[m] = recenter_to_cell_lines(omics[m], predictor.preproc, m)

    clin = pd.read_csv(os.path.join(args.tcga_dir, "clinical_response.csv"))
    clin["label"] = clin["response"].astype(str).str.strip().str.lower().map(RESPONDER)
    clin = clin.dropna(subset=["label"])
    name_to_id = {d["drug_name"].lower(): d["drug_id"] for d in predictor.drugs}
    clin["drug_id"] = clin["drug_name"].astype(str).str.lower().map(name_to_id)
    unmatched = sorted(clin.loc[clin["drug_id"].isna(), "drug_name"].unique())
    clin = clin.dropna(subset=["drug_id"])

    rows = []
    for pid, grp in clin.groupby("patient_id"):
        sample = {m: df.loc[pid].dropna().to_dict() for m, df in omics.items() if pid in df.index}
        if not sample:
            continue
        out = predictor.predict(sample, drug_ids=list(grp["drug_id"].unique()), explain=False)
        pred = {r["drug_id"]: r["sensitivity_score"] for r in out["predictions"]}
        for r in grp.itertuples():
            rows.append({"patient_id": pid, "drug_name": r.drug_name, "label": int(r.label),
                         "sensitivity_score": pred[r.drug_id]})
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No patients with both omics data and matched drug responses")

    report = {"n_pairs": len(res), "n_patients": int(res["patient_id"].nunique()),
              "unmatched_drugs": unmatched, "per_drug_auc": {}}
    for drug, grp in res.groupby("drug_name"):
        if len(grp) >= args.min_patients and grp["label"].nunique() == 2:
            report["per_drug_auc"][drug] = {"auc": float(roc_auc_score(grp["label"], grp["sensitivity_score"])),
                                            "n": int(len(grp))}
    if res["label"].nunique() == 2:
        # Rank within drug so drugs with different score scales are comparable.
        res["rank"] = res.groupby("drug_name")["sensitivity_score"].rank(pct=True)
        report["pooled_auc"] = float(roc_auc_score(res["label"], res["rank"]))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

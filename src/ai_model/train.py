"""Train DrugResponseNet. Runs locally or as an Amazon SageMaker training job.

Local:      PYTHONPATH=src python -m ai_model.train --data-dir data/canonical --model-dir models/latest
SageMaker:  launched by src/aws/sagemaker/launch_training.py; reads SM_CHANNEL_TRAIN / SM_MODEL_DIR.
"""
import argparse
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    from .drug_features import ATOM_DIM, HAVE_RDKIT, build_drug_bank
    from .metrics import full_report
    from .model import DrugResponseNet
    from .preprocess import OmicsPreprocessor, build_sample_matrix, load_drugs, load_omics, load_response
except ImportError:  # executed as a top-level script inside SageMaker
    from drug_features import ATOM_DIM, HAVE_RDKIT, build_drug_bank
    from metrics import full_report
    from model import DrugResponseNet
    from preprocess import OmicsPreprocessor, build_sample_matrix, load_drugs, load_omics, load_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data/canonical"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "models/latest"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "results"))
    p.add_argument("--target", default="ln_ic50")
    p.add_argument("--split", choices=["cell_line", "random"], default="cell_line",
                   help="cell_line = unseen cell lines in val/test (harder, more realistic)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--modality-dropout", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--top-k-expression", type=int, default=1000)
    p.add_argument("--top-k-cnv", type=int, default=500)
    p.add_argument("--top-k-mutation", type=int, default=300)
    p.add_argument("--top-k-proteomics", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_pairs(resp, mode, seed):
    idx = np.arange(len(resp))
    if mode == "random":
        train, rest = train_test_split(idx, test_size=0.3, random_state=seed)
        val, test = train_test_split(rest, test_size=0.5, random_state=seed)
        return train, val, test
    groups = resp["cell_line_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train, rest = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    v, t = next(gss2.split(rest, groups=groups[rest]))
    return train, rest[v], rest[t]


class Tensors:
    """All training data held as tensors; batches are gathered by index."""

    def __init__(self, feats, mask, drug_bank, sample_idx, drug_idx, y, device):
        self.feats = {m: torch.tensor(v, device=device) for m, v in feats.items()}
        self.mask = torch.tensor(mask, device=device)
        self.dx, self.da, self.dm = (torch.tensor(a, device=device) for a in drug_bank)
        self.sample_idx = torch.tensor(sample_idx, device=device)
        self.drug_idx = torch.tensor(drug_idx, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    def batch(self, rows):
        s, d = self.sample_idx[rows], self.drug_idx[rows]
        return {m: v[s] for m, v in self.feats.items()}, self.mask[s], self.dx[d], self.da[d], self.dm[d], self.y[rows]


def apply_modality_dropout(mask, p):
    if p <= 0:
        return mask
    dropped = mask & (torch.rand(mask.shape, device=mask.device) >= p)
    empty = ~dropped.any(1)
    dropped[empty] = mask[empty]  # always keep at least one modality
    return dropped


@torch.no_grad()
def predict_rows(model, data, rows, batch_size=1024):
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        omics, mask, dx, da, dm, _ = data.batch(rows[i:i + batch_size])
        out.append(model(omics, mask, dx, da, dm))
    return torch.cat(out).cpu().numpy() if out else np.array([])


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s rdkit=%s data_dir=%s", device, HAVE_RDKIT, args.data_dir)

    omics = load_omics(args.data_dir)
    resp = load_response(args.data_dir, args.target)
    drugs = load_drugs(args.data_dir)

    with_omics = set().union(*[set(df.index) for df in omics.values()])
    resp = resp[resp["cell_line_id"].isin(with_omics) & resp["drug_id"].isin(set(drugs["drug_id"]))]
    resp = resp.reset_index(drop=True)
    log.info("pairs=%d cell_lines=%d drugs=%d modalities=%s", len(resp), resp["cell_line_id"].nunique(),
             resp["drug_id"].nunique(), list(omics))

    train_i, val_i, test_i = split_pairs(resp, args.split, args.seed)
    train_cells = resp.loc[train_i, "cell_line_id"].unique()

    preproc = OmicsPreprocessor({
        "expression": args.top_k_expression, "cnv": args.top_k_cnv,
        "mutation": args.top_k_mutation, "proteomics": args.top_k_proteomics,
    }).fit(omics, train_cells)
    log.info("selected features: %s", {m: len(f) for m, f in preproc.features.items()})

    sample_ids = sorted(resp["cell_line_id"].unique())
    sample_pos = {s: i for i, s in enumerate(sample_ids)}
    feats, mask = build_sample_matrix(preproc, omics, sample_ids)

    drugs = drugs[drugs["drug_id"].isin(set(resp["drug_id"]))].reset_index(drop=True)
    drug_pos = {d: i for i, d in enumerate(drugs["drug_id"])}
    drug_bank = build_drug_bank(drugs["smiles"].tolist())

    y = resp[args.target].values.astype(np.float32)
    y_mean, y_std = float(y[train_i].mean()), float(y[train_i].std() or 1.0)
    data = Tensors(feats, mask, drug_bank,
                   resp["cell_line_id"].map(sample_pos).values, resp["drug_id"].map(drug_pos).values,
                   (y - y_mean) / y_std, device)

    model_cfg = {
        "modality_dims": {m: len(preproc.features[m]) for m in preproc.modalities},
        "atom_dim": ATOM_DIM, "d_model": args.d_model, "n_heads": args.n_heads,
        "n_layers": args.n_layers, "dropout": args.dropout,
    }
    model = DrugResponseNet(**model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.SmoothL1Loss()
    log.info("parameters=%d", sum(p.numel() for p in model.parameters()))

    best_rmse, best_state, bad_epochs = float("inf"), None, 0
    train_rows = torch.tensor(train_i, device=device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = train_rows[torch.randperm(len(train_rows), device=device)]
        for i in range(0, len(perm), args.batch_size):
            omics_b, mask_b, dx, da, dm, yb = data.batch(perm[i:i + args.batch_size])
            mask_b = apply_modality_dropout(mask_b, args.modality_dropout)
            loss = loss_fn(model(omics_b, mask_b, dx, da, dm), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(yb)
        val_pred = predict_rows(model, data, val_i) * y_std + y_mean
        val_rmse = float(np.sqrt(np.mean((val_pred - y[val_i]) ** 2)))
        val_r = float(np.corrcoef(val_pred, y[val_i])[0, 1]) if len(val_i) > 2 else float("nan")
        sched.step(val_rmse)
        # Parsed by SageMaker metric_definitions -> CloudWatch metrics.
        log.info("epoch=%d train_loss=%.4f val_rmse=%.4f val_pearson=%.4f time=%.1fs",
                 epoch, total / len(train_i), val_rmse, val_r, time.time() - t0)
        if val_rmse < best_rmse - 1e-4:
            best_rmse, bad_epochs = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    thresholds = resp.loc[train_i].groupby("drug_id")[args.target].median().to_dict()

    results = {}
    for name, rows in [("val", val_i), ("test", test_i)]:
        pred = predict_rows(model, data, rows) * y_std + y_mean
        report, per_drug = full_report(y[rows], pred, resp.loc[rows, "drug_id"].values, thresholds)
        results[name] = report
        if name == "test":
            out = resp.loc[rows, ["cell_line_id", "drug_id", "drug_name"]].copy()
            out["true_" + args.target], out["pred_" + args.target] = y[rows], pred
            out.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
            results["test_per_drug_pearson"] = per_drug
    log.info("test metrics: %s", json.dumps(results["test"]))
    for k in ("rmse", "pearson", "roc_auc"):
        if k in results["test"]:
            log.info("test_%s=%.4f", k, results["test"][k])

    # ---- model artefacts (packed into model.tar.gz by SageMaker) ----
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))
    preproc.save(os.path.join(args.model_dir, "preprocessor.json"))
    version = time.strftime("v%Y%m%d-%H%M%S")
    meta = {
        "version": version, "target": args.target, "y_mean": y_mean, "y_std": y_std,
        "model": model_cfg, "split": args.split, "rdkit": HAVE_RDKIT,
        "n_train_pairs": int(len(train_i)), "n_cell_lines": int(len(sample_ids)),
        "hyperparameters": vars(args),
    }
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    drug_records = [
        {"drug_id": r.drug_id, "drug_name": r.drug_name, "smiles": r.smiles,
         "threshold": float(thresholds.get(r.drug_id, y_mean))}
        for r in drugs.itertuples()
    ]
    with open(os.path.join(args.model_dir, "drugs.json"), "w") as f:
        json.dump(drug_records, f, indent=2)
    for d in (args.model_dir, args.output_dir):
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(dict(results, version=version), f, indent=2)
    log.info("saved model %s to %s", version, args.model_dir)


if __name__ == "__main__":
    main()

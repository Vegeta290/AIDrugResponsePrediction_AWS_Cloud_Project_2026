"""Load a trained model directory and produce explained predictions for one sample.

Shared by the SageMaker endpoint (inference.py), the FastAPI backend (local
inference mode) and the TCGA validation script.
"""
import json
import math
import os

import numpy as np
import torch

try:
    from .drug_features import build_drug_bank
    from .model import DrugResponseNet
    from .preprocess import OmicsPreprocessor
except ImportError:
    from drug_features import build_drug_bank
    from model import DrugResponseNet
    from preprocess import OmicsPreprocessor


class DrugResponsePredictor:
    def __init__(self, model_dir, device="cpu"):
        self.device = torch.device(device)
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(model_dir, "drugs.json")) as f:
            self.drugs = json.load(f)
        metrics_path = os.path.join(model_dir, "metrics.json")
        self.metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        self.preproc = OmicsPreprocessor.load(os.path.join(model_dir, "preprocessor.json"))
        self.model = DrugResponseNet(**self.config["model"]).to(self.device)
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.drug_index = {d["drug_id"]: i for i, d in enumerate(self.drugs)}
        bank = build_drug_bank([d["smiles"] for d in self.drugs])
        self.bank = tuple(torch.tensor(a, device=self.device) for a in bank)

    @property
    def version(self):
        return self.config.get("version", "unknown")

    def list_drugs(self):
        return [{"drug_id": d["drug_id"], "drug_name": d["drug_name"]} for d in self.drugs]

    def info(self):
        return {
            "version": self.version,
            "target": self.config.get("target"),
            "modalities": self.preproc.modalities,
            "n_features": {m: len(f) for m, f in self.preproc.features.items()},
            "n_drugs": len(self.drugs),
            "metrics": self.metrics,
        }

    def predict(self, sample, drug_ids=None, explain=True, top_k=10):
        """
        sample:   {modality: {feature: value}} with raw (unscaled) values in the
                  same units as the training data (e.g. log2(TPM+1) expression).
        drug_ids: drugs to score; None = every drug known to the model.
        Returns a dict with per-drug predictions sorted from most to least sensitive.
        """
        feats, mask_d, coverage = self.preproc.transform_sample(sample)
        if not any(mask_d.values()):
            raise ValueError("None of the sample's features match the model's feature set")
        ids = [d for d in (drug_ids or [d["drug_id"] for d in self.drugs]) if str(d) in self.drug_index]
        if not ids:
            raise ValueError("No known drug ids requested")
        ids = [str(d) for d in ids]
        n = len(ids)
        mods = self.preproc.modalities

        omics = {m: torch.tensor(np.repeat(feats[m][None, :], n, 0), device=self.device) for m in mods}
        if explain:
            for t in omics.values():
                t.requires_grad_(True)
        mask = torch.tensor([[mask_d[m] for m in mods]] * n, device=self.device)
        di = torch.tensor([self.drug_index[d] for d in ids], device=self.device)
        dx, da, dm = (b[di] for b in self.bank)

        with torch.set_grad_enabled(explain):
            pred, attn = self.model(omics, mask, dx, da, dm, return_attention=True)
            if explain:
                # Rows are independent, so the gradient of the sum gives per-drug gradients.
                pred.sum().backward()

        y = (pred.detach().cpu().numpy() * self.config["y_std"]) + self.config["y_mean"]
        attn = attn.detach().cpu().numpy()
        # CLS also attends to itself; report each input's share of the attention on inputs.
        attn = attn / np.clip(attn.sum(1, keepdims=True), 1e-9, None)
        results = []
        for row, drug_id in enumerate(ids):
            drug = self.drugs[self.drug_index[drug_id]]
            ln_ic50 = float(y[row])
            item = {
                "drug_id": drug_id,
                "drug_name": drug["drug_name"],
                "predicted_ln_ic50": round(ln_ic50, 4),
                "predicted_ic50_um": round(math.exp(ln_ic50), 4),
                "sensitivity_score": round(drug["threshold"] - ln_ic50, 4),
                "sensitivity": "Sensitive" if ln_ic50 < drug["threshold"] else "Resistant",
            }
            if explain:
                item["modality_attention"] = {
                    name: round(float(w), 4) for name, w in zip(mods + ["drug"], attn[row])
                }
                item["top_features"] = self._top_features(omics, feats, mask_d, row, top_k)
            results.append(item)
        results.sort(key=lambda r: r["predicted_ln_ic50"])
        return {
            "model_version": self.version,
            "coverage": {m: round(c, 4) for m, c in coverage.items()},
            "predictions": results,
        }

    def _top_features(self, omics, feats, mask_d, row, top_k):
        """Gradient x input attribution (positive = pushes towards resistance)."""
        scored = []
        for m, t in omics.items():
            if not mask_d[m] or t.grad is None:
                continue
            attr = t.grad[row].cpu().numpy() * feats[m]
            for j in np.argsort(-np.abs(attr))[:top_k]:
                if attr[j] != 0:
                    scored.append((m, self.preproc.features[m][j], float(attr[j])))
        scored.sort(key=lambda s: -abs(s[2]))
        return [
            {"modality": m, "feature": f, "attribution": round(a, 5),
             "direction": "resistance" if a > 0 else "sensitivity"}
            for m, f, a in scored[:top_k]
        ]

"""Multi-omics preprocessing: loading, cleaning, feature selection and normalisation.

Canonical input layout (one directory, produced by datasets/scripts/*):
    expression.csv   rows = samples (first column = sample id), columns = genes
    cnv.csv          same layout
    mutation.csv     same layout, binary 0/1
    proteomics.csv   same layout (optional)
    response.csv     cell_line_id, drug_id, drug_name, ln_ic50, auc
    drugs.csv        drug_id, drug_name, smiles

The fitted preprocessor is serialised to JSON so the exact same transform is
applied at training time, inside the SageMaker endpoint and in the backend.
"""
import json
import os

import numpy as np
import pandas as pd

MODALITIES = ["expression", "cnv", "mutation", "proteomics"]
DEFAULT_TOP_K = {"expression": 1000, "cnv": 500, "mutation": 300, "proteomics": 200}
MAX_MISSING_FRACTION = 0.2


def load_omics(data_dir):
    """Load every available omics matrix. Returns {modality: DataFrame}."""
    omics = {}
    for m in MODALITIES:
        path = os.path.join(data_dir, m + ".csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str)
        df = df.loc[~df.index.duplicated()]
        df = df.apply(pd.to_numeric, errors="coerce")
        omics[m] = df
    if not omics:
        raise FileNotFoundError("No omics files found in %s" % data_dir)
    return omics


def load_response(data_dir, target="ln_ic50"):
    resp = pd.read_csv(os.path.join(data_dir, "response.csv"))
    resp["cell_line_id"] = resp["cell_line_id"].astype(str)
    resp["drug_id"] = resp["drug_id"].astype(str)
    resp = resp.dropna(subset=[target])
    # Replicate screens of the same pair are averaged.
    resp = resp.groupby(["cell_line_id", "drug_id"], as_index=False).agg(
        {"drug_name": "first", target: "mean"}
    )
    return resp


def load_drugs(data_dir):
    drugs = pd.read_csv(os.path.join(data_dir, "drugs.csv"))
    drugs["drug_id"] = drugs["drug_id"].astype(str)
    drugs["smiles"] = drugs["smiles"].fillna("")
    return drugs.drop_duplicates("drug_id")


class OmicsPreprocessor:
    """Per-modality missing-value filtering, variance-based selection and z-scoring."""

    def __init__(self, top_k=None):
        self.top_k = dict(DEFAULT_TOP_K, **(top_k or {}))
        self.features = {}
        self.mean = {}
        self.std = {}

    @property
    def modalities(self):
        return [m for m in MODALITIES if m in self.features]

    def fit(self, omics, train_ids):
        train_ids = set(train_ids)
        for m, df in omics.items():
            sub = df.loc[df.index.isin(train_ids)]
            if len(sub) < 5:
                continue
            sub = sub.loc[:, sub.isna().mean() <= MAX_MISSING_FRACTION]
            if m == "mutation":
                # Ignore genes mutated in <1% of training samples.
                sub = sub.loc[:, sub.fillna(0).mean() >= 0.01]
            var = sub.var(skipna=True).fillna(0)
            var = var[var > 0]
            keep = var.sort_values(ascending=False).index[: self.top_k.get(m, 500)]
            sub = sub[keep]
            mean = sub.mean(skipna=True)
            std = sub.std(skipna=True).replace(0, 1.0).fillna(1.0)
            self.features[m] = list(map(str, keep))
            self.mean[m] = mean.values.astype(np.float32)
            self.std[m] = std.values.astype(np.float32)
        return self

    def transform_frame(self, df, modality):
        """DataFrame (samples x features) -> z-scored float32 array; missing values -> 0 (train mean)."""
        aligned = df.reindex(columns=self.features[modality])
        z = (aligned.values.astype(np.float32) - self.mean[modality]) / self.std[modality]
        return np.nan_to_num(z, nan=0.0)

    def transform_sample(self, sample):
        """Transform a single sample given as {modality: {feature: value}}.

        Returns (features {modality: array[d]}, mask {modality: bool}, coverage {modality: fraction}).
        """
        feats, mask, coverage = {}, {}, {}
        for m in self.modalities:
            values = (sample or {}).get(m) or {}
            idx = {f: i for i, f in enumerate(self.features[m])}
            raw = np.full(len(idx), np.nan, dtype=np.float32)
            hits = 0
            for f, v in values.items():
                i = idx.get(str(f))
                if i is not None and v is not None and not (isinstance(v, float) and np.isnan(v)):
                    raw[i] = float(v)
                    hits += 1
            z = np.nan_to_num((raw - self.mean[m]) / self.std[m], nan=0.0)
            feats[m] = z.astype(np.float32)
            mask[m] = hits > 0
            coverage[m] = hits / max(len(idx), 1)
        return feats, mask, coverage

    def to_dict(self):
        return {
            "top_k": self.top_k,
            "features": self.features,
            "mean": {m: v.tolist() for m, v in self.mean.items()},
            "std": {m: v.tolist() for m, v in self.std.items()},
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(d.get("top_k"))
        obj.features = d["features"]
        obj.mean = {m: np.array(v, dtype=np.float32) for m, v in d["mean"].items()}
        obj.std = {m: np.array(v, dtype=np.float32) for m, v in d["std"].items()}
        return obj

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def build_sample_matrix(preproc, omics, sample_ids):
    """Return ({modality: array[n, d]}, mask array[n, M]) for the given ordered sample ids."""
    sample_ids = list(sample_ids)
    feats, masks = {}, []
    for m in preproc.modalities:
        df = omics[m]
        present = np.array([s in df.index for s in sample_ids])
        frame = df.reindex(sample_ids)
        feats[m] = preproc.transform_frame(frame, m)
        masks.append(present)
    return feats, np.stack(masks, axis=1)


def parse_long_csv(text):
    """Parse an uploaded sample file with columns modality,feature,value.

    An optional sample_id column allows several samples per file.
    Returns {sample_id: {modality: {feature: value}}}.
    """
    import io

    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"modality", "feature", "value"} - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: %s" % ", ".join(sorted(missing)))
    if "sample_id" not in df.columns:
        df["sample_id"] = "sample"
    df["modality"] = df["modality"].astype(str).str.strip().str.lower()
    unknown = set(df["modality"]) - set(MODALITIES)
    if unknown:
        raise ValueError("Unknown modality values: %s (allowed: %s)" % (sorted(unknown), MODALITIES))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    out = {}
    for (sid, m), grp in df.groupby(["sample_id", "modality"]):
        out.setdefault(str(sid), {})[m] = dict(zip(grp["feature"].astype(str), grp["value"].astype(float)))
    return out

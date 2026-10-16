"""Convert raw DepMap (CCLE) + GDSC downloads into the canonical training layout.

Downloads (see datasets/README.md):
  DepMap portal  https://depmap.org/portal/download/all/
    Model.csv                                        cell line metadata (ModelID <-> SangerModelID)
    OmicsExpressionProteinCodingGenesTPMLogp1.csv    RNA-seq log2(TPM+1)
    OmicsCNGene.csv                                  gene-level copy number
    OmicsSomaticMutationsMatrixDamaging.csv          damaging mutation matrix
    (optional) a proteomics / RPPA matrix
  GDSC           https://www.cancerrxgene.org/downloads/bulk_download
    GDSC2_fitted_dose_response_*.xlsx (or .csv)      LN_IC50 / AUC per cell line x drug
  Drug SMILES: run fetch_smiles.py afterwards (PubChem lookup by drug name).

File names change between DepMap releases; pass the paths explicitly if they differ.

    python datasets/scripts/prepare_depmap_gdsc.py --depmap-dir raw/depmap \
        --gdsc raw/gdsc/GDSC2_fitted_dose_response.xlsx --out data/canonical
"""
import argparse
import os
import re

import pandas as pd

META_COLS = {"SequencingID", "ModelID", "IsDefaultEntryForModel", "ModelConditionID",
             "IsDefaultEntryForMC", "ProfileID"}


def read_depmap_matrix(path):
    """Read a DepMap matrix into samples x genes with plain gene symbols as columns."""
    df = pd.read_csv(path, low_memory=False)
    if "ModelID" in df.columns:
        if "IsDefaultEntryForModel" in df.columns:
            df = df[df["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true", "1"])]
        df = df.set_index("ModelID")
    else:
        df = df.set_index(df.columns[0])
    df = df.drop(columns=[c for c in df.columns if c in META_COLS], errors="ignore")
    df.columns = [re.sub(r"\s*\(.*\)$", "", str(c)) for c in df.columns]  # "TP53 (7157)" -> "TP53"
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df.index = df.index.astype(str)
    df.index.name = "cell_line_id"
    return df.loc[~df.index.duplicated()].apply(pd.to_numeric, errors="coerce")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depmap-dir", required=True)
    p.add_argument("--gdsc", required=True, help="GDSC fitted dose response .xlsx or .csv")
    p.add_argument("--out", default="data/canonical")
    p.add_argument("--model-file", default="Model.csv")
    p.add_argument("--expression", default="OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    p.add_argument("--cnv", default="OmicsCNGene.csv")
    p.add_argument("--mutation", default="OmicsSomaticMutationsMatrixDamaging.csv")
    p.add_argument("--proteomics", default=None, help="optional proteomics/RPPA matrix")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = pd.read_csv(os.path.join(args.depmap_dir, args.model_file))
    sanger_to_model = models.dropna(subset=["SangerModelID"]).set_index("SangerModelID")["ModelID"].to_dict()
    cosmic_to_model = {}
    if "COSMICID" in models.columns:
        m = models.dropna(subset=["COSMICID"])
        cosmic_to_model = dict(zip(m["COSMICID"].astype(int), m["ModelID"]))

    for name in ("expression", "cnv", "mutation", "proteomics"):
        fname = getattr(args, name)
        if not fname:
            continue
        path = fname if os.path.isabs(fname) else os.path.join(args.depmap_dir, fname)
        if not os.path.exists(path):
            print("skip %s (not found: %s)" % (name, path))
            continue
        df = read_depmap_matrix(path)
        if name == "mutation":
            df = (df.fillna(0) > 0).astype(int)
        df.to_csv(os.path.join(args.out, name + ".csv"))
        print("%-11s %d samples x %d features" % (name, df.shape[0], df.shape[1]))

    gdsc = pd.read_excel(args.gdsc) if args.gdsc.endswith((".xlsx", ".xls")) else pd.read_csv(args.gdsc)
    gdsc.columns = [c.upper() for c in gdsc.columns]
    gdsc["cell_line_id"] = gdsc["SANGER_MODEL_ID"].map(sanger_to_model)
    if cosmic_to_model and "COSMIC_ID" in gdsc.columns:
        gdsc["cell_line_id"] = gdsc["cell_line_id"].fillna(gdsc["COSMIC_ID"].map(cosmic_to_model))
    before = len(gdsc)
    gdsc = gdsc.dropna(subset=["cell_line_id", "LN_IC50"])
    resp = gdsc.rename(columns={"DRUG_ID": "drug_id", "DRUG_NAME": "drug_name", "LN_IC50": "ln_ic50", "AUC": "auc"})
    resp = resp[["cell_line_id", "drug_id", "drug_name", "ln_ic50", "auc"]].drop_duplicates()
    resp.to_csv(os.path.join(args.out, "response.csv"), index=False)
    print("response    %d pairs (%d dropped without a DepMap match)" % (len(resp), before - len(gdsc)))

    drugs_path = os.path.join(args.out, "drugs.csv")
    if not os.path.exists(drugs_path):
        resp[["drug_id", "drug_name"]].drop_duplicates("drug_id").assign(smiles="").to_csv(drugs_path, index=False)
        print("drugs.csv written without SMILES -> run datasets/scripts/fetch_smiles.py next")

    meta_cols = [c for c in ["ModelID", "OncotreeLineage", "OncotreePrimaryDisease"] if c in models.columns]
    models[meta_cols].rename(columns={"ModelID": "cell_line_id"}).to_csv(
        os.path.join(args.out, "metadata.csv"), index=False)


if __name__ == "__main__":
    main()

"""Generate a small synthetic multi-omics + drug response dataset in the canonical layout.

The data has planted biology so the model has something real to learn, e.g.
EGFR inhibitors work better on EGFR-high lines, Olaparib on BRCA1-mutant lines,
and ABCB1 (drug efflux pump) expression causes resistance to Doxorubicin.

It exists so the full pipeline (training, API, dashboard, AWS deployment) can
be tested in minutes. Replace it with real DepMap/GDSC data for the final results:
see prepare_depmap_gdsc.py.

    python datasets/scripts/generate_synthetic.py --out data/canonical
"""
import argparse
import os

import numpy as np
import pandas as pd

DRUGS = [
    ("1006", "Erlotinib", "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"),
    ("1010", "Gefitinib", "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"),
    ("1558", "Lapatinib", "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl"),
    ("1079", "Imatinib", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"),
    ("51", "Dasatinib", "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO"),
    ("1085", "Sorafenib", "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F"),
    ("1017", "Olaparib", "C1CC1C(=O)N2CCN(CC2)C(=O)C3=C(C=CC(=C3)CC4=NNC(=O)C5=CC=CC=C54)F"),
    ("1073", "5-Fluorouracil", "C1=C(C(=O)NC(=O)N1)F"),
    ("1190", "Gemcitabine", "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F"),
    ("1008", "Methotrexate", "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O"),
    ("1012", "Vorinostat", "C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO"),
    ("133", "Doxorubicin", "CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O"),
]
KEY_GENES = ["EGFR", "ERBB2", "KRAS", "TP53", "BRCA1", "BRCA2", "ABCB1", "ABL1", "PTEN", "MYC",
             "PIK3CA", "BRAF", "TYMS", "DHFR", "HDAC1", "CDKN2A", "RB1", "SMAD4", "VEGFA", "KIT"]
LINEAGES = ["Lung", "Breast", "Colorectal", "Skin", "Blood", "Pancreas", "Ovary", "Brain"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/canonical")
    p.add_argument("--cells", type=int, default=500)
    p.add_argument("--genes", type=int, default=1500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--example-dir", nargs="*", default=["datasets", "src/frontend"],
                   help="where to write sample_patient.csv (an unseen sample for demos)")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    n, g, k = args.cells, args.genes, 10
    genes = KEY_GENES + ["GENE%04d" % i for i in range(g - len(KEY_GENES))]
    ids = ["ACH-%06d" % (i + 1) for i in range(n)]
    lineage = rng.choice(LINEAGES, n)
    lineage_effect = {l: rng.normal(0, 0.8, k) for l in LINEAGES}

    z = rng.normal(0, 1, (n, k)) + np.stack([lineage_effect[l] for l in lineage])
    w = rng.normal(0, 1, (k, g)) * (rng.random((k, g)) < 0.2)
    mut_w = rng.normal(0, 0.8, (k, g))
    expr = np.clip(5 + z @ w * 0.6 + rng.normal(0, 0.7, (n, g)), 0, None)
    cnv = 0.35 * (expr - expr.mean(0)) / expr.std(0) + rng.normal(0, 0.5, (n, g))
    mut = (rng.random((n, g)) < 1 / (1 + np.exp(-(z @ mut_w - 3.0)))).astype(int)

    gi = {name: i for i, name in enumerate(genes)}
    # Planted biology on key genes.
    expr[:, gi["EGFR"]] += np.where(np.isin(lineage, ["Lung", "Colorectal"]), 2.0, 0) + rng.normal(0, 1.2, n)
    expr[:, gi["ABCB1"]] += rng.exponential(1.5, n)
    expr[:, gi["ABL1"]] += np.where(lineage == "Blood", 2.5, 0) + rng.normal(0, 0.5, n)
    mut[:, gi["BRCA1"]] = rng.random(n) < np.where(np.isin(lineage, ["Breast", "Ovary"]), 0.35, 0.05)
    mut[:, gi["KRAS"]] = rng.random(n) < np.where(np.isin(lineage, ["Pancreas", "Colorectal"]), 0.6, 0.1)

    prot_genes = genes[:200]
    prot = expr[:, [gi[x] for x in prot_genes]] * 0.8 + rng.normal(0, 0.6, (n, len(prot_genes)))
    has_prot = rng.random(n) < 0.5  # proteomics only exists for a subset, as in DepMap

    def frame(values, cols, rows=ids):
        return pd.DataFrame(values, index=pd.Index(rows, name="cell_line_id"), columns=cols)

    frame(np.round(expr, 3), genes).to_csv(os.path.join(args.out, "expression.csv"))
    frame(np.round(cnv, 3), genes).to_csv(os.path.join(args.out, "cnv.csv"))
    frame(mut, genes).to_csv(os.path.join(args.out, "mutation.csv"))
    frame(np.round(prot[has_prot], 3), prot_genes, [i for i, h in zip(ids, has_prot) if h]).to_csv(
        os.path.join(args.out, "proteomics.csv"))

    ez = (expr - expr.mean(0)) / expr.std(0)
    rows = []
    for drug_id, name, _ in DRUGS:
        base = rng.normal(2.0, 1.0)
        effect = z @ rng.normal(0, 0.35, k)
        if name in ("Erlotinib", "Gefitinib", "Lapatinib"):
            effect -= 1.2 * ez[:, gi["EGFR"]] - 0.8 * mut[:, gi["KRAS"]]
        if name == "Lapatinib":
            effect -= 0.8 * ez[:, gi["ERBB2"]]
        if name in ("Imatinib", "Dasatinib"):
            effect -= 1.3 * ez[:, gi["ABL1"]]
        if name == "Olaparib":
            effect -= 2.0 * mut[:, gi["BRCA1"]]
        if name in ("Doxorubicin", "Vorinostat"):
            effect += 1.0 * ez[:, gi["ABCB1"]]
        if name in ("5-Fluorouracil", "Methotrexate"):
            effect += 0.8 * ez[:, gi["TYMS"]] + 0.5 * ez[:, gi["DHFR"]]
        ln_ic50 = base + effect + rng.normal(0, 0.45, n)
        for i in np.where(rng.random(n) < 0.85)[0]:
            rows.append((ids[i], drug_id, name, round(float(ln_ic50[i]), 4),
                         round(float(1 / (1 + np.exp(-(ln_ic50[i] - 2)))), 4)))
    resp = pd.DataFrame(rows, columns=["cell_line_id", "drug_id", "drug_name", "ln_ic50", "auc"])
    resp.to_csv(os.path.join(args.out, "response.csv"), index=False)
    pd.DataFrame(DRUGS, columns=["drug_id", "drug_name", "smiles"]).to_csv(
        os.path.join(args.out, "drugs.csv"), index=False)
    pd.DataFrame({"cell_line_id": ids, "lineage": lineage}).to_csv(
        os.path.join(args.out, "metadata.csv"), index=False)

    # Demo upload file: a new EGFR-high lung "patient" from the same process.
    zp = rng.normal(0, 1, k) + lineage_effect["Lung"]
    pe = np.clip(5 + zp @ w * 0.6 + rng.normal(0, 0.7, g), 0, None)
    pe[gi["EGFR"]] += 4.0
    pm = (rng.random(g) < 1 / (1 + np.exp(-(zp @ mut_w - 3.0)))).astype(int)
    example = pd.concat([
        pd.DataFrame({"modality": "expression", "feature": genes, "value": np.round(pe, 3)}),
        pd.DataFrame({"modality": "cnv", "feature": genes, "value": np.round(rng.normal(0, 0.5, g), 3)}),
        pd.DataFrame({"modality": "mutation", "feature": genes, "value": pm}),
    ])
    for d in args.example_dir:
        if os.path.isdir(d):
            example.to_csv(os.path.join(d, "sample_patient.csv"), index=False)

    print("wrote %d cell lines, %d genes, %d drugs, %d response pairs to %s"
          % (n, g, len(DRUGS), len(resp), args.out))


if __name__ == "__main__":
    # numpy + macOS Accelerate emits spurious matmul overflow warnings; the output is finite.
    with np.errstate(all="ignore"):
        main()

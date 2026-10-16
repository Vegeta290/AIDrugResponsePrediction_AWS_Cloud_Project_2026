# Datasets

All training code reads one **canonical layout** (directory `data/canonical/`):

| File | Content |
|---|---|
| `expression.csv` | rows = cell lines (`cell_line_id` first column), columns = genes, log2(TPM+1) |
| `cnv.csv` | same layout, gene-level copy number |
| `mutation.csv` | same layout, 0/1 damaging mutation |
| `proteomics.csv` | optional, same layout |
| `response.csv` | `cell_line_id, drug_id, drug_name, ln_ic50, auc` |
| `drugs.csv` | `drug_id, drug_name, smiles` |

## Option A: synthetic (for development and CI)
```bash
python datasets/scripts/generate_synthetic.py --out data/canonical
```

## Option B: real data (for the report)

| Dataset | Download | Used for |
|---|---|---|
| DepMap (CCLE) | https://depmap.org/portal/data_page/?tab=allData: `Model.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsCNGene.csv`, `OmicsSomaticMutationsMatrixDamaging.csv` | omics features |
| GDSC2 | https://www.cancerrxgene.org/downloads/bulk_download: `GDSC2_fitted_dose_response_*.xlsx` | IC50/AUC labels |
| PubChem | queried automatically by `fetch_smiles.py` | drug SMILES → molecular graphs |
| TCGA | https://portal.gdc.cancer.gov/ | patient validation (see `docs/AWS_DEPLOYMENT.md` §9) |

```bash
pip install openpyxl
python datasets/scripts/prepare_depmap_gdsc.py --depmap-dir raw/depmap --gdsc raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx
python datasets/scripts/fetch_smiles.py --drugs data/canonical/drugs.csv
```

Cell lines are matched between DepMap and GDSC through `Model.csv` (`SangerModelID` / `COSMICID`).
Raw files are not committed to git because of their size; `data/` and `raw/` are git-ignored.

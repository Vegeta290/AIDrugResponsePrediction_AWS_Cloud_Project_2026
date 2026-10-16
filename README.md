# AI-Driven Cloud Platform for Personalized Drug Response Prediction using Multi-Omics Data

BITE412L Cloud Computing · VIT Vellore · Abhijeet Sinha (23BIT0157), Sarthak Pathak (23BIT0187), Shriyans Pratihari (23BIT0219)

The platform predicts how a tumour or cell line responds to anti-cancer drugs, measured as ln(IC50). It uses
**gene expression, copy number, mutations and proteomics** together with each drug's **molecular graph**.
Every prediction comes with an explanation. The platform is deployed on AWS: S3, Lambda, SageMaker, EC2, RDS, IAM and CloudWatch.

## What is implemented

| Proposal item | Implementation |
|---|---|
| Multi-omics preprocessing (cleaning, normalisation, feature selection) | [src/ai_model/preprocess.py](src/ai_model/preprocess.py): drops features with >20% missing values, selects top-variance features per omics, z-scores them using the training split only |
| **GNN** for drug molecules | [src/ai_model/model.py](src/ai_model/model.py) `DrugGNN`: 3-layer residual GCN over RDKit atom graphs ([drug_features.py](src/ai_model/drug_features.py)) |
| **Transformer** multimodal fusion | `DrugResponseNet`: one token per omics plus a drug token, fused by a Transformer with a [CLS] head. Missing omics are masked out; modality dropout is used during training |
| IC50/AUC prediction on DepMap + GDSC | [src/ai_model/train.py](src/ai_model/train.py), with a cell-line-blind split by default (test cell lines are never seen in training) |
| Evaluation: accuracy, precision, recall, ROC-AUC | [src/ai_model/metrics.py](src/ai_model/metrics.py): RMSE, Pearson, Spearman, R², plus sensitive-vs-resistant classification metrics |
| Explainable AI | Attention weights per modality and gradient×input attribution per gene ([predictor.py](src/ai_model/predictor.py)) |
| TCGA patient validation | [src/ai_model/validate_tcga.py](src/ai_model/validate_tcga.py): responder vs. non-responder AUC |
| Backend REST API, RBAC | [src/backend](src/backend): FastAPI, JWT, admin/researcher/clinician roles, PostgreSQL (RDS), S3 storage |
| Dashboard | [src/frontend](src/frontend): upload, predict, explanation charts, history, model metrics, user admin |
| S3 + Lambda automatic preprocessing and retraining trigger | [src/aws/lambda_preprocess/handler.py](src/aws/lambda_preprocess/handler.py) |
| SageMaker training and endpoint | [src/aws/sagemaker](src/aws/sagemaker) |
| EC2, RDS, IAM, VPC, SSM Parameter Store, API Gateway, CloudWatch, CloudTrail | [src/aws/cloudformation/drugresponse-stack.yaml](src/aws/cloudformation/drugresponse-stack.yaml) |
| EKS (optional) | [src/aws/k8s/deployment.yaml](src/aws/k8s/deployment.yaml) |

## Quick start (local, about 5 minutes)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python datasets/scripts/generate_synthetic.py --out data/canonical    # demo data; see datasets/README.md for real data
PYTHONPATH=src python -m ai_model.train --data-dir data/canonical --model-dir models/latest --epochs 30
pytest -q tests

ADMIN_PASSWORD=adminpass123 PYTHONPATH=src uvicorn backend.app.main:app --port 8000
# http://localhost:8000  → admin@example.com / adminpass123 → upload datasets/sample_patient.csv → Predict
```

On the synthetic data, a 25-epoch run on unseen cell lines reached test Pearson r ≈ 0.84 and ROC-AUC ≈ 0.87.
The synthetic data contains planted biology (EGFR→erlotinib, BRCA1→olaparib, ABCB1→doxorubicin resistance), so these numbers only show that the pipeline works.
**Report the numbers you get from real DepMap/GDSC data.**

## Deploy to AWS

Follow **[docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md)**. In short:

```bash
# Free-tier mode (default, FREE_TIER_ONLY=true in src/aws/config.env): no SageMaker, model trained locally
src/aws/deploy_stack.sh                          # 1. infrastructure (CloudFormation)
src/aws/upload_data.sh data/canonical            # 2. data → S3
src/aws/ec2/deploy_app.sh                        # 3. API + dashboard on EC2 (uploads models/latest), behind API Gateway HTTPS

# Full mode (FREE_TIER_ONLY=false, SageMaker is billed)
python src/aws/sagemaker/launch_training.py      # train on SageMaker, then run deploy_app.sh
python src/aws/sagemaker/deploy_endpoint.py      # (optional) serverless SageMaker endpoint
```

## Repository layout

```
datasets/scripts/      synthetic generator, DepMap+GDSC converter, PubChem SMILES fetcher
src/ai_model/          preprocessing, GNN+Transformer model, training (SageMaker-ready), inference, TCGA validation
src/backend/           FastAPI app + Dockerfile
src/frontend/          dashboard (HTML/JS, Chart.js)
src/aws/               CloudFormation, Lambda, SageMaker scripts, EC2 deploy, k8s manifest
tests/                 pipeline, API and Lambda tests
docs/                  AWS deployment guide
```

## Sample upload format

```csv
modality,feature,value
expression,EGFR,9.12
cnv,EGFR,0.84
mutation,KRAS,1
proteomics,EGFR,2.3
```

Expression is log2(TPM+1) (DepMap units). Mutation values are 0/1. Add a `sample_id` column to put several samples in one file.

> Research prototype. It is not a medical device and must not be used for treatment decisions.

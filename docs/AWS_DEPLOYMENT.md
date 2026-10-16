# AWS Deployment Guide

This guide takes the project from a laptop to a running AWS deployment. The services match the
proposal: **S3, Lambda, SageMaker, EC2, RDS (PostgreSQL), IAM, CloudWatch**, plus SSM
Parameter Store, VPC, API Gateway and CloudTrail. QuickSight and EKS are optional extras.

```
 researchers ──HTTPS──► API Gateway ──► EC2 (Docker: FastAPI + dashboard + model) ──► RDS PostgreSQL
                                             │  ▲                                       (private subnet)
                         S3 uploads/processed│  │model.tar.gz
                                             ▼  │
 bulk CSV ─► S3 incoming/ ──trigger──► Lambda ──► SageMaker endpoint (optional) ─► S3 predictions/
 new data ─► S3 triggers/retrain.json ─► Lambda ──► SageMaker training job ─► S3 models/
                         CloudWatch logs, metrics, alarms, dashboard  ·  CloudTrail audit log
```

The default region is **ap-south-1 (Mumbai)**. You can change it in `src/aws/config.env`.

> **Free-tier mode (default).** `FREE_TIER_ONLY=true` in `src/aws/config.env` keeps everything in the AWS
> free tier: t3.micro EC2, db.t3.micro RDS, secrets in Parameter Store, CloudTrail without S3 data events,
> and **no SageMaker**. Train the model on your laptop (step 1) and deploy it with
> `src/aws/ec2/deploy_app.sh`, which then uploads `models/latest` automatically. Skip steps 5 and 7, and the
> Lambda ignores `triggers/retrain.json`. Set `FREE_TIER_ONLY=false` to use the SageMaker steps.

---

## 0. Prerequisites (one-time, about 20 min)

1. **AWS account.** Sign in as the root user once, enable MFA on it, and stop using it after that.
2. **IAM admin user for deployment** (IAM console → Users → Create user):
   - Name it `deployer` and attach the `AdministratorAccess` policy. The stack creates IAM roles, so deployment needs admin rights.
   - Under Security credentials, create an access key for **CLI** use.
3. **Install tools on your laptop:**
   ```bash
   # macOS
   brew install awscli
   aws configure            # paste the access key, region ap-south-1, output json
   aws sts get-caller-identity   # must print your account id
   ```
   You also need Python 3.9+ and Docker. Docker is only needed for local testing, because the EC2 instance builds the image itself.
4. **Python environment:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```
5. **Budget alarm (strongly recommended).** In Billing → Budgets → Create budget → "Monthly cost budget", set USD 20 and enter your email.
6. **Check your SageMaker quotas.** New accounts sometimes have a quota of 0 for some instance types. Open Service Quotas → Amazon SageMaker and check:
   - `ml.m5.xlarge for training job usage` ≥ 1
   - `Maximum memory size per serverless endpoint` / `serverless endpoint concurrency` > 0

   If either is 0, request an increase now. Approval can take a day.

## 1. Test locally first (about 5 min)

```bash
python datasets/scripts/generate_synthetic.py --out data/canonical
PYTHONPATH=src python -m ai_model.train --data-dir data/canonical --model-dir models/latest --epochs 30
pytest -q tests
ADMIN_PASSWORD=adminpass123 PYTHONPATH=src uvicorn backend.app.main:app --port 8000
# open http://localhost:8000, sign in as admin@example.com / adminpass123, upload datasets/sample_patient.csv
```

You can also run with a real PostgreSQL database: `docker compose up --build`.

## 2. Get the real datasets (DepMap + GDSC)

The synthetic data is enough to exercise the whole pipeline. For the results in your report, use real data:

1. From **DepMap** (https://depmap.org/portal/data_page/?tab=allData), download these files from the latest release:
   `Model.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `OmicsCNGene.csv`,
   `OmicsSomaticMutationsMatrixDamaging.csv`. Put them in `raw/depmap/`.
2. From **GDSC** (https://www.cancerrxgene.org/downloads/bulk_download), download
   `GDSC2_fitted_dose_response_*.xlsx`. Put it in `raw/gdsc/`.
3. Convert the files and fetch SMILES strings:
   ```bash
   pip install openpyxl
   python datasets/scripts/prepare_depmap_gdsc.py --depmap-dir raw/depmap \
          --gdsc raw/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx --out data/canonical
   python datasets/scripts/fetch_smiles.py --drugs data/canonical/drugs.csv
   ```
   File names change between releases. If a file is reported as "not found", pass its actual name with `--expression`, `--cnv` or `--mutation`.

The raw files are large (several GB), so do this on a machine with at least 16 GB RAM. A SageMaker Studio notebook also works.

## 3. Create the infrastructure (CloudFormation, about 15 min)

```bash
chmod +x src/aws/*.sh src/aws/ec2/*.sh
ALERT_EMAIL=you@vitstudent.ac.in src/aws/deploy_stack.sh
```

A single command creates the following resources from
[`src/aws/cloudformation/drugresponse-stack.yaml`](../src/aws/cloudformation/drugresponse-stack.yaml):

| Resource | What it is |
|---|---|
| **VPC** | 2 public subnets (EC2) and 2 private subnets (RDS, not reachable from the internet) |
| **S3** `drugresp-data-<acct>-<region>` | Versioned, encrypted, public access blocked, HTTPS only |
| **Lambda** `drugresp-preprocess` | Triggered on `incoming/*.csv` and `triggers/retrain.json` |
| **RDS PostgreSQL** `drugresp-db` | db.t3.micro, encrypted. Its password is generated into Parameter Store (`/drugresp/db-password`) |
| **EC2** `drugresp-api` | t3.medium with Docker installed. Managed through SSM, so there is no SSH key or open port 22 |
| **API Gateway** (HTTP API) | Free HTTPS URL in front of the EC2 server |
| **IAM roles** | Least-privilege roles for EC2, Lambda and SageMaker |
| **SSM Parameter Store** | SecureString parameters `/drugresp/db-password`, `jwt-secret`, `admin-email`, `admin-password` (free, unlike Secrets Manager) |
| **CloudWatch** | Log groups, 5 alarms → SNS email, and the `drugresp-overview` dashboard |
| **CloudTrail** | Audit trail of API calls and S3 data access |

If you set an alert email, check your inbox and **confirm the SNS subscription**.

**Verify in the console:** CloudFormation → `drugresp-stack` shows CREATE_COMPLETE, and the Outputs tab lists `ApiUrl`, `DataBucketName` and the other values.

## 4. Upload the training data to S3

```bash
src/aws/upload_data.sh data/canonical
```

**Verify:** S3 → bucket → `datasets/canonical/` contains the CSV files.

## 5. Train on SageMaker (about 10–30 min)

```bash
pip install -r src/aws/sagemaker/requirements.txt
export STACK=drugresp-stack AWS_REGION=ap-south-1
python src/aws/sagemaker/launch_training.py --epochs 40            # CPU, about $0.25/h
# real DepMap/GDSC data: --instance-type ml.g4dn.xlarge --spot      # GPU spot, much faster
```

The script does the following:
- Uploads `src/ai_model` to `s3://<bucket>/code/sourcedir.tar.gz`. The retraining Lambda reuses this copy.
- Starts a training job on the **PyTorch 2.1** container. RDKit and the other packages are installed from `src/ai_model/requirements.txt`.
- Streams the logs to your terminal and saves `model.tar.gz` to `s3://<bucket>/models/<job>/output/`.

**Verify:** SageMaker → Training jobs → the job's metrics charts show `val_rmse` falling and `val_pearson` rising. You can also check CloudWatch → Log groups → `/aws/sagemaker/TrainingJobs`. Copy the test metrics (Pearson, RMSE, ROC-AUC, precision, recall) from the end of the log into your report.

## 6. Deploy the API and dashboard on EC2 (about 10 min)

```bash
src/aws/ec2/deploy_app.sh          # INFERENCE_MODE=local: the model runs inside the EC2 container
```

The script packages the code, uploads it to S3 and runs
[`setup_app.sh`](../src/aws/ec2/setup_app.sh) on the instance through **SSM Run Command**. On the instance, the setup script:
- reads the DB password, JWT key and admin login from SSM Parameter Store
- writes a root-only env file
- builds the Docker image
- starts it with the `awslogs` log driver, so logs go to CloudWatch `/drugresp/api`
- downloads the latest trained model from S3

At the end it prints the dashboard URL. Read the admin login with:

```bash
aws ssm get-parameter --with-decryption --name /drugresp/admin-password --query Parameter.Value --output text
# user name: admin@drugresponse.local
```

Open the `ApiUrl` (https://xxxx.execute-api.ap-south-1.amazonaws.com/). Sign in as admin, then go to **Users** and create accounts for researchers and clinicians. Self sign-up is disabled in the cloud.

**Verify:** upload `datasets/sample_patient.csv` and run a prediction. Then check:
- RDS: Query Editor (or the History tab) shows the rows.
- S3: `uploads/` and `processed/` contain the files.
- CloudWatch: the `DrugResponse` namespace has `Predictions` and `PredictionLatency` metrics.

## 7. (Optional) SageMaker real-time endpoint

You can serve the model from a SageMaker endpoint instead of from the EC2 container. A **serverless** endpoint costs nothing while idle:

```bash
python src/aws/sagemaker/deploy_endpoint.py            # serverless, 4 GB, about 5–10 min
python src/aws/sagemaker/test_endpoint.py              # prints predictions for the example patient
INFERENCE_MODE=sagemaker src/aws/ec2/deploy_app.sh     # switch the API to the endpoint
```

The first request after the endpoint has been idle takes 20–60 seconds (cold start). For a live demo you may prefer `--instance-type ml.t2.medium`, which stays always on and is billed hourly. Delete it afterwards.

## 8. Lambda pipelines (automatic preprocessing and retraining)

**Bulk batch prediction.** Upload a CSV with columns `sample_id,modality,feature,value` to `incoming/`:
```bash
aws s3 cp my_cohort.csv s3://<bucket>/incoming/my_cohort.csv
```
The Lambda validates and cleans the file: it drops unknown modalities and NA values, binarises mutations and writes one JSON per sample to `processed/my_cohort/`. If the SageMaker endpoint from step 7 exists, it also scores every sample and writes `predictions/my_cohort.json`.

**Continuous retraining** (steps 10–11 of the workflow diagram):
```bash
src/aws/retrain.sh data/canonical      # sync new data and drop triggers/retrain.json
```
The Lambda then starts a new SageMaker training job. Review that job's metrics before you redeploy it with step 6 or step 7. This keeps a human in the loop, which matters in a clinical setting.

**Verify:** CloudWatch → `/aws/lambda/drugresp-preprocess` logs, and SageMaker → Training jobs shows a new job.

## 9. Validate on TCGA patients (objective 4)

1. In the GDC Data Portal (https://portal.gdc.cancer.gov/), pick a project such as TCGA-BRCA, TCGA-LUAD or TCGA-COAD and download:
   - **Gene expression** (STAR counts, TPM column). Convert it to log2(TPM+1) and gene symbols to match DepMap.
   - **Masked somatic mutations** (MAF) → a 0/1 patient × gene matrix.
   - **Gene-level copy number** (optional).
   - **Clinical drug data** (the `drug` / treatment tables with `treatment_or_therapy` and `response`).
2. Save them in the canonical layout in `data/tcga/` with rows = patient barcodes (`TCGA-XX-XXXX`), plus
   `clinical_response.csv` (`patient_id,drug_name,response`).
3. Run:
   ```bash
   aws s3 cp s3://<bucket>/models/<job>/output/model.tar.gz . && mkdir -p models/aws && tar -xzf model.tar.gz -C models/aws
   PYTHONPATH=src python -m ai_model.validate_tcga --model-dir models/aws --tcga-dir data/tcga
   ```
   The script reports a per-drug ROC-AUC (responders vs. non-responders) and a pooled AUC. It re-centres tumour data onto the cell-line distribution first, because cell lines and tumours differ systematically. Expect modest AUCs (around 0.55–0.70). Cell-line-to-patient transfer is hard, and the literature reports similar numbers, so present this honestly in the report.

## 10. (Optional) Amazon QuickSight analytics

The API creates two reporting views in RDS: `v_prediction_report` and `v_drug_summary`.
1. Sign up for QuickSight. Enterprise edition is required to reach a private RDS database, and there is a 30-day free trial.
2. Manage QuickSight → Security & permissions → VPC connections → Add. Choose the `drugresp-vpc`, both private subnets and a security group.
3. Add a rule to the RDS security group allowing port 5432 from that QuickSight security group.
4. Datasets → New → RDS → choose `drugresp-db` and database `drugresponse`. Use the master username `drugadmin` and the password from `/drugresp/db-password`, then select `v_prediction_report`.
5. Build visuals, for example "% sensitive by drug" (bar), "predictions over time" (line) and "IC50 distribution by cancer type" (box).

## 11. (Optional) Amazon EKS instead of a single EC2 instance

```bash
aws ecr create-repository --repository-name drugresponse-api
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker build --platform linux/amd64 -f src/backend/Dockerfile -t <acct>.dkr.ecr.<region>.amazonaws.com/drugresponse-api:latest .
docker push <acct>.dkr.ecr.<region>.amazonaws.com/drugresponse-api:latest
eksctl create cluster --name drugresp --region <region> --nodes 2 --node-type t3.medium   # about 15 min
kubectl create secret generic drugresponse-env --from-env-file=app.env
kubectl apply -f src/aws/k8s/deployment.yaml
```

EKS costs about $0.10/h for the control plane alone, plus the nodes. Treat it as a demo and run `eksctl delete cluster` afterwards.

## 12. IAM for the team

Create one IAM user per team member instead of sharing keys:
- `Abhijeet` (AI/data): `AmazonSageMakerFullAccess` and S3 access to the project bucket.
- `Sarthak` (cloud/backend): `AdministratorAccess`, or PowerUser plus IAM role management.
- `Shriyans` (frontend/integration): `AmazonSSMFullAccess`, `CloudWatchReadOnlyAccess` and QuickSight.

Dashboard users are separate from IAM users. The application manages them with roles: admin manages users, researchers and clinicians see only their own samples.

## 13. Monitoring checklist for the demo

- CloudWatch → Dashboards → `drugresp-overview`: EC2 CPU, predictions and latency, Lambda, RDS, SageMaker.
- CloudWatch → Alarms: 5 alarms, all in the OK state.
- CloudWatch → Logs Insights on `/drugresp/api`:
  `fields @timestamp, msg, path, status, ms | filter ispresent(path) | sort @timestamp desc`
- CloudTrail → Event history: who changed what.

## 14. Costs and cleanup

The figures below are rough on-demand prices in ap-south-1. Check the AWS Pricing Calculator for current rates.

| Item | Approx. cost |
|---|---|
| EC2 t3.medium (24×7) + 30 GB gp3 + Elastic IP | ~$1.3/day |
| RDS db.t3.micro + 20 GB | ~$0.6/day |
| SageMaker training ml.m5.xlarge | ~$0.25 per hour of training |
| SageMaker serverless endpoint | per request, $0 when idle |
| S3, Lambda, API Gateway, CloudWatch, Parameter Store | under $1/month at this scale |

**Save money between sessions:** stop the EC2 instance and the RDS instance from the console. Note that RDS automatically restarts after 7 days.

**Delete everything after the evaluation:**
```bash
aws sagemaker delete-endpoint --endpoint-name drugresp-endpoint      # if created
aws cloudformation delete-stack --stack-name drugresp-stack
# Retained on purpose (empty and delete manually): the data and trail S3 buckets,
# the RDS final snapshot, the artifacts bucket, the sagemaker-<region>-<acct> bucket.
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ResourceLimitExceeded` when training | Request a quota increase (step 0.6) or use `--instance-type ml.m5.large` |
| `deploy_app.sh` fails with "No completed training job" | Run step 5 first, or set `MODEL_S3_URI=s3://.../model.tar.gz` |
| Dashboard shows 503 "Model is not loaded" | `aws logs tail /drugresp/api --follow` and look for the model download error |
| API returns 500 on the first request | Check that RDS is "Available". The container also retries on restart |
| SSM send-command: instance not found | Wait 2–3 minutes after stack creation for the SSM agent to register |
| Lambda never fires | The key must start with `incoming/` and end with `.csv` (or be `triggers/retrain.json`) |
| Serverless endpoint times out | It is a cold start, so retry. Or raise `--memory-mb 6144` |

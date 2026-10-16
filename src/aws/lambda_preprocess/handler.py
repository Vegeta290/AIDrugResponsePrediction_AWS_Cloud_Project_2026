"""AWS Lambda: S3-triggered preprocessing, batch inference and retraining.

Only boto3 + standard library, so no Lambda layer is needed.

Triggers (configured in the CloudFormation template):
  s3://<bucket>/incoming/<name>.csv
      Researchers drop bulk multi-omics files here (columns sample_id,modality,feature,value).
      The Lambda validates and cleans the file, writes one JSON per sample to processed/,
      calls the SageMaker endpoint for every sample and writes predictions/<name>.json.
  s3://<bucket>/triggers/retrain.json
      Starts a SageMaker training job on datasets/canonical/ (continuous retraining loop).
      Optional JSON body: {"hyperparameters": {"epochs": "60"}, "instance_type": "ml.m5.2xlarge"}
"""
import csv
import io
import json
import logging
import math
import os
import time
import urllib.parse

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

MODALITIES = {"expression", "cnv", "mutation", "proteomics"}
s3 = boto3.client("s3")
sm = boto3.client("sagemaker")
smr = boto3.client("sagemaker-runtime")
cw = boto3.client("cloudwatch")


def clean_rows(text):
    """Validate and normalise rows. Returns ({sample: {modality: {feature: value}}}, errors)."""
    samples, errors = {}, []
    reader = csv.DictReader(io.StringIO(text))
    fields = {(f or "").strip().lower() for f in (reader.fieldnames or [])}
    missing = {"modality", "feature", "value"} - fields
    if missing:
        return {}, ["missing columns: %s" % ", ".join(sorted(missing))]
    for line, row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        modality = row["modality"].lower()
        if modality not in MODALITIES:
            errors.append("line %d: unknown modality %r" % (line, row["modality"]))
            continue
        try:
            value = float(row["value"])
        except ValueError:
            continue  # blank / NA values are treated as missing
        if math.isnan(value) or math.isinf(value) or not row["feature"]:
            continue
        if modality == "mutation":
            value = 1.0 if value > 0 else 0.0
        sample = row.get("sample_id") or "sample"
        samples.setdefault(sample, {}).setdefault(modality, {})[row["feature"]] = value
    return samples, errors[:50]


def handle_incoming(bucket, key):
    name = os.path.splitext(os.path.basename(key))[0]
    text = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8-sig")
    samples, errors = clean_rows(text)
    report = {"source": key, "samples": len(samples), "errors": errors, "predictions": {}}

    for sid, sample in samples.items():
        s3.put_object(Bucket=bucket, Key="processed/%s/%s.json" % (name, sid),
                      Body=json.dumps(sample).encode(), ServerSideEncryption="AES256")

    endpoint = os.environ.get("ENDPOINT_NAME") if sagemaker_enabled() else None
    if endpoint and samples:
        for sid, sample in samples.items():
            try:
                resp = smr.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                                           Body=json.dumps({"sample": sample, "explain": True, "top_k": 5}))
                report["predictions"][sid] = json.loads(resp["Body"].read())
            except Exception as e:  # keep processing the other samples
                log.exception("inference failed for %s", sid)
                report["predictions"][sid] = {"error": str(e)}
    s3.put_object(Bucket=bucket, Key="predictions/%s.json" % name, Body=json.dumps(report).encode(),
                  ServerSideEncryption="AES256", ContentType="application/json")
    cw.put_metric_data(Namespace=os.environ.get("METRICS_NAMESPACE", "DrugResponse"), MetricData=[
        {"MetricName": "BatchSamplesProcessed", "Value": len(samples), "Unit": "Count"},
        {"MetricName": "BatchValidationErrors", "Value": len(errors), "Unit": "Count"},
    ])
    return {"samples": len(samples), "errors": len(errors)}


def sagemaker_enabled():
    # ENABLE_SAGEMAKER=false (stack parameter FreeTierOnly=true) keeps the pipeline in the AWS free tier.
    return os.environ.get("ENABLE_SAGEMAKER", "true").lower() != "false"


def handle_retrain(bucket, key):
    if not sagemaker_enabled():
        log.info("retraining skipped: SageMaker is disabled (FreeTierOnly); train locally and redeploy")
        return {"training_job": None, "skipped": "sagemaker disabled"}
    try:
        cfg = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read() or b"{}")
    except ValueError:
        cfg = {}
    region = os.environ["AWS_REGION"]
    job = "drugresp-%s" % time.strftime("%Y%m%d-%H%M%S")
    hp = {
        # Script-mode settings understood by the SageMaker PyTorch training container.
        "sagemaker_program": "train.py",
        "sagemaker_submit_directory": "s3://%s/code/sourcedir.tar.gz" % bucket,
        "sagemaker_region": region,
        "epochs": "40",
    }
    hp.update({k: str(v) for k, v in (cfg.get("hyperparameters") or {}).items()})
    sm.create_training_job(
        TrainingJobName=job,
        RoleArn=os.environ["SAGEMAKER_ROLE_ARN"],
        AlgorithmSpecification={
            "TrainingImage": os.environ["TRAINING_IMAGE"], "TrainingInputMode": "File",
            "MetricDefinitions": [
                {"Name": "val_rmse", "Regex": r"val_rmse=([0-9\.]+)"},
                {"Name": "val_pearson", "Regex": r"val_pearson=([-0-9\.]+)"},
                {"Name": "test_pearson", "Regex": r"test_pearson=([-0-9\.]+)"},
            ],
        },
        HyperParameters={k: json.dumps(v) for k, v in hp.items()},
        InputDataConfig=[{
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3DataDistributionType": "FullyReplicated",
                                            "S3Uri": "s3://%s/datasets/canonical/" % bucket}},
        }],
        OutputDataConfig={"S3OutputPath": "s3://%s/models/" % bucket},
        ResourceConfig={"InstanceType": cfg.get("instance_type", os.environ.get("TRAINING_INSTANCE", "ml.m5.xlarge")),
                        "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 3 * 3600},
        Tags=[{"Key": "project", "Value": "drugresponse"}],
    )
    log.info("started training job %s", job)
    return {"training_job": job}


def lambda_handler(event, context):
    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        log.info("event for s3://%s/%s", bucket, key)
        if key.startswith("incoming/") and key.lower().endswith(".csv"):
            results.append(handle_incoming(bucket, key))
        elif key.startswith("triggers/") and key.endswith("retrain.json"):
            results.append(handle_retrain(bucket, key))
        else:
            log.info("ignored key %s", key)
    return {"results": results}

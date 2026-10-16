"""Step 3: train the model as an Amazon SageMaker training job.

    python src/aws/sagemaker/launch_training.py --epochs 40            # CPU ml.m5.xlarge
    python src/aws/sagemaker/launch_training.py --instance-type ml.g4dn.xlarge --spot

Training logs and metrics (val_rmse, val_pearson, test_pearson) appear in
CloudWatch under /aws/sagemaker/TrainingJobs. The trained model.tar.gz lands
in s3://<bucket>/models/<job-name>/output/.
"""
import argparse
import os

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

from common import FRAMEWORK_VERSION, PY_VERSION, save_state, stack_outputs, upload_source

METRICS = [
    {"Name": "train_loss", "Regex": r"train_loss=([0-9\.]+)"},
    {"Name": "val_rmse", "Regex": r"val_rmse=([0-9\.]+)"},
    {"Name": "val_pearson", "Regex": r"val_pearson=([-0-9\.]+)"},
    {"Name": "test_rmse", "Regex": r"test_rmse=([0-9\.]+)"},
    {"Name": "test_pearson", "Regex": r"test_pearson=([-0-9\.]+)"},
    {"Name": "test_roc_auc", "Regex": r"test_roc_auc=([0-9\.]+)"},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stack", default=os.environ.get("STACK", "drugresp-stack"))
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-southeast-2"))
    p.add_argument("--instance-type", default="ml.m5.xlarge")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--split", default="cell_line", choices=["cell_line", "random"])
    p.add_argument("--spot", action="store_true", help="managed spot training (up to ~70%% cheaper)")
    p.add_argument("--no-wait", action="store_true")
    args = p.parse_args()

    out = stack_outputs(args.stack, args.region)
    bucket, role = out["DataBucketName"], out["SageMakerRoleArn"]
    session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
    source = upload_source(bucket, args.region)
    print("uploaded training code to", source)

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=source,
        role=role,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        instance_count=1,
        instance_type=args.instance_type,
        hyperparameters={"epochs": args.epochs, "split": args.split},
        output_path="s3://%s/models/" % bucket,
        code_location="s3://%s/code" % bucket,
        base_job_name="drugresp",
        metric_definitions=METRICS,
        use_spot_instances=args.spot,
        max_run=3 * 3600,
        max_wait=4 * 3600 if args.spot else None,
        checkpoint_s3_uri=("s3://%s/checkpoints/" % bucket) if args.spot else None,
        sagemaker_session=session,
        tags=[{"Key": "project", "Value": "drugresponse"}],
    )
    estimator.fit({"train": "s3://%s/datasets/canonical/" % bucket}, wait=not args.no_wait, logs="All")

    job = estimator.latest_training_job.name
    state = {"training_job": job, "region": args.region}
    if not args.no_wait:
        state["model_data"] = estimator.model_data
        print("model artefact:", estimator.model_data)
    save_state(state)
    print("training job:", job)


if __name__ == "__main__":
    main()

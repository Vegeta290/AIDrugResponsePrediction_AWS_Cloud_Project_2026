"""Step 5 (optional): deploy the trained model to a SageMaker endpoint.

    python src/aws/sagemaker/deploy_endpoint.py                     # serverless (pay per request)
    python src/aws/sagemaker/deploy_endpoint.py --instance-type ml.t2.medium   # always-on real-time

Creating the endpoint the first time takes ~5-10 minutes. Re-running with a
new model updates the existing endpoint in place (no downtime).
"""
import argparse
import os
import time

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

from common import AI_MODEL_DIR, FRAMEWORK_VERSION, PY_VERSION, load_state, save_state, stack_outputs


def latest_training_job(sm):
    jobs = sm.list_training_jobs(NameContains="drugresp", StatusEquals="Completed",
                                 SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
    if not jobs["TrainingJobSummaries"]:
        raise SystemExit("No completed training job found; run launch_training.py first")
    name = jobs["TrainingJobSummaries"][0]["TrainingJobName"]
    return sm.describe_training_job(TrainingJobName=name)["ModelArtifacts"]["S3ModelArtifacts"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stack", default=os.environ.get("STACK", "drugresp-stack"))
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-southeast-2"))
    p.add_argument("--model-data", help="s3://.../model.tar.gz (default: newest completed training job)")
    p.add_argument("--instance-type", help="real-time instance type; omit for serverless")
    p.add_argument("--memory-mb", type=int, default=4096, choices=[1024, 2048, 3072, 4096, 5120, 6144])
    p.add_argument("--max-concurrency", type=int, default=5)
    args = p.parse_args()

    out = stack_outputs(args.stack, args.region)
    endpoint, role = out["SageMakerEndpointName"], out["SageMakerRoleArn"]
    sm = boto3.client("sagemaker", region_name=args.region)
    session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
    model_data = args.model_data or latest_training_job(sm)
    print("deploying", model_data)

    model = PyTorchModel(
        model_data=model_data, role=role, entry_point="inference.py", source_dir=AI_MODEL_DIR,
        framework_version=FRAMEWORK_VERSION, py_version=PY_VERSION, sagemaker_session=session,
        name="drugresp-%s" % time.strftime("%Y%m%d-%H%M%S"),
    )
    serverless = None if args.instance_type else ServerlessInferenceConfig(
        memory_size_in_mb=args.memory_mb, max_concurrency=args.max_concurrency)
    model.create(instance_type=args.instance_type or "ml.m5.large", serverless_inference_config=serverless)

    variant = {"VariantName": "AllTraffic", "ModelName": model.name}
    if serverless:
        variant["ServerlessConfig"] = {"MemorySizeInMB": args.memory_mb, "MaxConcurrency": args.max_concurrency}
    else:
        variant.update({"InstanceType": args.instance_type, "InitialInstanceCount": 1})
    config_name = model.name + "-cfg"
    sm.create_endpoint_config(EndpointConfigName=config_name, ProductionVariants=[variant])

    try:
        sm.describe_endpoint(EndpointName=endpoint)
        print("updating endpoint", endpoint)
        sm.update_endpoint(EndpointName=endpoint, EndpointConfigName=config_name)
    except sm.exceptions.ClientError:
        print("creating endpoint", endpoint)
        sm.create_endpoint(EndpointName=endpoint, EndpointConfigName=config_name)
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint, WaiterConfig={"Delay": 30, "MaxAttempts": 60})
    print("endpoint %s is InService" % endpoint)
    state = load_state()
    state.update({"endpoint": endpoint, "endpoint_model_data": model_data})
    save_state(state)


if __name__ == "__main__":
    main()

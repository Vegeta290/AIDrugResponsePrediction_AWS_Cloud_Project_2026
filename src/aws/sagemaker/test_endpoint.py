"""Send the example patient file to the SageMaker endpoint and print the predictions.

    python src/aws/sagemaker/test_endpoint.py [path/to/sample.csv]
"""
import csv
import json
import os
import sys

import boto3

from common import ROOT, stack_outputs


def main():
    region = os.environ.get("AWS_REGION", "ap-southeast-2")
    out = stack_outputs(os.environ.get("STACK", "drugresp-stack"), region)
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "datasets", "sample_patient.csv")
    sample = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sample.setdefault(row["modality"], {})[row["feature"]] = float(row["value"])

    runtime = boto3.client("sagemaker-runtime", region_name=region)
    resp = runtime.invoke_endpoint(EndpointName=out["SageMakerEndpointName"], ContentType="application/json",
                                   Body=json.dumps({"sample": sample, "explain": True, "top_k": 5}))
    result = json.loads(resp["Body"].read())
    print("model", result["model_version"], "coverage", result["coverage"])
    for r in result["predictions"]:
        print("%-18s IC50 %8.2f uM  %s" % (r["drug_name"], r["predicted_ic50_um"], r["sensitivity"]))


if __name__ == "__main__":
    main()

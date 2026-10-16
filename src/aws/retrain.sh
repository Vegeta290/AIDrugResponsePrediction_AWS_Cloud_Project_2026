#!/usr/bin/env bash
# Trigger the retraining Lambda: upload new data (optional) then drop triggers/retrain.json.
#   src/aws/retrain.sh                 # retrain on what is already in datasets/canonical/
#   src/aws/retrain.sh data/canonical  # sync new data first
source "$(dirname "$0")/common.sh"
BUCKET=$(out DataBucketName)
[[ $# -ge 1 ]] && bash src/aws/upload_data.sh "$1"
# The Lambda reads code from s3://$BUCKET/code/sourcedir.tar.gz (uploaded by launch_training.py).
echo '{"hyperparameters": {"epochs": "40"}}' > build/retrain.json
aws s3 cp build/retrain.json "s3://$BUCKET/triggers/retrain.json"
echo "Retraining requested. Follow it with:"
echo "  aws sagemaker list-training-jobs --name-contains drugresp --sort-by CreationTime --sort-order Descending --max-results 3"

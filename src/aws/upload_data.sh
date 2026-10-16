#!/usr/bin/env bash
# Step 2: upload the canonical training data to S3 (datasets/canonical/).
#   src/aws/upload_data.sh [local_dir]     default: data/canonical
source "$(dirname "$0")/common.sh"
SRC="${1:-data/canonical}"
BUCKET=$(out DataBucketName)
aws s3 sync "$SRC" "s3://$BUCKET/datasets/canonical/" --exclude "*" \
  --include "expression.csv" --include "cnv.csv" --include "mutation.csv" --include "proteomics.csv" \
  --include "response.csv" --include "drugs.csv" --include "metadata.csv"
aws s3 ls "s3://$BUCKET/datasets/canonical/" --human-readable

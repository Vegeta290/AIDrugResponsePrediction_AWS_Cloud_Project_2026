#!/usr/bin/env bash
# Sourced by the deployment scripts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source src/aws/config.env
export AWS_DEFAULT_REGION="$AWS_REGION"
mkdir -p build

# out <OutputKey>  -> value of a CloudFormation stack output
out() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

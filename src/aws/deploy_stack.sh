#!/usr/bin/env bash
# Step 1: create / update all AWS infrastructure with CloudFormation.
source "$(dirname "$0")/common.sh"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ARTIFACTS="${PROJECT}-artifacts-${ACCOUNT}-${AWS_REGION}"
if ! aws s3api head-bucket --bucket "$ARTIFACTS" 2>/dev/null; then
  echo "Creating artifacts bucket $ARTIFACTS"
  aws s3 mb "s3://$ARTIFACTS" --region "$AWS_REGION"
fi

# Secrets as SSM SecureString parameters (free, unlike Secrets Manager). Created once, never overwritten.
param() {
  aws ssm get-parameter --name "/$PROJECT/$1" >/dev/null 2>&1 ||
    aws ssm put-parameter --name "/$PROJECT/$1" --type SecureString --value "$2" >/dev/null
}
param db-password "$(openssl rand -hex 16)"
param jwt-secret "$(openssl rand -hex 32)"
param admin-email "admin@drugresponse.local"
param admin-password "$(openssl rand -hex 10)"

aws cloudformation package \
  --template-file src/aws/cloudformation/drugresponse-stack.yaml \
  --s3-bucket "$ARTIFACTS" \
  --output-template-file build/packaged.yaml

aws cloudformation deploy \
  --template-file build/packaged.yaml \
  --stack-name "$STACK" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides ProjectName="$PROJECT" AlertEmail="$ALERT_EMAIL" FreeTierOnly="$FREE_TIER_ONLY" InstanceType="$INSTANCE_TYPE" \
  --no-fail-on-empty-changeset

aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs" --output table

#!/usr/bin/env bash
# Step 4: build + deploy the API/dashboard container on the EC2 instance (no SSH needed: uses SSM).
#   INFERENCE_MODE=local     src/aws/ec2/deploy_app.sh   # model from latest SageMaker training job, runs on EC2
#   INFERENCE_MODE=sagemaker src/aws/ec2/deploy_app.sh   # forward predictions to the SageMaker endpoint
#   MODEL_S3_URI=s3://.../model.tar.gz src/aws/ec2/deploy_app.sh
#   LOCAL_MODEL=models/latest src/aws/ec2/deploy_app.sh       # upload a locally trained model (no SageMaker cost)
source "$(dirname "$0")/../common.sh"

BUCKET=$(out DataBucketName)
INSTANCE=$(out ApiServerInstanceId)

if [[ "$FREE_TIER_ONLY" == "true" ]]; then
  [[ "$INFERENCE_MODE" == "sagemaker" ]] && { echo "FREE_TIER_ONLY=true: SageMaker endpoints are not free"; exit 1; }
  [[ -z "${MODEL_S3_URI:-}" ]] && LOCAL_MODEL="${LOCAL_MODEL:-models/latest}"   # skip the SageMaker job lookup
fi

if [[ -n "${LOCAL_MODEL:-}" ]]; then
  COPYFILE_DISABLE=1 tar -czf build/model.tar.gz -C "$LOCAL_MODEL" .
  aws s3 cp build/model.tar.gz "s3://$BUCKET/models/local/model.tar.gz"
  MODEL_S3_URI="s3://$BUCKET/models/local/model.tar.gz"
fi

if [[ "$INFERENCE_MODE" == "local" && -z "${MODEL_S3_URI:-}" ]]; then
  JOB=$(aws sagemaker list-training-jobs --name-contains drugresp --status-equals Completed \
        --sort-by CreationTime --sort-order Descending --max-results 1 \
        --query "TrainingJobSummaries[0].TrainingJobName" --output text)
  [[ "$JOB" == "None" ]] && { echo "No completed training job; run launch_training.py or set MODEL_S3_URI"; exit 1; }
  MODEL_S3_URI=$(aws sagemaker describe-training-job --training-job-name "$JOB" \
        --query ModelArtifacts.S3ModelArtifacts --output text)
fi
echo "inference mode: $INFERENCE_MODE ${MODEL_S3_URI:-}"

COPYFILE_DISABLE=1 tar --exclude "__pycache__" --exclude "*.pyc" -czf build/app.tar.gz \
  src/ai_model src/backend src/frontend
aws s3 cp build/app.tar.gz "s3://$BUCKET/deploy/app.tar.gz"
aws s3 cp src/aws/ec2/setup_app.sh "s3://$BUCKET/deploy/setup_app.sh"

ENVS="BUCKET=$BUCKET REGION=$AWS_REGION DB_HOST=$(out DatabaseEndpoint) PARAM_PREFIX=$(out ParameterPrefix)"
ENVS="$ENVS LOG_GROUP=$(out AppLogGroupName)"
ENVS="$ENVS INFERENCE_MODE=$INFERENCE_MODE MODEL_S3_URI=${MODEL_S3_URI:-} SAGEMAKER_ENDPOINT=$(out SageMakerEndpointName)"
PARAMS=$(python3 -c 'import json,sys; print(json.dumps({"commands": [
  "aws s3 cp s3://%s/deploy/setup_app.sh /opt/drugresponse/setup_app.sh --region %s" % (sys.argv[1], sys.argv[2]),
  "env %s bash /opt/drugresponse/setup_app.sh" % sys.argv[3]], "executionTimeout": ["3600"]}))' \
  "$BUCKET" "$AWS_REGION" "$ENVS")

CMD=$(aws ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
  --comment "deploy drugresponse api" --parameters "$PARAMS" --query Command.CommandId --output text)
echo "SSM command $CMD sent to $INSTANCE (docker build takes ~5-10 min on first run)"

while true; do
  STATUS=$(aws ssm get-command-invocation --command-id "$CMD" --instance-id "$INSTANCE" \
           --query Status --output text 2>/dev/null || echo Pending)
  case "$STATUS" in Pending|InProgress|Delayed) printf "."; sleep 15 ;; *) echo; break ;; esac
done
aws ssm get-command-invocation --command-id "$CMD" --instance-id "$INSTANCE" \
  --query "[Status, StandardOutputContent, StandardErrorContent]" --output text | tail -40
echo
echo "Dashboard: $(out ApiUrl)"
echo "Admin login: $(aws ssm get-parameter --with-decryption --name "$(out ParameterPrefix)admin-email" --query Parameter.Value --output text)"
echo "Password:    aws ssm get-parameter --region $AWS_REGION --with-decryption --name $(out ParameterPrefix)admin-password --query Parameter.Value --output text"

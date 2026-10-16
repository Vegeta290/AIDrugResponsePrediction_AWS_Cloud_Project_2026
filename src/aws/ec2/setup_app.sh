#!/bin/bash
# Runs ON the EC2 instance (as root, via SSM Run Command from deploy_app.sh).
# Pulls the app bundle from S3, reads secrets from SSM Parameter Store, builds and (re)starts the API container.
# Required env: BUCKET REGION DB_HOST PARAM_PREFIX LOG_GROUP INFERENCE_MODE
# Optional env: MODEL_S3_URI (local mode), SAGEMAKER_ENDPOINT (sagemaker mode)
set -euo pipefail
cd /opt/drugresponse

aws s3 cp "s3://$BUCKET/deploy/app.tar.gz" app.tar.gz --region "$REGION"
rm -rf app && mkdir app && tar -xzf app.tar.gz -C app

param() { aws ssm get-parameter --name "${PARAM_PREFIX}$1" --with-decryption --region "$REGION" --query Parameter.Value --output text; }

DB_USER=drugadmin
DB_PASS=$(param db-password | python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.stdin.read().strip(), safe=''))")

umask 077
cat > app.env <<EOF
DATABASE_URL=postgresql+psycopg2://${DB_USER}:${DB_PASS}@${DB_HOST}:5432/drugresponse?sslmode=require
STORAGE_BACKEND=s3
S3_BUCKET=${BUCKET}
AWS_REGION=${REGION}
AWS_DEFAULT_REGION=${REGION}
INFERENCE_MODE=${INFERENCE_MODE}
MODEL_S3_URI=${MODEL_S3_URI:-}
SAGEMAKER_ENDPOINT=${SAGEMAKER_ENDPOINT:-}
JWT_SECRET=$(param jwt-secret)
ADMIN_EMAIL=$(param admin-email)
ADMIN_PASSWORD=$(param admin-password)
ALLOW_SIGNUP=false
CLOUDWATCH_METRICS=true
EOF

docker build -f app/src/backend/Dockerfile -t drugresponse-api:latest app
docker rm -f drugresponse-api 2>/dev/null || true
docker run -d --name drugresponse-api --restart unless-stopped -p 80:8000 \
  --env-file /opt/drugresponse/app.env \
  --log-driver awslogs --log-opt awslogs-region="$REGION" \
  --log-opt awslogs-group="$LOG_GROUP" --log-opt awslogs-stream="api-$(hostname)" \
  drugresponse-api:latest
docker image prune -f >/dev/null

for i in $(seq 1 60); do
  if curl -fsS http://localhost/health; then echo; echo "API is up"; exit 0; fi
  sleep 5
done
echo "API did not become healthy; recent logs:"; docker logs --tail 80 drugresponse-api; exit 1

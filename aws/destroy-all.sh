#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/2-application"
INFRA_DIR="${SCRIPT_DIR}/1-infrastructure"

ECR_REPO_NAME="langchain-pepwave"

AUTO_APPROVE="false"

usage() {
  cat <<'EOF'
Tear down ALL AWS resources for langchain-pepwave.

Deletes (in order):
  1. All Docker images in the ECR repository
  2. The application stack (ECS cluster/service/task definition)
  3. The infrastructure stack (RDS, ALB, ECR, IAM, security groups, etc.)

WARNING: This permanently deletes the RDS database with NO final snapshot.

Usage:
  ./destroy-all.sh [--auto-approve]

Options:
  --auto-approve  Skip Terraform confirmation prompts.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-approve)
      AUTO_APPROVE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${APP_DIR}/main.tf" || ! -f "${INFRA_DIR}/main.tf" ]]; then
  echo "❌ Could not find Terraform configs at ${APP_DIR} and ${INFRA_DIR}" >&2
  exit 1
fi

TF_ARGS=()
if [[ "${AUTO_APPROVE}" == "true" ]]; then
  TF_ARGS+=("-auto-approve")
fi

echo "⚠️  This will permanently delete all AWS resources for langchain-pepwave,"
echo "    including the RDS database (no final snapshot)."
if [[ "${AUTO_APPROVE}" != "true" ]]; then
  read -r -p "Type 'destroy' to continue: " CONFIRM
  if [[ "${CONFIRM}" != "destroy" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

echo ""
echo "🧹 Step 1/4: Emptying ECR repository '${ECR_REPO_NAME}'..."
if aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" >/dev/null 2>&1; then
  IMAGE_IDS="$(aws ecr list-images --repository-name "${ECR_REPO_NAME}" --query 'imageIds' --output json)"
  if [[ "${IMAGE_IDS}" != "[]" ]]; then
    aws ecr batch-delete-image \
      --repository-name "${ECR_REPO_NAME}" \
      --image-ids "${IMAGE_IDS}" >/dev/null
    echo "   ✅ Deleted all images."
  else
    echo "   ✅ Repository already empty."
  fi
else
  echo "   ✅ Repository does not exist; skipping."
fi

echo ""
echo "🧹 Step 2/4: Destroying application stack (ECS)..."
terraform -chdir="${APP_DIR}" destroy "${TF_ARGS[@]+"${TF_ARGS[@]}"}"

echo ""
echo "🧹 Step 3/4: Destroying infrastructure stack (RDS, ALB, ECR, IAM, ...)..."
terraform -chdir="${INFRA_DIR}" destroy "${TF_ARGS[@]+"${TF_ARGS[@]}"}"

echo ""
echo "🔍 Step 4/4: Verifying no billable resources remain..."
echo "--- ECS clusters:"
aws ecs list-clusters --query 'clusterArns' --output table
echo "--- RDS instances:"
aws rds describe-db-instances --query 'DBInstances[].DBInstanceIdentifier' --output table
echo "--- Load balancers:"
aws elbv2 describe-load-balancers --query 'LoadBalancers[].LoadBalancerName' --output table
echo "--- ECR repositories:"
aws ecr describe-repositories --query 'repositories[].repositoryName' --output table

echo ""
echo "✅ Teardown complete. Review the lists above: anything related to"
echo "   langchain-pepwave should be gone. Also check the AWS Billing console"
echo "   for leftovers (snapshots, public IPv4 addresses, Route 53 hosted zones)."

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/2-application"

AUTO_APPROVE="false"
SKIP_VALIDATE="false"

usage() {
  cat <<'EOF'
Redeploy the ECS service (force new deployment).

Usage:
  ./redeploy-app.sh [--auto-approve] [--skip-validate]

Options:
  --auto-approve    Run terraform apply without prompting.
  --skip-validate   Skip ./validate-image.sh check.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-approve)
      AUTO_APPROVE="true"
      shift
      ;;
    --skip-validate)
      SKIP_VALIDATE="true"
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

if [[ ! -f "${APP_DIR}/main.tf" ]]; then
  echo "❌ Could not find Terraform config at: ${APP_DIR}" >&2
  exit 1
fi

if [[ "${SKIP_VALIDATE}" != "true" ]]; then
  echo "🔍 Validating Docker image exists in ECR..."
  (cd "${APP_DIR}" && ./validate-image.sh)
fi

echo "🚀 Forcing ECS service redeploy (terraform replace)..."

if [[ "${AUTO_APPROVE}" == "true" ]]; then
  terraform -chdir="${APP_DIR}" apply -replace="aws_ecs_service.app" -auto-approve
else
  terraform -chdir="${APP_DIR}" apply -replace="aws_ecs_service.app"
fi

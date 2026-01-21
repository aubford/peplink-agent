#!/usr/bin/env bash
#
# Source-only observability helpers for this repo's AWS deployment.
#
# Usage (from repo root or anywhere):
#   source langchain-pepwave/aws/observability-tools.sh
#
# Then run helpers like:
#   aws_whoami
#   ecs_service
#   cw_tail
#   ecr_latest
#   compare_ecr_vs_ecs
#
# IMPORTANT:
# - This file is intended to be SOURCED (not executed). It defines variables and functions only.
# - It is targeted to this repo’s Terraform + ECS Fargate + ALB setup under `langchain-pepwave/aws/`.

################################################################################
# Configuration (override by exporting before sourcing)
################################################################################

# Region to target. Override with: export AWS_REGION=us-east-1
AWS_REGION="${AWS_REGION:-us-east-1}"

# ECS identifiers for this repo (match `aws/2-application/main.tf`).
ECS_CLUSTER_NAME="${ECS_CLUSTER_NAME:-langchain-pepwave-cluster}"
ECS_SERVICE_NAME="${ECS_SERVICE_NAME:-langchain-pepwave-service}"

# ECR repository name created by Terraform (match `aws/1-infrastructure/main.tf`).
ECR_REPOSITORY_NAME="${ECR_REPOSITORY_NAME:-langchain-pepwave}"
ECR_IMAGE_TAG="${ECR_IMAGE_TAG:-latest}"

# CloudWatch Logs group created by Terraform (match `aws/1-infrastructure/main.tf`).
# This is the *application container* logs written via the awslogs driver.
CW_LOG_GROUP="${CW_LOG_GROUP:-/ecs/langchain-pepwave}"

# Optional: ALB target group ARN (used for ALB health checks).
# If unset, the helper `tf_target_group_arn` can try to pull it from Terraform outputs.
TARGET_GROUP_ARN="${TARGET_GROUP_ARN:-}"

# Where Terraform lives in this repo (used only by helper functions that query outputs).
TF_INFRA_DIR="${TF_INFRA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/1-infrastructure}"
TF_APP_DIR="${TF_APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/2-application}"

# Terraform state file locations (these exist after `terraform apply`).
# We prefer reading state directly so sourcing remains fast and does not require `terraform init`.
TF_INFRA_STATE="${TF_INFRA_STATE:-${TF_INFRA_DIR}/terraform.tfstate}"
TF_APP_STATE="${TF_APP_STATE:-${TF_APP_DIR}/terraform.tfstate}"

# Autoload targeting vars on source (recommended).
# Override with: export OBS_AUTOLOAD=false
OBS_AUTOLOAD="${OBS_AUTOLOAD:-true}"
# If true, prints what was loaded during sourcing.
OBS_AUTOLOAD_VERBOSE="${OBS_AUTOLOAD_VERBOSE:-false}"

################################################################################
# Safety / UX helpers
################################################################################

_require_cmd() {
  # Fail fast with a helpful message if a required CLI isn't available.
  # We intentionally keep this minimal and source-safe (no set -e in this file).
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "❌ Missing required command: $cmd" >&2
    return 127
  fi
}

_hr() {
  printf '%s\n' "--------------------------------------------------------------------------------"
}

################################################################################
# Terraform-backed variable loading (optional)
################################################################################

_tfstate_output_raw() {
  # Internal helper: read a Terraform output value from a local tfstate file.
  #
  # This avoids `terraform output`, which can require provider plugins or registry access.
  # It reads: .outputs[output_name].value
  #
  # Usage:
  #   _tfstate_output_raw "/path/to/terraform.tfstate" "cloudwatch_log_group_name"
  local tfstate_path="$1"
  local output_name="$2"

  _require_cmd python || return $?
  python - <<'PY' "$tfstate_path" "$output_name"
import json, sys
path = sys.argv[1]
name = sys.argv[2]
try:
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
except FileNotFoundError:
  raise SystemExit(2)

outputs = data.get("outputs") or {}
val = (outputs.get(name) or {}).get("value")
if val is None:
  raise SystemExit(3)

if isinstance(val, (dict, list)):
  print(json.dumps(val))
else:
  print(val)
PY
}

_tf_output_raw() {
  # Internal helper: fetch a raw Terraform output value.
  #
  # IMPORTANT:
  # - This requires Terraform to be initialized in the target directory.
  # - If Terraform isn't initialized (or can't download providers), this will fail.
  local tf_dir="$1"
  local output_name="$2"

  _require_cmd terraform || return $?
  terraform -chdir="$tf_dir" output -raw "$output_name"
}

obs_tf_load() {
  # Populate this file's variables from Terraform outputs.
  #
  # Why:
  # - Avoid hardcoding values like log group name, target group ARN, cluster/service names
  # - Ensure your observability commands always target the deployed resources
  #
  # What it loads (when available):
  # - From Phase 1 (`$TF_INFRA_DIR`):
  #   - aws_region            -> AWS_REGION
  #   - cloudwatch_log_group_name -> CW_LOG_GROUP
  #   - target_group_arn      -> TARGET_GROUP_ARN
  #   - ecr_repository_url    -> ECR_REPOSITORY_NAME (derived from URL)
  # - From Phase 2 (`$TF_APP_DIR`):
  #   - ecs_cluster_name      -> ECS_CLUSTER_NAME
  #   - ecs_service_name      -> ECS_SERVICE_NAME
  #
  # Notes:
  # - This is NOT run automatically on source.
  # - If Terraform outputs can’t be read, it leaves your existing values unchanged.
  #
  # Usage:
  #   source langchain-pepwave/aws/observability-tools.sh
  #   obs_tf_load
  _require_cmd terraform || return $?

  local infra_region infra_log_group infra_tg_arn infra_ecr_url
  infra_region="$(_tf_output_raw "$TF_INFRA_DIR" aws_region 2>/dev/null || true)"
  infra_log_group="$(_tf_output_raw "$TF_INFRA_DIR" cloudwatch_log_group_name 2>/dev/null || true)"
  infra_tg_arn="$(_tf_output_raw "$TF_INFRA_DIR" target_group_arn 2>/dev/null || true)"
  infra_ecr_url="$(_tf_output_raw "$TF_INFRA_DIR" ecr_repository_url 2>/dev/null || true)"

  if [[ -n "$infra_region" ]]; then AWS_REGION="$infra_region"; fi
  if [[ -n "$infra_log_group" ]]; then CW_LOG_GROUP="$infra_log_group"; fi
  if [[ -n "$infra_tg_arn" ]]; then TARGET_GROUP_ARN="$infra_tg_arn"; fi
  if [[ -n "$infra_ecr_url" ]]; then
    # e.g. 003765....dkr.ecr.us-east-1.amazonaws.com/langchain-pepwave
    ECR_REPOSITORY_NAME="${infra_ecr_url##*/}"
  fi

  local app_cluster app_service
  app_cluster="$(_tf_output_raw "$TF_APP_DIR" ecs_cluster_name 2>/dev/null || true)"
  app_service="$(_tf_output_raw "$TF_APP_DIR" ecs_service_name 2>/dev/null || true)"
  if [[ -n "$app_cluster" ]]; then ECS_CLUSTER_NAME="$app_cluster"; fi
  if [[ -n "$app_service" ]]; then ECS_SERVICE_NAME="$app_service"; fi

  echo "✅ Loaded vars from Terraform (best-effort):"
  echo "  AWS_REGION=$AWS_REGION"
  echo "  ECS_CLUSTER_NAME=$ECS_CLUSTER_NAME"
  echo "  ECS_SERVICE_NAME=$ECS_SERVICE_NAME"
  echo "  ECR_REPOSITORY_NAME=$ECR_REPOSITORY_NAME"
  echo "  ECR_IMAGE_TAG=$ECR_IMAGE_TAG"
  echo "  CW_LOG_GROUP=$CW_LOG_GROUP"
  echo "  TARGET_GROUP_ARN=${TARGET_GROUP_ARN:-<unset>}"
}

obs_autoload() {
  # Autoload “always-needed” targeting variables from local Terraform state.
  #
  # This is safe to run during sourcing because:
  # - It only reads local JSON files (tfstate)
  # - It does not touch AWS
  #
  # If state files are missing, it silently leaves defaults in place.
  local infra_region infra_log_group infra_tg_arn infra_ecr_url
  local app_cluster app_service

  infra_region="$(_tfstate_output_raw "$TF_INFRA_STATE" aws_region 2>/dev/null || true)"
  infra_log_group="$(_tfstate_output_raw "$TF_INFRA_STATE" cloudwatch_log_group_name 2>/dev/null || true)"
  infra_tg_arn="$(_tfstate_output_raw "$TF_INFRA_STATE" target_group_arn 2>/dev/null || true)"
  infra_ecr_url="$(_tfstate_output_raw "$TF_INFRA_STATE" ecr_repository_url 2>/dev/null || true)"

  app_cluster="$(_tfstate_output_raw "$TF_APP_STATE" ecs_cluster_name 2>/dev/null || true)"
  app_service="$(_tfstate_output_raw "$TF_APP_STATE" ecs_service_name 2>/dev/null || true)"

  if [[ -n "$infra_region" ]]; then AWS_REGION="$infra_region"; fi
  if [[ -n "$infra_log_group" ]]; then CW_LOG_GROUP="$infra_log_group"; fi
  if [[ -n "$infra_tg_arn" ]]; then TARGET_GROUP_ARN="$infra_tg_arn"; fi
  if [[ -n "$infra_ecr_url" ]]; then ECR_REPOSITORY_NAME="${infra_ecr_url##*/}"; fi

  if [[ -n "$app_cluster" ]]; then ECS_CLUSTER_NAME="$app_cluster"; fi
  if [[ -n "$app_service" ]]; then ECS_SERVICE_NAME="$app_service"; fi

  if [[ "$OBS_AUTOLOAD_VERBOSE" == "true" ]]; then
    echo "✅ obs_autoload: loaded targeting vars from tfstate:"
    echo "  AWS_REGION=$AWS_REGION"
    echo "  ECS_CLUSTER_NAME=$ECS_CLUSTER_NAME"
    echo "  ECS_SERVICE_NAME=$ECS_SERVICE_NAME"
    echo "  ECR_REPOSITORY_NAME=$ECR_REPOSITORY_NAME"
    echo "  ECR_IMAGE_TAG=$ECR_IMAGE_TAG"
    echo "  CW_LOG_GROUP=$CW_LOG_GROUP"
    echo "  TARGET_GROUP_ARN=${TARGET_GROUP_ARN:-<unset>}"
  fi
}

################################################################################
# Core sanity checks
################################################################################

aws_whoami() {
  # Confirm which AWS identity is currently active (useful when multiple profiles exist).
  _require_cmd aws || return $?
  aws sts get-caller-identity --region "$AWS_REGION"
}

################################################################################
# CloudWatch Logs (app container logs)
################################################################################

cw_tail() {
  # Live-tail the ECS container logs for this service.
  #
  # When to use:
  # - Immediately after a deploy (to see startup exceptions)
  # - When ALB returns 503 and you suspect tasks are crashing
  #
  # Examples:
  #   cw_tail                 # defaults to last 30 minutes, follow
  #   cw_tail 2h              # last 2 hours
  #   cw_tail 10m --no-follow # last 10 minutes, no follow
  _require_cmd aws || return $?

  local since="${1:-30m}"
  local follow="${2:---follow}"

  aws logs tail "$CW_LOG_GROUP" \
    --region "$AWS_REGION" \
    --since "$since" \
    $follow
}

cw_streams() {
  # List the most recent log streams and their timestamps.
  #
  # Useful to:
  # - See which task attempted to start most recently
  # - Copy a specific stream name for deeper inspection with cw_stream()
  _require_cmd aws || return $?

  local max_items="${1:-10}"
  aws logs describe-log-streams \
    --log-group-name "$CW_LOG_GROUP" \
    --order-by LastEventTime \
    --descending \
    --max-items "$max_items" \
    --region "$AWS_REGION" \
    --query 'logStreams[].{name:logStreamName, lastEvent:lastEventTimestamp, firstEvent:firstEventTimestamp, storedBytes:storedBytes}' \
    --output table
}

cw_stream() {
  # Fetch recent log events from a specific log stream.
  #
  # Example:
  #   cw_stream "web/web/<task-id>" 200
  _require_cmd aws || return $?

  local stream_name="${1:-}"
  local limit="${2:-200}"
  if [[ -z "$stream_name" ]]; then
    echo "Usage: cw_stream <logStreamName> [limit]" >&2
    return 2
  fi

  aws logs get-log-events \
    --log-group-name "$CW_LOG_GROUP" \
    --log-stream-name "$stream_name" \
    --limit "$limit" \
    --region "$AWS_REGION" \
    --query 'events[].message' \
    --output text
}

cw_errors() {
  # Pull the most recent error-ish lines from the log group (fast triage).
  #
  # Notes:
  # - Filter patterns in CloudWatch are not regex; they are term/pattern based.
  # - This is best-effort: it tries to catch common Python failure signatures.
  #
  # Example:
  #   cw_errors 20    # last 20 minutes
  _require_cmd aws || return $?
  _require_cmd python || return $?

  local minutes="${1:-20}"
  local start_ms
  start_ms="$(python - <<PY
import time
print(int((time.time() - (${minutes} * 60)) * 1000))
PY
)"

  aws logs filter-log-events \
    --log-group-name "$CW_LOG_GROUP" \
    --region "$AWS_REGION" \
    --start-time "$start_ms" \
    --filter-pattern '?Traceback ?ImportError ?ModuleNotFoundError ?ERROR ?Exception' \
    --limit 50 \
    --query 'events[].message' \
    --output text
}

################################################################################
# ECS service & tasks
################################################################################

ecs_service() {
  # Show current desired/running/pending counts plus recent events.
  # This is the quickest way to see:
  # - whether tasks are failing to start
  # - whether targets are being registered/deregistered with the ALB
  _require_cmd aws || return $?

  aws ecs describe-services \
    --cluster "$ECS_CLUSTER_NAME" \
    --services "$ECS_SERVICE_NAME" \
    --region "$AWS_REGION" \
    --query 'services[0].{desired:desiredCount,running:runningCount,pending:pendingCount,deployments:deployments[].{status:status,desired:desiredCount,running:runningCount,failed:failedTasks,taskDef:taskDefinition},events:events[0:12].[createdAt,message]}' \
    --output json
}

ecs_tasks() {
  # List tasks for this ECS service.
  #
  # Example:
  #   ecs_tasks RUNNING
  #   ecs_tasks STOPPED
  _require_cmd aws || return $?

  local status="${1:-RUNNING}"
  aws ecs list-tasks \
    --cluster "$ECS_CLUSTER_NAME" \
    --service-name "$ECS_SERVICE_NAME" \
    --desired-status "$status" \
    --region "$AWS_REGION" \
    --output json
}

ecs_describe_tasks() {
  # Describe ECS tasks (useful for exit codes, stop reasons, and image digests).
  #
  # You can pass task ARNs directly, or pipe ARNs into xargs:
  #   aws ecs list-tasks ... --query 'taskArns' --output text | xargs ecs_describe_tasks
  _require_cmd aws || return $?

  if [[ $# -lt 1 ]]; then
    echo "Usage: ecs_describe_tasks <taskArn> [taskArn...]" >&2
    return 2
  fi

  aws ecs describe-tasks \
    --cluster "$ECS_CLUSTER_NAME" \
    --tasks "$@" \
    --region "$AWS_REGION" \
    --query 'tasks[].{taskArn:taskArn,createdAt:createdAt,startedAt:startedAt,stoppedAt:stoppedAt,stopCode:stopCode,stoppedReason:stoppedReason,taskDef:taskDefinitionArn,containers:containers[].{name:name,lastStatus:lastStatus,exitCode:exitCode,reason:reason,image:image,imageDigest:imageDigest}}' \
    --output json
}

ecs_recent_failures() {
  # Convenience: show the last N STOPPED tasks with key fields:
  # - stop reason
  # - container exit code
  # - image digest (to confirm which ECR image actually ran)
  #
  # Example:
  #   ecs_recent_failures 10
  _require_cmd aws || return $?
  _require_cmd python || return $?

  local count="${1:-5}"

  python - <<PY
import json, subprocess
list_cmd = [
  "aws","ecs","list-tasks",
  "--cluster","$ECS_CLUSTER_NAME",
  "--service-name","$ECS_SERVICE_NAME",
  "--desired-status","STOPPED",
  "--region","$AWS_REGION",
  "--max-results", str($count),
  "--output","json",
]
arns = json.loads(subprocess.check_output(list_cmd, text=True)).get("taskArns", [])
print(f"STOPPED_TASKS={len(arns)}")
if not arns:
  raise SystemExit(0)
desc_cmd = [
  "aws","ecs","describe-tasks",
  "--cluster","$ECS_CLUSTER_NAME",
  "--region","$AWS_REGION",
  "--tasks", *arns,
  "--output","json",
]
out = json.loads(subprocess.check_output(desc_cmd, text=True))
for t in out.get("tasks", []):
  c = (t.get("containers") or [{}])[0]
  print(json.dumps({
    "createdAt": t.get("createdAt"),
    "stoppedAt": t.get("stoppedAt"),
    "taskArn": t.get("taskArn"),
    "stopCode": t.get("stopCode"),
    "stoppedReason": t.get("stoppedReason"),
    "containerExitCode": c.get("exitCode"),
    "containerReason": c.get("reason"),
    "imageDigest": c.get("imageDigest"),
    "taskDefinitionArn": t.get("taskDefinitionArn"),
  }, default=str))
PY
}

################################################################################
# ECR vs ECS (confirm tasks are running the image you think they are)
################################################################################

ecr_latest() {
  # Show the digest + push time for the ECR tag (default: :latest).
  # This is key when you push a new :latest but ECS seems to still crash
  # with an "old" bug: you want to confirm the digest actually changed.
  _require_cmd aws || return $?

  aws ecr describe-images \
    --repository-name "$ECR_REPOSITORY_NAME" \
    --image-ids "imageTag=${ECR_IMAGE_TAG}" \
    --region "$AWS_REGION" \
    --query 'imageDetails[0].{imageDigest:imageDigest,imageTags:imageTags,imagePushedAt:imagePushedAt}' \
    --output json
}

compare_ecr_vs_ecs() {
  # Compare ECR's digest for the tag (usually :latest) with the digest that ECS
  # STOPPED tasks actually ran.
  #
  # When to use:
  # - After pushing a new image and redeploying, but tasks still show an older error.
  # - To confirm that ECS is really pulling the image you just pushed.
  _require_cmd aws || return $?
  _require_cmd python || return $?

  _hr
  echo "ECR digest for ${ECR_REPOSITORY_NAME}:${ECR_IMAGE_TAG} (region=$AWS_REGION)"
  local ecr_json
  ecr_json="$(aws ecr describe-images --repository-name "$ECR_REPOSITORY_NAME" --image-ids "imageTag=${ECR_IMAGE_TAG}" --region "$AWS_REGION" --output json)"
  python -c 'import json,sys; data=json.loads(sys.argv[1]); img=(data.get("imageDetails") or [{}])[0]; print(json.dumps({"imageDigest": img.get("imageDigest"), "imagePushedAt": img.get("imagePushedAt"), "imageTags": img.get("imageTags")}, default=str))' "$ecr_json"

  _hr
  echo "Most recent STOPPED task image digests (service=${ECS_SERVICE_NAME})"
  ecs_recent_failures 5
  _hr
}

################################################################################
# ALB target group health (503s usually mean 0 healthy targets)
################################################################################

tf_target_group_arn() {
  # Try to read the target group ARN from Terraform Phase 1 outputs.
  #
  # Notes:
  # - This requires that Terraform is initialized in `aws/1-infrastructure/`.
  # - If terraform init fails on your machine, set TARGET_GROUP_ARN manually.
  _require_cmd terraform || return $?

  terraform -chdir="$TF_INFRA_DIR" output -raw target_group_arn
}

alb_target_health() {
  # Show ALB target health for the ECS target group.
  #
  # This is the fastest way to explain a 503:
  # - if there are 0 healthy targets, the ALB will return 503.
  #
  # If TARGET_GROUP_ARN is not set, this will try to fetch it from Terraform outputs.
  _require_cmd aws || return $?

  local tg="${TARGET_GROUP_ARN:-}"
  if [[ -z "$tg" ]]; then
    tg="$(tf_target_group_arn 2>/dev/null || true)"
  fi
  if [[ -z "$tg" ]]; then
    echo "Usage: set TARGET_GROUP_ARN or ensure Terraform outputs are available." >&2
    return 2
  fi

  aws elbv2 describe-target-health \
    --target-group-arn "$tg" \
    --region "$AWS_REGION" \
    --query 'TargetHealthDescriptions[].{Target:Target.Id,Port:Target.Port,State:TargetHealth.State,Reason:TargetHealth.Reason,Description:TargetHealth.Description}' \
    --output table
}

################################################################################
# ECS Exec (only works when there's a RUNNING task)
################################################################################

ecs_exec() {
  # Open an interactive shell into the running container (ECS Exec).
  #
  # When to use:
  # - Confirm environment variables are present
  # - Run curl locally inside the task, check connectivity, etc.
  #
  # Requirements:
  # - The service has enable_execute_command = true (it does in `2-application/main.tf`)
  # - There is at least one RUNNING task
  # - Your IAM principal has ecs:ExecuteCommand and related SSM permissions
  _require_cmd aws || return $?

  local task_arn
  task_arn="$(aws ecs list-tasks --cluster "$ECS_CLUSTER_NAME" --service-name "$ECS_SERVICE_NAME" --desired-status RUNNING --region "$AWS_REGION" --query 'taskArns[0]' --output text)"
  if [[ -z "$task_arn" || "$task_arn" == "None" ]]; then
    echo "❌ No RUNNING tasks found for service ${ECS_SERVICE_NAME}" >&2
    return 1
  fi

  aws ecs execute-command \
    --cluster "$ECS_CLUSTER_NAME" \
    --task "$task_arn" \
    --container web \
    --command "/bin/sh" \
    --interactive \
    --region "$AWS_REGION"
}

################################################################################
# Quick “what should I run” recipe for this repo
################################################################################

obs_quickcheck() {
  # A quick triage sequence for the common "ALB shows 503" symptom:
  # - Check if targets are healthy (ALB)
  # - Check ECS service counts + failedTasks
  # - Pull recent app errors from CloudWatch
  _hr
  echo "1) ALB target health (0 healthy => 503)"
  alb_target_health || true
  _hr
  echo "2) ECS service status + recent events"
  ecs_service || true
  _hr
  echo "3) Recent CloudWatch log errors"
  cw_errors 30 || true
  _hr
  echo "4) Compare ECR :latest digest vs task digests"
  compare_ecr_vs_ecs || true
  _hr
}

################################################################################
# Auto-run on source (optional)
################################################################################

# If you want sourcing this file to be completely “silent/no-op”, export:
#   export OBS_AUTOLOAD=false
#
# Otherwise we auto-populate the targeting vars from local Terraform state so the
# other helper functions work immediately.
if [[ "$OBS_AUTOLOAD" == "true" ]]; then
  obs_autoload >/dev/null 2>&1 || true
fi

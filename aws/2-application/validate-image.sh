#!/bin/bash

set -e

echo "🔍 Validating Docker image exists in ECR..."

# Check if Phase 1 is complete
if [ ! -f "../1-infrastructure/terraform.tfstate" ]; then
    echo "❌ Error: Phase 1 not complete. Run 'terraform apply' in ../1-infrastructure/ first."
    exit 1
fi

# Get ECR repository URL from Phase 1
ECR_URL=$(terraform -chdir=../1-infrastructure output -raw ecr_repository_url)
REPO_NAME=$(echo $ECR_URL | cut -d'/' -f2)
REGION=$(terraform -chdir=../1-infrastructure output -raw aws_region || echo "us-east-1")

echo "📍 Checking ECR Repository: $ECR_URL"

# Check if the image exists
if aws ecr describe-images --repository-name $REPO_NAME --image-ids imageTag=latest --region $REGION >/dev/null 2>&1; then
    echo "✅ Docker image found in ECR!"
    echo "🚀 Ready to deploy Phase 2"
else
    echo "❌ Error: Docker image 'latest' not found in ECR repository '$REPO_NAME'"
    echo ""
    echo "Please build and push your Docker image first:"
    echo "1. cd ../  # Go back to aws directory"
    echo "2. ./build-and-push.sh"
    exit 1
fi
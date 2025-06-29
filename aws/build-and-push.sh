#!/bin/bash

set -e

echo "🐳 Building and pushing Docker image to ECR..."

# Check if we're in the right directory
if [ ! -f "1-infrastructure/terraform.tfstate" ]; then
    echo "❌ Error: Please run this script from the aws/ directory after Phase 1 is complete"
    exit 1
fi

# Get ECR repository URL from Phase 1 output
ECR_URL=$(terraform -chdir=1-infrastructure output -raw ecr_repository_url)
REGISTRY=$(echo $ECR_URL | cut -d'/' -f1)
AWS_REGION=$(terraform -chdir=1-infrastructure output -raw aws_region || echo "us-east-1")

echo "📍 ECR Repository: $ECR_URL"

# Navigate to project root (langchain-pepwave directory)
cd ..

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY

echo "🏗️  Building Docker image..."
docker build -t langchain-pepwave .

echo "🏷️  Tagging image for ECR..."
docker tag langchain-pepwave:latest $ECR_URL:latest

echo "📤 Pushing image to ECR..."
docker push $ECR_URL:latest

echo "✅ Docker image successfully pushed to ECR!"
echo ""
echo "Next steps:"
echo "1. cd aws/2-application"
echo "2. terraform init"
echo "3. terraform apply"
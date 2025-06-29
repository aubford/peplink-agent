# LangChain Pepwave - 2-Phase Terraform Deployment

This deployment uses a **2-phase approach** to avoid failures during initial deployment:

1. **Phase 1**: Deploy infrastructure (RDS, ECR, IAM roles)
2. **Phase 2**: Deploy application (ECS) after Docker image is ready

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform installed
3. Docker installed and running

## Phase 1: Infrastructure Setup

### 1. Configure Secrets

Navigate to the infrastructure directory:
```bash
cd 1-infrastructure
```

Store your API keys in AWS Secrets Manager:
```bash
./setup-secrets.sh
```

This script will prompt you to enter the following secrets (input will be hidden):
- **PostgreSQL Database Password** - Choose a secure password for your database
- **Pinecone API Key** - Your Pinecone vector database API key
- **OpenAI API Key** - Your OpenAI API key for LLM access
- **Cohere API Key** - Your Cohere API key for embeddings/LLM

### 2. Deploy Infrastructure

```bash
terraform init
terraform plan
terraform apply
```

This creates:
- ECR repository for your Docker images
- RDS PostgreSQL database
- CloudWatch log group
- VPC networking resources

### 3. Build and Push Docker Image

After infrastructure is deployed, get the ECR repository URL:
```bash
terraform output ecr_repository_url
```

Navigate back to the project root and build/push your image:
```bash
cd ../../

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(terraform -chdir=langchain-pepwave/aws/1-infrastructure output -raw ecr_repository_url | cut -d'/' -f1)

# Build the Docker image
docker build -t langchain-pepwave .

# Tag for ECR
docker tag langchain-pepwave:latest $(terraform -chdir=langchain-pepwave/aws/1-infrastructure output -raw ecr_repository_url):latest

# Push to ECR
docker push $(terraform -chdir=langchain-pepwave/aws/1-infrastructure output -raw ecr_repository_url):latest
```

## Phase 2: Application Deployment

Now that your Docker image is in ECR, deploy the ECS application:

```bash
cd langchain-pepwave/aws/2-application

# Optional: Validate that Docker image exists in ECR
./validate-image.sh

terraform init
terraform plan
terraform apply
```

This creates:
- ECS Fargate cluster
- ECS task definition (using your Docker image)
- ECS service (runs your application)

## What This Creates

### Phase 1 Infrastructure:
- **ECR Repository** - Stores your Docker images
- **RDS PostgreSQL** - Database (publicly accessible for demo)
- **CloudWatch Log Group** - Application logs

### Phase 2 Application:
- **ECS Cluster** - Container orchestration
- **ECS Task Definition** - Container specification
- **ECS Service** - Runs and maintains your application

## Workflow Summary

```
1. cd 1-infrastructure && ./setup-secrets.sh
2. terraform init && terraform apply          # Deploy infrastructure
3. cd .. && ./build-and-push.sh              # Build and push image
4. cd 2-application && ./validate-image.sh   # Validate image exists
5. terraform init && terraform apply         # Deploy application
```

## Security Note

This is a minimal demo configuration with:
- Publicly accessible RDS instance
- No custom security groups (uses default VPC)
- Simplified networking

For production use, add proper security groups, private subnets, and network isolation.

## Cleanup

To tear down everything:
```bash
# Destroy application first
cd 2-application && terraform destroy

# Then destroy infrastructure
cd ../1-infrastructure && terraform destroy
```
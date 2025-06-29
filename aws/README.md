# LangChain Pepwave - 2-Phase Terraform Deployment

This deployment uses a **2-phase approach** to avoid failures during initial deployment:

1. **Phase 1**: Deploy infrastructure (RDS, ECR, CloudWatch)
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

### 3. Build and Push Docker Image

After infrastructure is deployed, use the helper script to build and push your Docker image:

```bash
cd ..
./build-and-push.sh
```

This script will:
- Get the ECR repository URL from Phase 1
- Login to ECR
- Build your Docker image
- Tag and push it to ECR

## Phase 2: Application Deployment

Now that your Docker image is in ECR, deploy the ECS application:

```bash
cd 2-application

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
1. cd 1-infrastructure
2. ./setup-secrets.sh                        # Store API keys
3. terraform init && terraform apply         # Deploy infrastructure
4. cd .. && ./build-and-push.sh              # Build and push image
5. cd 2-application && ./validate-image.sh   # Validate image exists
6. terraform init && terraform apply         # Deploy application
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
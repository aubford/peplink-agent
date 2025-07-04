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

Create your secrets files in both directories:
```bash
# Create secrets file for Phase 1
cd 1-infrastructure
cp terraform.tfvars.example terraform.tfvars

# Create secrets file for Phase 2
cd ../2-application
cp terraform.tfvars.example terraform.tfvars
```

Edit both `terraform.tfvars` files and fill in your actual values:
- **postgres_password** - Choose a secure password for your database
- **pinecone_api_key** - Your Pinecone API key
- **openai_api_key** - Your OpenAI API key for LLM access
- **cohere_api_key** - Your Cohere API key for embeddings/LLM

The `terraform.tfvars` files are gitignored so your secrets won't be committed to version control.

### 2. Deploy Infrastructure

Navigate to the infrastructure directory:
```bash
cd 1-infrastructure
terraform init
terraform plan
terraform apply
```

This creates:
- ECR repository for your Docker images
- RDS PostgreSQL database
- CloudWatch log group
- ECS task execution role (for pulling images and logging)
- Security group (allows inbound traffic on port 8000)

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
- **ECS Task Execution Role** - IAM role for ECS tasks to access AWS services
- **Security Group** - Allows inbound HTTP traffic on port 8000

### Phase 2 Application:
- **ECS Cluster** - Container orchestration
- **ECS Task Definition** - Container specification
- **ECS Service** - Runs and maintains your application

## Workflow Summary

```
1. cd 1-infrastructure && cp terraform.tfvars.example terraform.tfvars  # Create secrets for Phase 1
2. cd ../2-application && cp terraform.tfvars.example terraform.tfvars  # Create secrets for Phase 2
3. # Edit both terraform.tfvars files with your API keys
4. cd ../1-infrastructure && terraform init && terraform apply          # Deploy infrastructure
5. cd .. && ./build-and-push.sh                                        # Build and push image
6. cd 2-application && ./validate-image.sh                             # Validate image exists
7. terraform init && terraform apply                                   # Deploy application
```

## Replace the existing image with a new one

`./build-and-push.sh` will build and push the image to ECR.

Force new deployment:
```bash
cd 2-application
terraform apply -replace="aws_ecs_service.app"
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
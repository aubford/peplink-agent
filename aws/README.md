# LangChain Pepwave - Terraform Deployment

This is a minimal Terraform configuration for deploying the LangChain Pepwave application to AWS using ECS Fargate and RDS PostgreSQL.

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform installed
3. Secrets already created in AWS Secrets Manager (use `setup-secrets.sh`)
4. Docker image pushed to ECR

## Deployment

1. Initialize Terraform:
   ```bash
   terraform init
   ```

2. Plan the deployment:
   ```bash
   terraform plan
   ```

3. Apply the configuration:
   ```bash
   terraform apply
   ```

## What This Creates

- RDS PostgreSQL database (publicly accessible for demo)
- ECS Fargate cluster and service
- IAM roles for ECS tasks
- CloudWatch log group

## Security Note

This is a minimal demo configuration with:
- Publicly accessible RDS instance
- No custom security groups (uses default VPC)
- Simplified networking

For production use, add proper security groups, private subnets, and network isolation.
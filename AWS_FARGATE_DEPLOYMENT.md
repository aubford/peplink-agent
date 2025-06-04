# AWS Fargate Deployment Guide for LangChain-Pepwave

This guide walks you through deploying the LangChain-Pepwave RAG chatbot on AWS Fargate using ECS.

## Architecture Overview

The deployment consists of:
- **Web Application**: FastAPI app running on port 8000
- **PostgreSQL Database**: For LangGraph checkpoints (containerized)
- **External Services**: Pinecone (vector store), OpenAI, Cohere, LangSmith APIs
- **AWS Services**: ECS Fargate, ECR, Secrets Manager, CloudWatch

## Prerequisites

1. **AWS CLI configured** with appropriate permissions
2. **Docker** installed and running
3. **API Keys** for:
   - Pinecone
   - OpenAI
   - LangSmith
   - Cohere
   - PostgreSQL password

## Step-by-Step Deployment

### 1. Build and Push Docker Image

The Docker image has already been built and pushed to ECR. If you need to update it:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 003765455645.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t langchain-pepwave .

# Tag and push
docker tag langchain-pepwave:latest 003765455645.dkr.ecr.us-east-1.amazonaws.com/langchain-pepwave:latest
docker push 003765455645.dkr.ecr.us-east-1.amazonaws.com/langchain-pepwave:latest
```

### 2. Set Up Secrets Manager

Run the secrets setup script to securely store your API keys:

```bash
./aws/setup-secrets.sh
```

This will prompt you to enter:
- PostgreSQL password
- Pinecone API key
- OpenAI API key
- LangSmith API key
- Cohere API key

### 3. Deploy to Fargate

Run the deployment script:

```bash
./aws/deploy.sh
```

This script will:
- Create ECS cluster
- Set up CloudWatch logging
- Create IAM roles
- Register task definition
- Create security groups
- Launch the service

### 4. Access Your Application

After deployment completes, the script will display the public IP address where your application is accessible:

```
🎉 Service available at: http://[PUBLIC_IP]:8000
```

## Configuration Details

### Task Definition
- **CPU**: 1024 (1 vCPU)
- **Memory**: 3072 MB (3 GB)
- **PostgreSQL Container**: 1024 MB
- **Web App Container**: 2048 MB

### Security
- All API keys stored in AWS Secrets Manager
- Security group allows traffic on ports 80, 443, and 8000
- Tasks run in public subnets with public IP assignment

### Health Checks
- **PostgreSQL**: `pg_isready -U postgres`
- **Web App**: HTTP check on `/api/threads` endpoint

## Monitoring and Troubleshooting

### CloudWatch Logs
View logs in CloudWatch at log group: `/ecs/langchain-pepwave`

### ECS Console
Monitor service health: https://console.aws.amazon.com/ecs/home?region=us-east-1#/clusters/langchain-pepwave-cluster/services

### Common Issues

1. **Service fails to start**: Check CloudWatch logs for container errors
2. **Health checks failing**: Ensure all environment variables and secrets are configured
3. **Database connection issues**: Verify PostgreSQL container is healthy
4. **API errors**: Check that all external API keys are valid and have sufficient quotas

## Scaling and Production Considerations

### For Production Use:

1. **Load Balancer**: Add an Application Load Balancer for better availability
2. **RDS**: Replace containerized PostgreSQL with managed RDS for persistence
3. **Auto Scaling**: Configure ECS service auto scaling based on CPU/memory
4. **SSL/TLS**: Add SSL certificate for HTTPS
5. **Custom Domain**: Set up Route 53 for custom domain
6. **Monitoring**: Add detailed CloudWatch dashboards and alarms

### Sample Production Architecture:

```bash
# Create ALB and target group
aws elbv2 create-load-balancer --name langchain-pepwave-alb --subnets subnet-xxx subnet-yyy --security-groups sg-xxx

# Create RDS PostgreSQL instance
aws rds create-db-instance --db-instance-identifier langchain-pepwave-db --db-instance-class db.t3.micro --engine postgres --master-username postgres --master-user-password [PASSWORD] --allocated-storage 20

# Update task definition to use RDS endpoint instead of localhost
```

## Cost Optimization

- **Fargate Pricing**: ~$0.04048/vCPU/hour + $0.004445/GB/hour
- **Estimated Monthly Cost**: ~$35-45 for 1 task running 24/7
- **Cost Reduction**: Use Fargate Spot for dev/test environments

## Cleanup

To remove all resources:

```bash
# Delete ECS service
aws ecs delete-service --cluster langchain-pepwave-cluster --service langchain-pepwave-service --force

# Delete ECS cluster
aws ecs delete-cluster --cluster langchain-pepwave-cluster

# Delete security group
aws ec2 delete-security-group --group-id [SECURITY_GROUP_ID]

# Delete secrets (optional)
aws secretsmanager delete-secret --secret-id langchain-pepwave/postgres-password
aws secretsmanager delete-secret --secret-id langchain-pepwave/pinecone-api-key
# ... etc for other secrets
```

## Support

For issues with the application itself, refer to the main README.md.
For AWS-specific deployment issues, check the AWS documentation or contact AWS support.
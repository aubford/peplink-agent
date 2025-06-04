#!/bin/bash

set -e

# Configuration
ACCOUNT_ID="003765455645"
REGION="us-east-1"
CLUSTER_NAME="langchain-pepwave-cluster"
SERVICE_NAME="langchain-pepwave-service"
TASK_DEFINITION_NAME="langchain-pepwave-task"
ECR_REPOSITORY="langchain-pepwave"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "🚀 Starting AWS Fargate deployment for LangChain-Pepwave..."

# Step 1: Create ECS Cluster if it doesn't exist
echo "📋 Creating/Updating ECS Cluster..."
aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION || echo "Cluster may already exist"

# Step 2: Create CloudWatch Log Group
echo "📝 Creating CloudWatch Log Group..."
aws logs create-log-group --log-group-name "/ecs/langchain-pepwave" --region $REGION || echo "Log group may already exist"

# Step 3: Create/Update IAM roles (if they don't exist)
echo "🔐 Checking IAM roles..."

# Check if execution role exists
if ! aws iam get-role --role-name ecsTaskExecutionRole >/dev/null 2>&1; then
    echo "Creating ECS Task Execution Role..."
    aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'
    
    aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
fi

# Check if task role exists
if ! aws iam get-role --role-name ecsTaskRole >/dev/null 2>&1; then
    echo "Creating ECS Task Role..."
    aws iam create-role --role-name ecsTaskRole --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ecs-tasks.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'
fi

# Step 4: Register Task Definition
echo "📋 Registering ECS Task Definition..."
TASK_DEF_ARN=$(aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json --region $REGION --query 'taskDefinition.taskDefinitionArn' --output text)
echo "Task Definition ARN: $TASK_DEF_ARN"

# Step 5: Get default VPC and subnets
echo "🌐 Getting VPC information..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text --region $REGION)
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[].SubnetId' --output text --region $REGION)
SUBNET_ARRAY=$(echo $SUBNET_IDS | tr ' ' ',' | sed 's/,$//')

echo "VPC ID: $VPC_ID"
echo "Subnets: $SUBNET_ARRAY"

# Step 6: Create Security Group
SECURITY_GROUP_NAME="langchain-pepwave-sg"
echo "🔒 Creating Security Group..."

# Check if security group already exists
EXISTING_SG=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$VPC_ID" --query 'SecurityGroups[0].GroupId' --output text --region $REGION 2>/dev/null || echo "None")

if [ "$EXISTING_SG" = "None" ]; then
    SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name $SECURITY_GROUP_NAME --description "Security group for LangChain Pepwave" --vpc-id $VPC_ID --region $REGION --query 'GroupId' --output text)
    
    # Allow HTTP traffic on port 8000
    aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region $REGION
    
    # Allow HTTPS traffic on port 443 (for ALB)
    aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 443 --cidr 0.0.0.0/0 --region $REGION
    
    # Allow HTTP traffic on port 80 (for ALB)
    aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 80 --cidr 0.0.0.0/0 --region $REGION
else
    SECURITY_GROUP_ID=$EXISTING_SG
fi

echo "Security Group ID: $SECURITY_GROUP_ID"

# Step 7: Create or Update ECS Service
echo "🚀 Creating/Updating ECS Service..."

# Check if service exists
if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $REGION --query 'services[0].serviceName' --output text 2>/dev/null | grep -q $SERVICE_NAME; then
    echo "Updating existing service..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --task-definition $TASK_DEF_ARN \
        --region $REGION
else
    echo "Creating new service..."
    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_DEF_ARN \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ARRAY],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
        --region $REGION
fi

echo "✅ Deployment complete!"
echo "🌐 Your service is starting up. It may take a few minutes to be fully available."
echo "📋 Check the ECS console for service status: https://console.aws.amazon.com/ecs/home?region=$REGION#/clusters/$CLUSTER_NAME/services"

# Get service endpoint
echo "🔍 Getting service endpoint..."
sleep 30  # Wait for service to start

TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME --region $REGION --query 'taskArns[0]' --output text)
if [ "$TASK_ARN" != "None" ] && [ "$TASK_ARN" != "" ]; then
    PUBLIC_IP=$(aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN --region $REGION --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text | xargs -I {} aws ec2 describe-network-interfaces --network-interface-ids {} --region $REGION --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
    
    if [ "$PUBLIC_IP" != "None" ] && [ "$PUBLIC_IP" != "" ]; then
        echo "🎉 Service available at: http://$PUBLIC_IP:8000"
    else
        echo "⏳ Service is still starting. Check ECS console for public IP."
    fi
else
    echo "⏳ Service is still starting. Check ECS console for status."
fi 
#!/bin/bash

set -e

REGION="us-east-1"

echo "🔐 Setting up AWS Secrets Manager for LangChain-Pepwave..."

# Function to create or update a secret
create_or_update_secret() {
    local secret_name=$1
    local secret_description=$2

    echo "Please enter your $secret_description:"
    read -rs secret_value

    # Check if secret exists
    if aws secretsmanager describe-secret --secret-id "$secret_name" --region $REGION >/dev/null 2>&1; then
        echo "Updating existing secret: $secret_name"
        aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "$secret_value" \
            --region $REGION
    else
        echo "Creating new secret: $secret_name"
        aws secretsmanager create-secret \
            --name "$secret_name" \
            --description "$secret_description" \
            --secret-string "$secret_value" \
            --region $REGION
    fi

    echo "✅ Secret $secret_name configured successfully"
    echo ""
}

echo "This script will help you securely store your API keys in AWS Secrets Manager."
echo "You'll be prompted to enter each API key. The input will be hidden for security."
echo ""

# Create/update all required secrets
create_or_update_secret "langchain-pepwave/POSTGRES_PASSWORD" "PostgreSQL Database Password"
create_or_update_secret "langchain-pepwave/PINECONE_API_KEY" "Pinecone API Key"
create_or_update_secret "langchain-pepwave/OPENAI_API_KEY" "OpenAI API Key"
create_or_update_secret "langchain-pepwave/COHERE_API_KEY" "Cohere API Key"

echo "🎉 All secrets have been configured successfully!"
echo "Your API keys are now securely stored in AWS Secrets Manager."
echo ""
echo "Next steps:"
echo "1. Make sure your ECS task execution role has access to these secrets"
echo "2. Run the deployment script: ./aws/deploy.sh"

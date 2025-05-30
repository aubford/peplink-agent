# LangChain Pepwave Deployment Guide

Deploy the LangChain Pepwave RAG application with PostgreSQL persistence locally or on AWS.

## Quick Start

### Local Development
```bash
cd langchain-pepwave
cp env.example .env
# Edit .env with your API keys
./deploy.sh local
```
App available at `http://localhost:8000`

### AWS Deployment
```bash
# One-time setup
./deploy.sh aws-setup

# Store secrets in AWS Parameter Store
aws ssm put-parameter --name "/copilot/langchain-pepwave/production/secrets/OPENAI_API_KEY" --value "your_key" --type "SecureString"
aws ssm put-parameter --name "/copilot/langchain-pepwave/production/secrets/PINECONE_API_KEY" --value "your_key" --type "SecureString"
aws ssm put-parameter --name "/copilot/langchain-pepwave/production/secrets/LANGSMITH_API_KEY" --value "your_key" --type "SecureString"
aws ssm put-parameter --name "/copilot/langchain-pepwave/production/secrets/COHERE_API_KEY" --value "your_key" --type "SecureString"
aws ssm put-parameter --name "/copilot/langchain-pepwave/production/secrets/POSTGRES_PASSWORD" --value "your_secure_password" --type "SecureString"

# Deploy updates
./deploy.sh aws-deploy

# Get service URL
copilot svc show --name web --env production
```

## Environment Variables

Required in `.env` file:
```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/langgraph?sslmode=disable
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-pepwave
```

## Architecture

- **PostgreSQL Checkpointer**: Persistent conversation storage via LangGraph
- **FastAPI**: RESTful API with streaming chat
- **Docker**: Containerized deployment
- **AWS ECS/Fargate**: Scalable cloud deployment with AWS Copilot

## Prerequisites

**Local**: Docker, Docker Compose
**AWS**: AWS CLI configured, AWS Copilot CLI (`brew install aws/tap/copilot-cli`)

## Commands

```bash
./deploy.sh local          # Run locally
./deploy.sh aws-setup      # Setup AWS infrastructure
./deploy.sh aws-deploy     # Deploy to AWS
./deploy.sh logs           # Show logs
./deploy.sh stop           # Stop local services
./deploy.sh clean          # Clean up Docker resources
./deploy.sh aws-destroy    # Destroy AWS infrastructure
```

## Database Access

**Local**: `docker-compose exec postgres psql -U postgres -d langgraph`
**AWS**: Managed Aurora Serverless in RDS Console

## Logs

**Local**: `docker-compose logs -f web`
**AWS**: `copilot svc logs --name web --env production --follow`

## Estimated AWS Costs (Low Traffic)
- Aurora Serverless: $5-20/month
- ECS Fargate: $10-30/month
- Load Balancer: $20/month
- **Total**: ~$35-70/month

## Troubleshooting

1. **PostgresCheckpointer import error**: `pip install langgraph-checkpoint-postgres psycopg[binary,pool]`
2. **Database connection failed**: Check `docker-compose ps` and DATABASE_URL
3. **AWS deployment issues**: Verify `aws sts get-caller-identity` and `copilot --version`
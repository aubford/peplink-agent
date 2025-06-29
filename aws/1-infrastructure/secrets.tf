# Data sources for existing secrets
data "aws_secretsmanager_secret" "postgres_password" {
  name = "langchain-pepwave/POSTGRES_PASSWORD"
}

data "aws_secretsmanager_secret_version" "postgres_password" {
  secret_id = data.aws_secretsmanager_secret.postgres_password.id
}

data "aws_secretsmanager_secret" "pinecone_api_key" {
  name = "langchain-pepwave/PINECONE_API_KEY"
}

data "aws_secretsmanager_secret_version" "pinecone_api_key" {
  secret_id = data.aws_secretsmanager_secret.pinecone_api_key.id
}

data "aws_secretsmanager_secret" "openai_api_key" {
  name = "langchain-pepwave/OPENAI_API_KEY"
}

data "aws_secretsmanager_secret_version" "openai_api_key" {
  secret_id = data.aws_secretsmanager_secret.openai_api_key.id
}

data "aws_secretsmanager_secret" "cohere_api_key" {
  name = "langchain-pepwave/COHERE_API_KEY"
}

data "aws_secretsmanager_secret_version" "cohere_api_key" {
  secret_id = data.aws_secretsmanager_secret.cohere_api_key.id
}
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources to get infrastructure from Phase 1
data "terraform_remote_state" "infrastructure" {
  backend = "local"
  config = {
    path = "../1-infrastructure/terraform.tfstate"
  }
}

# Validation to ensure Phase 1 is complete
locals {
  validate_phase1 = data.terraform_remote_state.infrastructure.outputs.ecr_repository_url != null ? true : file("ERROR: Phase 1 not complete. Run 'terraform apply' in 1-infrastructure/ first.")
}

# Data sources for secrets (needed for container definition)
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

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "langchain-pepwave-cluster"

  tags = {
    Name = "langchain-pepwave-cluster"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = "langchain-pepwave-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"

  container_definitions = jsonencode([
    {
      name      = "web"
      image     = "${var.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/langchain-pepwave:latest"
      essential = true
      memory    = 2048

      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "DATABASE_URL"
          value = "postgresql://postgres:${data.aws_secretsmanager_secret_version.postgres_password.secret_string}@${data.terraform_remote_state.infrastructure.outputs.rds_endpoint}:5432/langgraph?sslmode=require"
        },
        {
          name  = "PINECONE_API_KEY"
          value = data.aws_secretsmanager_secret_version.pinecone_api_key.secret_string
        },
        {
          name  = "OPENAI_API_KEY"
          value = data.aws_secretsmanager_secret_version.openai_api_key.secret_string
        },
        {
          name  = "COHERE_API_KEY"
          value = data.aws_secretsmanager_secret_version.cohere_api_key.secret_string
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = data.terraform_remote_state.infrastructure.outputs.cloudwatch_log_group_name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "web"
        }
      }
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "app" {
  name            = "langchain-pepwave-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.terraform_remote_state.infrastructure.outputs.subnet_ids
    assign_public_ip = true
  }
}
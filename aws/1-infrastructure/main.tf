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

# ECR Repository
resource "aws_ecr_repository" "app" {
  name                 = "langchain-pepwave"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "langchain-pepwave"
  }
}

# Data sources for existing resources
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# RDS Database
resource "aws_db_instance" "postgres" {
  identifier     = "langchain-pepwave-db"
  engine         = "postgres"
  engine_version = "15.12"
  instance_class = "db.t3.micro"

  allocated_storage = 20
  storage_type      = "gp2"
  storage_encrypted = true

  db_name  = "langgraph"
  username = "postgres"
  password = var.postgres_password

  parameter_group_name = aws_db_parameter_group.postgres.name

  skip_final_snapshot = true
  publicly_accessible = true

  vpc_security_group_ids = [aws_security_group.rds.id]

  tags = {
    Name = "langchain-pepwave-db"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/langchain-pepwave"
  retention_in_days = 7
}

# ECS Task Execution Role
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "langchain-pepwave-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "langchain-pepwave-ecs-execution-role"
  }
}

# Attach AWS managed policy for ECS task execution
resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Role (for ECS Exec)
resource "aws_iam_role" "ecs_task_role" {
  name = "langchain-pepwave-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Add SSM permissions for ECS Exec
resource "aws_iam_role_policy" "ecs_exec_policy" {
  name = "ecs-exec-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssmmessages:CreateControlChannel",
          "ssmmessages:CreateDataChannel",
          "ssmmessages:OpenControlChannel",
          "ssmmessages:OpenDataChannel"
        ]
        Resource = "*"
      }
    ]
  })
}

# Security Group for ECS tasks (current - keeps direct access)
resource "aws_security_group" "ecs_tasks" {
  name        = "langchain-pepwave-ecs-tasks"
  description = "Security group for ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  # Allow inbound HTTP traffic from anywhere (direct access)
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP traffic for web application"
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "langchain-pepwave-ecs-tasks"
  }
}

# Security Group for RDS
resource "aws_security_group" "rds" {
  name        = "langchain-pepwave-rds"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = data.aws_vpc.default.id

  # Allow inbound PostgreSQL traffic from ECS tasks
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
    description     = "PostgreSQL traffic from ECS tasks"
  }

  # Allow inbound PostgreSQL traffic from anywhere (for demo/debugging)
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "PostgreSQL traffic from anywhere (demo only)"
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "langchain-pepwave-rds"
  }
}

# RDS Parameter Group
# TODO: Remove (not critical)
resource "aws_db_parameter_group" "postgres" {
  family = "postgres15"
  name   = "langchain-pepwave-postgres-params"

  tags = {
    Name = "langchain-pepwave-postgres-params"
  }
}

# Security Group for ALB
resource "aws_security_group" "alb" {
  name        = "langchain-pepwave-alb"
  description = "Security group for Application Load Balancer"
  vpc_id      = data.aws_vpc.default.id

  # Allow inbound HTTP traffic
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP traffic"
  }

  # Allow inbound HTTPS traffic (optional, for future SSL)
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS traffic"
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = {
    Name = "langchain-pepwave-alb"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "langchain-pepwave-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids

  enable_deletion_protection = false

  tags = {
    Name = "langchain-pepwave-alb"
  }
}

# Target Group for ECS tasks
resource "aws_lb_target_group" "app" {
  name        = "langchain-pepwave-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
∫
  tags = {
    Name = "langchain-pepwave-tg"
  }
}

# ALB Listener
resource "aws_lb_listener" "app" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }

  tags = {
    Name = "langchain-pepwave-listener"
  }
}


output "ecs_task_role_arn" {
  description = "ECS task role ARN"
  value       = aws_iam_role.ecs_task_role.arn
}
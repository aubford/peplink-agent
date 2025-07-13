output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.app.repository_url
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.postgres.endpoint
}

output "vpc_id" {
  description = "Default VPC ID"
  value       = data.aws_vpc.default.id
}

output "subnet_ids" {
  description = "Default subnet IDs"
  value       = data.aws_subnets.default.ids
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.app.name
}

output "ecs_task_execution_role_arn" {
  description = "ECS task execution role ARN"
  value       = aws_iam_role.ecs_task_execution_role.arn
}

output "ecs_security_group_id" {
  description = "Security group ID for ECS tasks"
  value       = aws_security_group.ecs_tasks.id
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB Zone ID"
  value       = aws_lb.main.zone_id
}

output "target_group_arn" {
  description = "Target group ARN"
  value       = aws_lb_target_group.app.arn
}

output "alb_url" {
  description = "Application URL"
  value       = "http://${aws_lb.main.dns_name}"
}

# Domain-related outputs
output "custom_domain_url" {
  description = "Custom domain URL (if domain is configured)"
  value       = var.domain_name != "" ? "https://${var.subdomain}.${var.domain_name}" : "Domain not configured"
}

output "route53_zone_id" {
  description = "Route 53 hosted zone ID (if domain is configured)"
  value       = var.domain_name != "" ? aws_route53_zone.main[0].zone_id : null
}

output "ssl_certificate_arn" {
  description = "SSL certificate ARN (if domain is configured)"
  value       = var.domain_name != "" ? aws_acm_certificate.app[0].arn : null
}
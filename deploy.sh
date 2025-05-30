#!/bin/bash

# LangChain Pepwave Deployment Script
# Supports local development and AWS ECS deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    echo "LangChain Pepwave Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  local         Run locally with docker-compose"
    echo "  build         Build Docker image"
    echo "  aws-setup     Set up AWS infrastructure using Copilot"
    echo "  aws-deploy    Deploy to AWS ECS"
    echo "  aws-destroy   Destroy AWS infrastructure"
    echo "  logs          Show application logs"
    echo "  stop          Stop local containers"
    echo "  clean         Clean up Docker resources"
    echo ""
    echo "Examples:"
    echo "  $0 local                    # Run locally"
    echo "  $0 aws-setup               # Set up AWS infrastructure"
    echo "  $0 aws-deploy              # Deploy to AWS"
    echo ""
}

# Check if required tools are installed
check_dependencies() {
    local deps=("docker" "docker-compose")

    if [[ "$1" == "aws"* ]]; then
        deps+=("aws" "copilot")
    fi

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &>/dev/null; then
            print_error "$dep is required but not installed."
            if [[ "$dep" == "copilot" ]]; then
                print_info "Install AWS Copilot: https://aws.github.io/copilot-cli/docs/getting-started/install/"
            fi
            exit 1
        fi
    done
}

# Check if .env file exists
check_env_file() {
    if [[ ! -f .env ]]; then
        print_warning ".env file not found. Creating from env.example..."
        if [[ -f env.example ]]; then
            cp env.example .env
            print_warning "Please edit .env file with your actual API keys before continuing."
            exit 1
        else
            print_error "env.example file not found. Please create .env file manually."
            exit 1
        fi
    fi
}

# Local development
run_local() {
    print_info "Running locally with docker-compose..."
    check_env_file

    # Build and start services
    docker-compose up --build -d

    print_info "Services starting up..."
    print_info "Web app will be available at http://localhost:8000"
    print_info "Use 'docker-compose logs -f' to follow logs"
    print_info "Use '$0 stop' to stop services"
}

# Build Docker image
build_image() {
    print_info "Building Docker image..."
    docker build -t langchain-pepwave:latest .
    print_info "Docker image built successfully"
}

# AWS setup using Copilot
aws_setup() {
    check_dependencies "aws-setup"
    print_info "Setting up AWS infrastructure with Copilot..."

    # Initialize Copilot app if not exists
    if [[ ! -d copilot ]]; then
        print_error "Copilot configuration not found. Make sure copilot/ directory exists."
        exit 1
    fi

    # Check if app exists
    if ! copilot app ls | grep -q "langchain-pepwave"; then
        print_info "Initializing Copilot application..."
        copilot app init langchain-pepwave
    fi

    # Deploy environment
    print_info "Deploying production environment..."
    copilot env deploy --name production

    # Deploy services
    print_info "Deploying database service..."
    copilot svc deploy --name postgres --env production

    print_info "Deploying web service..."
    copilot svc deploy --name web --env production

    print_info "AWS infrastructure setup complete!"
    print_info "Use 'copilot svc show' to see service details"
}

# AWS deployment
aws_deploy() {
    check_dependencies "aws-deploy"
    print_info "Deploying to AWS ECS..."

    # Deploy web service
    copilot svc deploy --name web --env production

    print_info "Deployment complete!"
    print_info "Use 'copilot svc show --name web --env production' to see service URL"
}

# Destroy AWS infrastructure
aws_destroy() {
    check_dependencies "aws-destroy"
    print_warning "This will destroy ALL AWS resources for the langchain-pepwave application."
    read -p "Are you sure? (y/N): " confirm

    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        print_info "Destroying AWS infrastructure..."
        copilot app delete
        print_info "AWS infrastructure destroyed."
    else
        print_info "Cancelled."
    fi
}

# Show logs
show_logs() {
    if docker-compose ps | grep -q "Up"; then
        print_info "Showing local logs..."
        docker-compose logs -f
    else
        print_info "No local containers running. For AWS logs, use:"
        print_info "copilot svc logs --name web --env production --follow"
    fi
}

# Stop local services
stop_local() {
    print_info "Stopping local services..."
    docker-compose down
    print_info "Local services stopped."
}

# Clean up Docker resources
clean_docker() {
    print_info "Cleaning up Docker resources..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_info "Docker cleanup complete."
}

# Main script logic
case "${1:-}" in
"local")
    check_dependencies "local"
    run_local
    ;;
"build")
    check_dependencies "build"
    build_image
    ;;
"aws-setup")
    aws_setup
    ;;
"aws-deploy")
    aws_deploy
    ;;
"aws-destroy")
    aws_destroy
    ;;
"logs")
    show_logs
    ;;
"stop")
    stop_local
    ;;
"clean")
    clean_docker
    ;;
"help" | "-h" | "--help")
    show_help
    ;;
"")
    print_error "No command specified."
    show_help
    exit 1
    ;;
*)
    print_error "Unknown command: $1"
    show_help
    exit 1
    ;;
esac

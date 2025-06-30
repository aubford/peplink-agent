.PHONY: help install dev web chat test clean lint format

# Virtual environment setup
VENV_DIR = langchain_pepwave_env
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# Create virtual environment if it doesn't exist
$(VENV_DIR):
	python -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

# Install dependencies
install: $(VENV_DIR)
	$(PIP) install -r requirements.txt

# Install development dependencies
dev: install
	$(PIP) install pytest black flake8 mypy

# Start the web application
web: $(VENV_DIR)
	@echo "🚀 Starting Pepwave ChatBot Web Application..."
	@echo "🌐 Open http://localhost:8000 in your browser"
	$(PYTHON) -m web_app

# Start the CLI chatbot
chat: $(VENV_DIR)
	@echo "🤖 Starting CLI ChatBot..."
	$(PYTHON) inference/chat_agentic.py

# Run tests
test: $(VENV_DIR)
	$(PYTHON) -m pytest tests/ -v

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Development server with auto-reload
dev-web: $(VENV_DIR)
	@echo "🔄 Starting development server with auto-reload..."
	@echo "🌐 Open http://localhost:8000 in your browser"
	$(PYTHON) -m uvicorn web_app.app:app --reload --host 0.0.0.0 --port 8000
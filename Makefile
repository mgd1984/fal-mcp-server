.PHONY: help install install-dev test lint format type-check security-check clean pre-commit setup-hooks run-checks all

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pip install pre-commit

# Testing
test: ## Run tests
	python test_server.py
	@echo "✅ All tests passed!"

test-verbose: ## Run tests with verbose output
	python -v test_server.py

# Code quality
lint: ## Run linting (flake8)
	flake8 main.py test_server.py examples.py

format: ## Format code with black and isort
	black --line-length 88 main.py test_server.py examples.py
	isort main.py test_server.py examples.py

format-check: ## Check if code is properly formatted
	black --check --line-length 88 main.py test_server.py examples.py
	isort --check-only main.py test_server.py examples.py

type-check: ## Run type checking with mypy
	mypy main.py test_server.py examples.py

security-check: ## Run security checks with bandit
	bandit -r . -f json -o bandit-report.json || true
	@echo "Security report generated: bandit-report.json"

# Pre-commit
setup-hooks: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

run-hooks: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Comprehensive checks
run-checks: format lint type-check security-check test ## Run all code quality checks
	@echo "🎉 All checks passed! Ready to commit."

pre-commit: run-checks ## Alias for run-checks

# Project management
clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -f bandit-report.json

setup: install-dev setup-hooks ## Complete development setup
	@echo "🚀 Development environment setup complete!"
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and add your FAL_KEY"
	@echo "  2. Run 'make test' to verify everything works"
	@echo "  3. Run 'make run-checks' before committing"

# Environment checks
check-env: ## Check if environment is properly configured
	@echo "🔍 Checking environment..."
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✅ FAL_KEY configured' if os.getenv('FAL_KEY') else '❌ FAL_KEY not found - check your .env file')"
	@python -c "import main; print('✅ Server imports successfully')"

# Development server
dev-server: check-env ## Run the MCP server for development
	@echo "🚀 Starting FAL.ai MCP Server..."
	@echo "Press Ctrl+C to stop"
	python main.py

# Git helpers
git-status: ## Show git status with ignored files
	git status
	@echo "\nIgnored files that exist:"
	@git ls-files --others --ignored --exclude-standard | head -10

# All-in-one targets
all: clean setup run-checks ## Clean, setup, and run all checks
	@echo "🎯 Project is ready for development!"

validate: run-checks check-env ## Validate the entire project
	@echo "✨ Project validation complete!"

# Quick development cycle
quick-check: format lint test ## Quick checks for development cycle
	@echo "⚡ Quick checks passed!"

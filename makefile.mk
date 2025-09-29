# LLM from Scratch - Development Makefile
# Usage: make <target>

.PHONY: help install test test-all test-part1 test-part2 test-part3 test-part4 clean format lint type-check pre-commit setup-dev

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install all dependencies
	uv sync --all-extras

setup-dev: install ## Set up development environment
	uv run pre-commit install

# Testing targets
test: ## Run all tests
	uv run pytest

test-fast: ## Run all tests excluding slow ones
	uv run pytest -m "not slow"

test-part1: ## Run tests for part 1 (Transformer Architecture)
	uv run pytest part_1/tests/ -v

test-part2: ## Run tests for part 2 (Training a Tiny LLM)
	uv run pytest part_2/tests/ -v

test-part3: ## Run tests for part 3 (Modern Architecture)
	uv run pytest part_3/tests/ -v

test-part4: ## Run tests for part 4 (Scaling Up)
	uv run pytest part_4/tests/ -v

test-part5: ## Run tests for part 5 (Mixture-of-Experts)
	uv run pytest part_5/tests/ -v

test-part6: ## Run tests for part 6 (Supervised Fine-Tuning)
	uv run pytest part_6/tests/ -v

test-part7: ## Run tests for part 7 (Reward Modeling)
	uv run pytest part_7/tests/ -v

test-part8: ## Run tests for part 8 (RLHF with PPO)
	uv run pytest part_8/tests/ -v

test-part9: ## Run tests for part 9 (RLHF with GRPO)
	uv run pytest part_9/tests/ -v

test-with-coverage: ## Run tests with coverage report
	uv run pytest --cov=. --cov-report=html --cov-report=term-missing

test-single: ## Run a single test (usage: make test-single TEST=part_1/tests/test_attn_math.py::test_single_head_matches_numpy)
	uv run pytest $(TEST) -v

# Code quality targets
format: ## Format code with black and isort
	uv run black .
	uv run isort .

format-check: ## Check code formatting without making changes
	uv run black --check .
	uv run isort --check-only .

lint: ## Run linting checks
	uv run black --check .
	uv run isort --check-only .

type-check: ## Run type checking with mypy
	uv run mypy part_1/ --ignore-missing-imports
	uv run mypy part_2/ --ignore-missing-imports
	uv run mypy part_3/ --ignore-missing-imports
	uv run mypy part_4/ --ignore-missing-imports

pre-commit: ## Run all pre-commit hooks
	uv run pre-commit run --all-files

# Running specific parts
run-part1-demo: ## Run attention numpy demo from part 1
	cd part_1 && uv run python attn_numpy_demo.py

run-part1-shapes: ## Run multi-head attention shapes demo
	cd part_1 && uv run python demo_mha_shapes.py

run-part2-train: ## Run training from part 2 (tiny dataset)
	cd part_2 && uv run python train.py

run-part2-sample: ## Run sampling from part 2
	cd part_2 && uv run python sample.py

run-part2-orchestrator: ## Run the full part 2 orchestrator (train, sample, eval)
	cd part_2 && uv run python orchestrator.py

run-part3-generate: ## Run generation demo from part 3
	cd part_3 && uv run python demo_generate.py

# Utility targets
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -delete
	rm -rf .venv/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

clean-cache: ## Clean only cache files (keep venv)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -delete

# Development workflow shortcuts
dev: setup-dev ## Full development setup
	@echo "Development environment ready!"
	@echo "Try: make test-part1"

quick-check: format-check test-fast ## Quick pre-commit check

full-check: format-check type-check test ## Full validation check

# Info targets
info: ## Show project information
	@echo "LLM from Scratch Project"
	@echo "========================"
	@echo "Python version: $(shell uv run python --version)"
	@echo "uv version: $(shell uv --version)"
	@echo "Virtual env: $(shell echo $$VIRTUAL_ENV)"
	@echo ""
	@echo "Available parts:"
	@ls -1 part_*/
	@echo ""
	@echo "Use 'make help' to see all available commands"

# Examples for common workflows
examples: ## Show example usage
	@echo "Common development workflows:"
	@echo ""
	@echo "  Initial setup:"
	@echo "    make dev"
	@echo ""
	@echo "  Run specific test:"
	@echo "    make test-single TEST=part_1/tests/test_attn_math.py::test_single_head_matches_numpy"
	@echo ""
	@echo "  Quick development cycle:"
	@echo "    make format && make test-part1"
	@echo ""
	@echo "  Full validation:"
	@echo "    make full-check"
	@echo ""
	@echo "  Run demos:"
	@echo "    make run-part1-demo"
	@echo "    make run-part2-train"
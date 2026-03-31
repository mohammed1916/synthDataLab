.PHONY: help test lint format typecheck check run-mock run docker-build docker-run clean

PYTHON     := python3
PY_SRC     := dataset_builder
DOCKER_TAG := synthdatalab:latest

## ── Help ─────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## ── Testing ──────────────────────────────────────────────────────────────────

test: ## Run the full test suite
	cd $(PY_SRC) && $(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	cd $(PY_SRC) && $(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=. --cov-report=term-missing --cov-report=html:../htmlcov
	@echo "Coverage report: htmlcov/index.html"

## ── Linting & Formatting ─────────────────────────────────────────────────────

lint: ## Check code with ruff (no fixes)
	$(PYTHON) -m ruff check $(PY_SRC)/

format: ## Auto-format code with ruff
	$(PYTHON) -m ruff format $(PY_SRC)/

format-check: ## Check formatting without modifying files
	$(PYTHON) -m ruff format --check $(PY_SRC)/

typecheck: ## Run mypy type checking
	$(PYTHON) -m mypy $(PY_SRC)/ --ignore-missing-imports

check: lint format-check typecheck ## Run all quality checks (CI equivalent)

## ── Running the pipeline ─────────────────────────────────────────────────────

run-mock: ## Run the full pipeline in mock (offline) mode
	cd $(PY_SRC) && $(PYTHON) main.py run-all --mock

run-mock-parallel: ## Run the full pipeline with 4 parallel mock workers
	cd $(PY_SRC) && $(PYTHON) main.py run-all --mock --workers 4

run: ## Run the full pipeline with the real Ollama backend
	cd $(PY_SRC) && $(PYTHON) main.py run-all

run-resume: ## Resume a previous run-all (skips completed steps)
	cd $(PY_SRC) && $(PYTHON) main.py run-all --resume

## ── Docker ───────────────────────────────────────────────────────────────────

docker-build: ## Build the Docker image
	docker build -t $(DOCKER_TAG) .

docker-run: ## Run the pipeline in Docker (mock mode, data persisted in ./data-out)
	mkdir -p data-out
	docker run --rm \
		-v "$(PWD)/data-out:/app/dataset_builder/data" \
		$(DOCKER_TAG) run-all --mock

docker-shell: ## Open a shell inside the Docker image for debugging
	docker run --rm -it --entrypoint /bin/bash \
		-v "$(PWD)/data-out:/app/dataset_builder/data" \
		$(DOCKER_TAG)

## ── Housekeeping ─────────────────────────────────────────────────────────────

clean: ## Remove build artefacts, caches, and generated data
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name '*.pyc' -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf dist/ build/ *.egg-info/
	rm -f $(PY_SRC)/data/*.jsonl $(PY_SRC)/data/*.json
	rm -rf $(PY_SRC)/data/logs/
	@echo "Clean complete."

install-dev: ## Install project + dev dependencies in the current environment
	$(PYTHON) -m pip install -e ".[dev]"

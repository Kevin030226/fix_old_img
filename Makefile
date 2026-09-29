# fix_old_img — developer entry points (plan §3.2)
#
#   make help          list targets
#   make install       editable install with the test extras
#   make test          unit + api + inference + integration tests (no GPU)
#   make lint          ruff + mypy
#   make bench         latency / throughput / memory benchmarks
#
# Windows users: run these from Git Bash, or use the equivalent commands in
# scripts/*.bat.

PYTHON ?= python
SRC    := src
PKG    := fiximg
#: Exactly what CI lints. A local `make lint` that checked less than the workflow
#: is a green light that CI can still refuse — the entry-point scripts and the
#: helper scripts in scripts/ used to be invisible to the local target.
LINT_PATHS := src tests benchmark scripts main.py worker.py run.py

.DEFAULT_GOAL := help
.PHONY: help install install-gpu install-all test test-gpu test-all lint fmt typecheck \
        bench bench-latency bench-throughput bench-memory serve worker migrate weights \
        verify-weights clean tree

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------- install
install: ## Editable install with test/dev extras (CPU-friendly)
	$(PYTHON) -m pip install -e ".[test,dev]"

install-gpu: ## Editable install with the GPU inference stack
	$(PYTHON) -m pip install -e ".[gpu,test,dev]"

install-all: ## Editable install with every optional extra
	$(PYTHON) -m pip install -e ".[gpu,redis,s3,otel,test,benchmark,dev]"

# ----------------------------------------------------------------------- tests
test: ## Run the CPU test suite (GPU-marked tests deselected)
	$(PYTHON) -m pytest -m "not gpu"

test-gpu: ## Run the GPU-marked tests (needs weights + CUDA)
	$(PYTHON) -m pytest -m gpu

test-all: ## Run every test including GPU
	$(PYTHON) -m pytest

test-postgres: ## Data path against a real PostgreSQL (needs FIXIMG_TEST_POSTGRES_URL)
	$(PYTHON) -m pytest tests/integration/test_postgres_server.py -q

cov: ## Run the CPU suite with coverage
	$(PYTHON) -m pytest -m "not gpu" --cov=$(PKG) --cov-report=term-missing

# ----------------------------------------------------------------------- quality
lint: ## Ruff lint (same path list as CI)
	$(PYTHON) -m ruff check $(LINT_PATHS)

syntax: ## Parse every file with this interpreter's grammar (CI runs it on 3.11)
	$(PYTHON) -m compileall -q $(LINT_PATHS)

fmt: ## Ruff format + autofix
	$(PYTHON) -m ruff check --fix $(LINT_PATHS)
	$(PYTHON) -m ruff format $(SRC) tests benchmark scripts

typecheck: ## mypy over the package
	$(PYTHON) -m mypy

# -------------------------------------------------------------------- benchmarks
bench: bench-latency bench-throughput bench-memory ## Run every benchmark

bench-latency: ## p50/p95 latency per task type
	$(PYTHON) -m benchmark.latency

bench-throughput: ## Queue throughput / worker saturation
	$(PYTHON) -m benchmark.throughput

bench-memory: ## Peak RSS (+ GPU) per stage
	$(PYTHON) -m benchmark.memory

# --------------------------------------------------------------------- runtime
serve: ## Start the web service (FastAPI + Gradio + inline worker)
	$(PYTHON) main.py

worker: ## Start a standalone GPU worker process
	$(PYTHON) worker.py

# ------------------------------------------------------------------- operations
db-status: ## Show the applied schema revision and anything pending
	$(PYTHON) -m $(PKG).infrastructure.db.migrations.runner status

db-upgrade: ## Apply pending schema migrations (alembic upgrade head)
	$(PYTHON) -m $(PKG).infrastructure.db.migrations.runner upgrade

db-downgrade: ## Roll the schema back one revision
	$(PYTHON) -m $(PKG).infrastructure.db.migrations.runner downgrade

db-history: ## Print the migration chain
	$(PYTHON) -m $(PKG).infrastructure.db.migrations.runner history

db-adopt: ## Stamp an existing database as current (no migrations run)
	$(PYTHON) -m $(PKG).infrastructure.db.migrations.runner stamp

db-check: ## Fail if rows are still stored in a format the code no longer writes
	$(PYTHON) -m $(PKG).infrastructure.db.migrations.runner check

migrate: ## Migrate V1 history rows into the V2/V3 task tables (dry-run)
	$(PYTHON) -m $(PKG).cli.migrate_v1

weights: ## Download model weights and rebuild the integrity manifest
	$(PYTHON) -m $(PKG).cli.download_weights download

verify-weights: ## Verify model weights against the manifest
	$(PYTHON) -m $(PKG).cli.verify_weights

lock: ## Regenerate the per-group locked requirement files (report §3.12)
	$(PYTHON) scripts/export_locks.py

lock-check: ## Verify requirements/<group>.txt are up to date
	$(PYTHON) scripts/export_locks.py --check

lock-freeze: ## Refresh requirements.lock from the current environment
	$(PYTHON) scripts/export_locks.py --freeze
	@echo "Now run 'make lock' to re-split the groups."

smoke-services: ## Verify Redis / database / object storage connectivity (plan §3.13)
	$(PYTHON) scripts/smoke_services.py

tree: ## Print the source tree
	@find $(SRC) -name '__pycache__' -prune -o -type f -name '*.py' -print | sort

clean: ## Remove caches and build artefacts
	@rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +

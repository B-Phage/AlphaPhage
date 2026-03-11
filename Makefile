# =====================
# Variables
# =====================

# Root folder for conda environments
CONDA_ROOT = conda_envs

# List of packages and their environments
# TODO: add the rest of the packages here
PACKAGES = utils orchestration variants hmm kaptive3api storage_io models training dataloaders external_db_provisioning

# Mapping environment names
ENV_PREPROCESSING = preprocessing
ENV_KAPTIVE3API = kaptive3api
ENV_TRAINING = training

# Define the config file path
DVC_CONFIG_LOCAL = .dvc/config.local

# TODO: update this list as the packages are implemented
ENV_LIST = $(ENV_PREPROCESSING) $(ENV_TRAINING) $(ENV_KAPTIVE3API)

# External Data Provisioning
HMM_YAML := ml_assets/config/hmm_data_v0.0.1.setup.yaml
KAPTIVE_YAML := ml_assets/config/kaptive3api_v0.0.1.setup.yaml
PYTHON := $(shell which python)  # uses the environment's python


# =====================
# Help
# =====================
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make setup-all       		- Setup all environments, install packages, and setup hooks"
	@echo "  make setup-<pkg>     		- Setup a single package environment and install it"
	@echo "  make check-dvc-config     	- Check the dvc OAuth"
	@echo "  make hooks           		- Install pre-commit and pre-push hooks in the current environment"
	@echo "  make secrets-cli     		- Installing detect-secrets CLI tool"
	@echo "  make secrets-baseline      	- Scanning repository and generating .secrets.baseline file..."
	@echo "  make hooks-update    		- Update pre-commit hooks and reinstall"
	@echo "  make dvc-<target>    		- Run DVC commands (pull, repro, push, dag, clean)"
	@echo "  make setup-hmm-data		- Pull the HMM data from external dependencies and extract"
	@echo "  make clean           		- Remove all conda environments"

# =====================
# Helper: Ensure conda env exists (create/update)
# =====================
define ensure_env
	@if conda env list | grep -q "^$1"; then \
		echo "=== Conda env '$1' exists, updating ==="; \
		conda env update -f conda_envs/$1.yml -n $1; \
	else \
		echo "=== Creating conda env '$1' ==="; \
		conda env create -f conda_envs/$1.yml -n $1; \
	fi
endef

# e.g. conda env create  --subdir osx-64 --file kaptive3api_hist.yml
# Also lock into osx-64:
# conda config --env --set subdir osx-64
define ensure_env_mac_emul
	@if conda env list | grep -q "^$1"; then \
		echo "=== Conda env '$1' exists, updating ==="; \
		conda env update -f conda_envs/$1.yml -n $1; \
	else \
		echo "=== Creating conda env '$1' ==="; \
		conda env create --subdir osx-64 -f conda_envs/$1.yml -n $1; \
		conda config --env --set subdir osx-64; \
	fi
endef

# =====================
# Main Setup
# =====================
.PHONY: setup-all
setup-all: \
    $(foreach pkg,$(PACKAGES),setup-$(pkg)) \
    hooks \
    setup-external-data \
    check-dvc-config \
    dvc-pull
	@echo "All environments, packages, hooks, and data are ready."

.PHONY: setup-external-data
setup-external-data: setup-hmm-data setup-kaptive-data
	@echo "External database provisioning complete."

# =====================
# Package-specific setups
# =====================
.PHONY: setup-utils
setup-utils:
	$(call ensure_env,$(ENV_PREPROCESSING))
	@echo "=== Installing utils package ==="
	conda run -n $(ENV_PREPROCESSING) pip install -e packages/utils

.PHONY: setup-orchestration
setup-orchestration:
	$(call ensure_env,$(ENV_PREPROCESSING))
	@echo "=== Installing orchestration package ==="
	conda run -n $(ENV_PREPROCESSING) pip install -e packages/orchestration

.PHONY: setup-variants
setup-variants:
	$(call ensure_env,$(ENV_PREPROCESSING))
	@echo "=== Installing variants package ==="
	conda run -n $(ENV_PREPROCESSING) pip install -e packages/variants

.PHONY: setup-representation
setup-representation:
	$(call ensure_env,$(ENV_PREPROCESSING))
	@echo "=== Installing representation package ==="
	conda run -n $(ENV_PREPROCESSING) pip install -e packages/representation

.PHONY: setup-kaptive3api
setup-kaptive3api:
	$(call ensure_env_mac_emul,$(ENV_KAPTIVE3API))
	@echo "=== Installing kaptive3api package ==="
	CONDA_SUBDIR=osx-64 conda run -n $(ENV_KAPTIVE3API) pip install -e packages/kaptive3api

.PHONY: setup-hmm
setup-hmm:
	$(call ensure_env,$(ENV_PREPROCESSING))
	@echo "=== Installing hmm preprocessing package ==="
	conda run -n $(ENV_PREPROCESSING) pip install -e packages/hmm

.PHONY: setup-external_db_provisioning
setup-external_db_provisioning:
	$(call ensure_env,$(ENV_PREPROCESSING))
	@echo "=== Installing external DB provisioning package ==="
	conda run -n $(ENV_PREPROCESSING) pip install -e packages/external_db_provisioning

.PHONY: setup-storage_io
setup-storage_io:
	$(call ensure_env,$(ENV_PREPROCESSING))
	@echo "=== Installing storage input/output package ==="
	conda run -n $(ENV_PREPROCESSING) pip install -e packages/storage_io

.PHONY: setup-training
setup-training:
	$(call ensure_env,$(ENV_TRAINING))
	@echo "=== Installing training package ==="
	conda run -n $(ENV_TRAINING) pip install -e packages/training

.PHONY: setup-dataloaders
setup-dataloaders:
	$(call ensure_env,$(ENV_TRAINING))
	@echo "=== Installing dataloaders package ==="
	conda run -n $(ENV_TRAINING) pip install -e packages/dataloaders

.PHONY: setup-models
setup-models:
	$(call ensure_env,$(ENV_TRAINING))
	@echo "=== Installing models package ==="
	conda run -n $(ENV_TRAINING) pip install -e packages/models

.PHONY: setup-inference
setup-inference:
# 	$(call ensure_env,$(ENV_INFERENCE))
	@echo "=== Installing inference package ==="
# 	conda run -n $(ENV_INFERENCE) pip install -e packages/inference

# =====================
# Pre-commit hooks
# =====================
.PHONY: hooks
hooks:
	@echo "=== Installing pre-commit and pre-push hooks in all environments ==="
	$(foreach env,$(ENV_LIST),conda run -n $(env) pre-commit install;)
	$(foreach env,$(ENV_LIST),conda run -n $(env) pre-commit install --hook-type pre-push;)
	@echo "Pre-commit hooks and pre-push installed successfully."

.PHONY: hooks-update
hooks-update:
	@echo "=== Updating pre-commit hooks in all environments ==="
	$(foreach env,$(ENV_LIST),conda run -n $(env) pre-commit autoupdate;)
	$(foreach env,$(ENV_LIST),conda run -n $(env) pre-commit install;)
	$(foreach env,$(ENV_LIST),conda run -n $(env) pre-commit install --hook-type pre-push;)
	@echo "Pre-commit hooks updated successfully."

# =====================
# Detect secrets
# =====================
.PHONY: secrets-cli
install-secrets-cli:
	@echo "Installing detect-secrets CLI in $(ENV_PREPROCESSING)..."
	conda run -n $(ENV_PREPROCESSING) pip install detect-secrets

.PHONY: secrets-baseline
init-secrets-baseline: install-secrets-cli
	@echo "Generating .secrets.baseline..."
	conda run -n $(ENV_PREPROCESSING) detect-secrets scan > .secrets.baseline
	@echo "Baseline created. Add and commit .secrets.baseline"

# =====================
# External Data Provisioning
# =====================

.PHONY: setup-hmm-data
setup-hmm-data:
	@echo "DEBUG: Using config $(HMM_YAML)"
	PYTHONPATH=$(shell pwd)/packages:$(PYTHONPATH) conda run -n $(ENV_PREPROCESSING) \
		python -m external_db_provisioning.hmm_data_setup --config $(HMM_YAML)

.PHONY: setup-kaptive-data
setup-kaptive-data:
	@echo "DEBUG: Using config $(KAPTIVE_YAML)"
	PYTHONPATH=$(shell pwd)/packages:$(PYTHONPATH) conda run -n $(ENV_PREPROCESSING) \
		python -m external_db_provisioning.kaptive_data_setup --config $(KAPTIVE_YAML)


# =====================
# DVC targets
# =====================

# Target to check the DVC authentication config
.PHONY: check-dvc-config
check-dvc-config:
	@echo "Checking local DVC config..."
	@if [ ! -f "$(DVC_CONFIG_LOCAL)" ]; then \
		echo "ERROR: $(DVC_CONFIG_LOCAL) does not exist."; \
		echo "Please create it and add your:"; \
		echo "		gdrive_client_id"; \
		echo "		gdrive_client_secret"; \
		echo "See CONTRIBUTING.md -> 2.3. Configure DVC Data Authentication"; \
		exit 1; \
	fi
	@if ! grep -q "gdrive_client_id" "$(DVC_CONFIG_LOCAL)" || ! grep -q "gdrive_client_secret" "$(DVC_CONFIG_LOCAL)"; then \
		echo "ERROR: $(DVC_CONFIG_LOCAL) is missing gdrive_client_id or gdrive_client_secret."; \
		echo "Please add your OAuth client credentials to the file."; \
		exit 1; \
	fi
	@echo "Local DVC config looks OK."

.PHONY: dvc-pull
dvc-pull:
	@echo "=== Pulling all DVC-tracked data and models ==="
	conda run -n $(ENV_PREPROCESSING) dvc pull

.PHONY: dvc-repro
dvc-repro:
	@echo "=== Reproducing DVC pipeline ==="
	conda run -n $(ENV_PREPROCESSING) dvc repro

.PHONY: dvc-graph
dvc-graph:
	@echo "=== Showing DVC pipeline graph ==="
	conda run -n $(ENV_PREPROCESSING) dvc dag

.PHONY: dvc-push
dvc-push:
	@echo "=== Pushing outputs to remote ==="
	conda run -n $(ENV_PREPROCESSING) dvc push

.PHONY: dvc-clean
dvc-clean:
	@echo "=== Cleaning untracked outputs ==="
	conda run -n $(ENV_PREPROCESSING) dvc gc -w

# =====================
# Tests
# =====================
.PHONY: test test-unit test-int test-all

# Default 'test' command (runs only unit tests for speed)
test: test-unit

test-unit: ## Run all unit tests via the shell script
	@echo "🧪 Running Unit Test Suite..."
	@bash tests/utils/run_all_unit_tests.sh

test-int: ## Run all integration tests via the shell script
	@echo "🪡 Running Integration Test Suite..."
	@bash tests/utils/run_all_integration_tests.sh

test-all: test-unit test-int ## Run both suites sequentially
	@echo "✅ All test suites completed."

# =====================
# Cleanup
# =====================
.PHONY: clean
clean:
	@echo "=== Removing all conda environments ==="
	$(foreach env,$(PACKAGES),conda env remove -n $(env) -y;)
	@echo "Cleanup complete."

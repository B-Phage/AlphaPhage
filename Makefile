# =====================
# Variables
# =====================

# Root folder for conda environments
CONDA_ROOT = conda_envs

# List of packages and their environments
# TODO: add the rest of the packages here
PACKAGES = utils orchestration preprocessing training models #feature_engineering training inference

# Mapping environment names
ENV_PREPROCESSING = preprocessing
ENV_FE = feature_engineering
ENV_TRAINING = training
ENV_INFERENCE = inference
ENV_DEV = root_dev

# Define the config file path
DVC_CONFIG_LOCAL = .dvc/config.local

# TODO: update this list as the packages are implemented
ENV_LIST = $(ENV_DEV) $(ENV_PREPROCESSING) $(ENV_TRAINING) #$(ENV_FE)  $(ENV_INFERENCE)

# =====================
# Help
# =====================
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make setup-all       		- Setup all environments, install packages, and setup hooks"
	@echo "  make setup-<pkg>     		- Setup a single package environment and install it"
	@echo "  make check-dvc-config     	- Check the dvc OAuth"
	@echo "  make hooks           		- Install pre-commit hooks in the current environment"
	@echo "  make secrets-cli     		- Installing detect-secrets CLI tool"
	@echo "  make secrets-baseline      - Scanning repository and generating .secrets.baseline file..."
	@echo "  make hooks-update    		- Update pre-commit hooks and reinstall"
	@echo "  make dvc-<target>    		- Run DVC commands (pull, repro, push, dag, clean)"
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

# =====================
# Main Setup
# =====================
.PHONY: setup-all
setup-all: $(foreach pkg,$(PACKAGES),setup-$(pkg)) hooks check-dvc-config dvc-pull
	@echo "All environments, packages, hooks, and DVC data are ready."

# =====================
# Package-specific setups
# =====================
.PHONY: setup-utils
setup-utils:
	$(call ensure_env,$(ENV_DEV))
	@echo "=== Installing utils package ==="
	conda run -n $(ENV_DEV) pip install -e packages/utils

.PHONY: setup-orchestration
setup-orchestration:
	$(call ensure_env,$(ENV_DEV))
	@echo "=== Installing orchestration package ==="
	conda run -n $(ENV_DEV) pip install -e packages/orchestration

.PHONY: setup-preprocessing
setup-preprocessing:
	$(call ensure_env,$(ENV_PREPROCESSING))
	@echo "=== Installing preprocessing package ==="
	conda run -n $(ENV_PREPROCESSING) pip install -e packages/preprocessing

.PHONY: setup-feature_engineering
setup-feature_engineering:
# 	$(call ensure_env,$(ENV_FE))
	@echo "=== Installing feature_engineering package ==="
# 	conda run -n $(ENV_FE) pip install -e packages/feature_engineering

.PHONY: setup-training
setup-training:
	$(call ensure_env,$(ENV_TRAINING))
	@echo "=== Installing training package ==="
	conda run -n $(ENV_TRAINING) pip install -e packages/training

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
	@echo "=== Installing pre-commit hooks in all environments ==="
	$(foreach env,$(ENV_LIST),conda run -n $(env) pre-commit install;)
	@echo "Pre-commit hooks installed successfully."

.PHONY: hooks-update
hooks-update:
	@echo "=== Updating pre-commit hooks in all environments ==="
	$(foreach env,$(ENV_LIST),conda run -n $(env) pre-commit autoupdate;)
	$(foreach env,$(ENV_LIST),conda run -n $(env) pre-commit install;)
	@echo "Pre-commit hooks updated successfully."

# =====================
# Detect secrets
# =====================
.PHONY: secrets-cli
install-secrets-cli:
	@echo "Installing detect-secrets CLI in $(ENV_DEV)..."
	conda run -n $(ENV_DEV) pip install detect-secrets

.PHONY: secrets-baseline
init-secrets-baseline: install-secrets-cli
	@echo "Generating .secrets.baseline..."
	conda run -n $(ENV_DEV) detect-secrets scan > .secrets.baseline
	@echo "Baseline created. Add and commit .secrets.baseline"

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
	dvc pull

.PHONY: dvc-repro
dvc-repro:
	@echo "=== Reproducing DVC pipeline ==="
	dvc repro

.PHONY: dvc-graph
dvc-graph:
	@echo "=== Showing DVC pipeline graph ==="
	dvc dag

.PHONY: dvc-push
dvc-push:
	@echo "=== Pushing outputs to remote ==="
	dvc push

.PHONY: dvc-clean
dvc-clean:
	@echo "=== Cleaning untracked outputs ==="
	dvc gc -w

# =====================
# Tests
# =====================
.PHONY: test

test: ## Run unit tests with correct path
	export PYTHONPATH=$PYTHONPATH:$(pwd)/workflows:$(pwd)/packages
	coverage run --source=${{ matrix.cov_source }} -m unittest ${{ matrix.test_files }}

# =====================
# Cleanup
# =====================
.PHONY: clean
clean:
	@echo "=== Removing all conda environments ==="
	$(foreach env,$(PACKAGES),conda env remove -n $(env) -y;)
	@echo "Cleanup complete."

# =====================
# Variables
# =====================

# Root folder for conda environments
CONDA_ROOT = conda_envs

# List of packages and their environments
PACKAGES = preprocessing feature_engineering training inference

# Mapping environment names
ENV_PREPROCESSING = preprocessing
ENV_FE = feature_engineering
ENV_TRAINING = training
ENV_INFERENCE = inference
ENV_DEV = root_dev

# =====================
# Help
# =====================
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make setup-all       - Setup all environments, install packages, and setup hooks"
	@echo "  make setup-<pkg>     - Setup a single package environment and install it"
	@echo "  make hooks           - Install pre-commit hooks in the current environment"
	@echo "  make secrets-cli     - Installing detect-secrets CLI tool"
	@echo "  make secrets-baseline           - Scanning repository and generating .secrets.baseline file..."
	@echo "  make hooks-update    - Update pre-commit hooks and reinstall"
	@echo "  make dvc-<target>    - Run DVC commands (pull, repro, push, dag, clean)"
	@echo "  make clean           - Remove all conda environments"

# =====================
# Main Setup
# =====================
.PHONY: setup-all
setup-all: $(foreach pkg,$(PACKAGES),setup-$(pkg)) hooks dvc-pull
	@echo "All environments, packages, hooks, and DVC data are ready."

# =====================
# Package-specific setups
# =====================
.PHONY: setup-preprocessing
setup-preprocessing:
	@echo "=== Setting up $(ENV_PREPROCESSING) ==="
	conda env create -f envs/preprocessing.yml -p $(CONDA_ROOT)/$(ENV_PREPROCESSING) || \
		conda env update -f envs/preprocessing.yml -p $(CONDA_ROOT)/$(ENV_PREPROCESSING)
	conda run -p $(CONDA_ROOT)/$(ENV_PREPROCESSING) pip install -e packages/preprocessing

.PHONY: setup-feature_engineering
setup-feature_engineering:
	@echo "=== Setting up $(ENV_FE) ==="
	conda env create -f envs/feature_engineering.yml -p $(CONDA_ROOT)/$(ENV_FE) || \
		conda env update -f envs/feature_engineering.yml -p $(CONDA_ROOT)/$(ENV_FE)
	conda run -p $(CONDA_ROOT)/$(ENV_FE) pip install -e packages/feature_engineering

.PHONY: setup-training
setup-training:
	@echo "=== Setting up $(ENV_TRAINING) ==="
	conda env create -f envs/training.yml -p $(CONDA_ROOT)/$(ENV_TRAINING) || \
		conda env update -f envs/training.yml -p $(CONDA_ROOT)/$(ENV_TRAINING)
	conda run -p $(CONDA_ROOT)/$(ENV_TRAINING) pip install -e packages/training

.PHONY: setup-inference
setup-inference:
	@echo "=== Setting up $(ENV_INFERENCE) ==="
	conda env create -f envs/inference.yml -p $(CONDA_ROOT)/$(ENV_INFERENCE) || \
		conda env update -f envs/inference.yml -p $(CONDA_ROOT)/$(ENV_INFERENCE)
	conda run -p $(CONDA_ROOT)/$(ENV_INFERENCE) pip install -e packages/inference

# =====================
# Pre-commit hooks
# =====================
.PHONY: hooks
hooks:
	@echo "=== Installing pre-commit hooks ==="
	# Install hooks in all environments where pre-commit is available
	$(foreach env,$(PACKAGES),conda run -p $(CONDA_ROOT)/$(env) pre-commit install;)
	@echo "Pre-commit hooks installed successfully."

.PHONY: secrets-cli
install-secrets-cli:
	@echo "Installing detect-secrets CLI tool in $(ENV_DEV) environment..."
	# Using standard Conda commands: activate environment, then run pip install
	conda run -n $(ENV_DEV) pip install detect-secrets

.PHONY: secrets-baseline
init-secrets-baseline: install-secrets-cli
	@echo "Scanning repository and generating .secrets.baseline file..."
	# Using standard Conda commands: activate environment, then run the tool
	conda run -n $(ENV_DEV) detect-secrets scan > .secrets.baseline
	@echo "Baseline created. Please run 'git add .secrets.baseline' and commit it."

.PHONY: hooks-update
hooks-update:
	@echo "=== Updating pre-commit hooks ==="
	$(foreach env,$(PACKAGES),conda run -p $(CONDA_ROOT)/$(env) pre-commit autoupdate;)
	$(foreach env,$(PACKAGES),conda run -p $(CONDA_ROOT)/$(env) pre-commit install;)
	@echo "Pre-commit hooks updated successfully."

# =====================
# DVC targets
# =====================

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
# Cleanup
# =====================
.PHONY: clean
clean:
	@echo "=== Removing all conda environments in $(CONDA_ROOT) ==="
	$(foreach env,$(PACKAGES),conda env remove -p $(CONDA_ROOT)/$(env) -y;)
	@echo "Cleanup complete."

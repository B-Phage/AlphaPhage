export ROOT_DIR := $(strip $(CURDIR))

# =====================
# Variables
# =====================
# Mapping environment names
ENV_PREPROC = preprocessing
ENV_TRAINING = training
ENV_KAPTIVE = kaptive3api


# Then define your inference path
INF_DATA_ROOT := $(abspath $(ROOT_DIR)/../databases/inference)

# Unique environments for cleanup and hooks
UNIQUE_ENVS = $(ENV_PREPROC) $(ENV_TRAINING) $(ENV_KAPTIVE)

# DVC Config
DVC_CONFIG_LOCAL = .dvc/config.local

# External Data Configs
HMM_YAML := ml_assets/config/hmm_data_v0.0.1.setup.yaml
KAPTIVE_YAML := ml_assets/config/kaptive3api_v0.0.1.setup.yaml

# =====================
# Help
# =====================
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make setup-all             - Setup all environments, packages, hooks, and data"
	@echo "  make hooks                 - Install pre-commit/push hooks in all environments"
	@echo "  make secrets-init          - Install detect-secrets and generate baseline"
	@echo "  make test                  - Run unit tests via shell script"
	@echo "  make clean                 - Remove all project conda environments"

# =====================
# Helpers
# =====================
define ensure_env
	@if conda env list | grep -q "^$1"; then \
		conda env update -f conda_envs/$1.yml -n $1; \
	else \
		conda env create -f conda_envs/$1.yml -n $1; \
	fi
endef

define ensure_env_mac_emul
	@if conda env list | grep -q "^$1"; then \
		conda env update -f conda_envs/$1.yml -n $1; \
	else \
		conda env create --subdir osx-64 -f conda_envs/$1.yml -n $1; \
		conda config --env --set subdir osx-64; \
	fi
endef

# =====================
# Main Setup
# =====================
.PHONY: setup-all
setup-all: setup-preproc-pkgs setup-training-pkgs setup-kaptive-pkgs hooks setup-external-data check-dvc-config dvc-pull

.PHONY: setup-preproc-pkgs
setup-preproc-pkgs:
	$(call ensure_env,$(ENV_PREPROC))
	@for pkg in utils orchestration variants representation hmm external_db_provisioning storage_io; do \
		conda run -n $(ENV_PREPROC) pip install -e packages/$$pkg; \
	done

.PHONY: setup-training-pkgs
setup-training-pkgs:
	$(call ensure_env,$(ENV_TRAINING))
	@for pkg in training dataloaders models; do \
		conda run -n $(ENV_TRAINING) pip install -e packages/$$pkg; \
	done

.PHONY: setup-kaptive-pkgs
setup-kaptive-pkgs:
	$(call ensure_env_mac_emul,$(ENV_KAPTIVE))
	CONDA_SUBDIR=osx-64 conda run -n $(ENV_KAPTIVE) pip install -e packages/kaptive3api

# =====================
# Hooks & Secrets
# =====================
.PHONY: hooks
hooks:
	@$(foreach env,$(UNIQUE_ENVS), \
		echo "Installing hooks in $(env)..."; \
		conda run -n $(env) pre-commit install; \
		conda run -n $(env) pre-commit install --hook-type pre-push; \
	)

.PHONY: secrets-init
secrets-init:
	conda run -n $(ENV_PREPROC) pip install detect-secrets
	conda run -n $(ENV_PREPROC) detect-secrets scan > .secrets.baseline

# =====================
# Data & DVC
# =====================
.PHONY: setup-external-data
setup-external-data:
	PYTHONPATH=$(shell pwd)/packages conda run -n $(ENV_PREPROC) python -m external_db_provisioning.hmm_data_setup --config $(HMM_YAML)
	PYTHONPATH=$(shell pwd)/packages conda run -n $(ENV_PREPROC) python -m external_db_provisioning.kaptive_data_setup --config $(KAPTIVE_YAML)

.PHONY: check-dvc-config
check-dvc-config:
	@if [ ! -f "$(DVC_CONFIG_LOCAL)" ] || ! grep -q "gdrive_client_id" "$(DVC_CONFIG_LOCAL)"; then \
		echo "ERROR: Missing $(DVC_CONFIG_LOCAL) or OAuth credentials."; \
		exit 1; \
	fi

.PHONY: dvc-pull
dvc-pull:
	conda run -n $(ENV_PREPROC) dvc pull

# =====================
# Tests & Cleanup
# =====================
.PHONY: test test-int test-all
test:
	@bash tests/run_all_unit_tests.sh

test-int:
	@bash tests/run_all_integration_tests.sh

test-all: test test-int

.PHONY: clean
clean:
	@$(foreach env,$(UNIQUE_ENVS), \
		echo "Removing $(env)..."; \
		conda env remove -n $(env) -y; \
	)

# =====================
# INFERENCE SNAPSHOT
# =====================

# Default values if not provided via command line
MODEL ?= default_run
EPOCH ?= 50
NAME  ?= snapshot_$(shell date +%Y%m%d_%H%M)

# e.g. usage:
# make snapshot MODEL=default_run EPOCH=61
.PHONY: snapshot
snapshot:
	@echo "🚀 Initiating snapshot for Model: $(MODEL), Epoch: $(EPOCH)"
	PYTHONPATH=$(shell pwd)/packages conda run -n $(ENV_TRAINING) \
	python packages/utils/scripts/snapshot_inference.py \
		--model $(MODEL) \
		--epoch $(EPOCH)

TRAIN_CONFIG = ml_assets/config/preprocessing_phage_v1.0.0.yaml
INF_CONFIG = ml_assets/config/inference_phage.yaml
BUFFER = 1.15

# Rule to generate the config
generate_inf_config:
	python src/create_inference_config.py \
		--training_config $(TRAIN_CONFIG) \
		--output $(INF_CONFIG) \
		--buffer $(BUFFER)


# =====================
# Inference Run
# =====================

.PHONY: run-inference
run-inference:
	@# Quick sanity check: Does the external data lake exist?
	@if [ ! -d "$(INF_DATA_ROOT)" ]; then \
		echo "ERROR: Inference directory not found at $(INF_DATA_ROOT)"; \
		exit 1; \
	fi
	@echo ">>> Starting Inference Pipeline..."
	@$(MAKE) -C inference all

.PHONY: clean-inference
	@$(MAKE) -C inference clean

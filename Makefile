# =====================
# Variables
# =====================
# Mapping environment names
ENV_PREPROC = preprocessing
ENV_TRAINING = training
ENV_KAPTIVE = kaptive3api

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
	@bash tests/utils/run_all_unit_tests.sh

test-int:
	@bash tests/utils/run_all_integration_tests.sh

test-all: test test-int

.PHONY: clean
clean:
	@$(foreach env,$(UNIQUE_ENVS), \
		echo "Removing $(env)..."; \
		conda env remove -n $(env) -y; \
	)

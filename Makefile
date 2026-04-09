export ROOT_DIR := $(strip $(CURDIR))

# =====================
# OS Detection
# =====================
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Platform defaults
ifeq ($(UNAME_S),Linux)
    CONDA_PLATFORM = linux-64
    CONDA_PLATFORM_EMUL =
    CONDA_EXE := $(shell command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)
else ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        CONDA_PLATFORM = osx-arm64
        CONDA_PLATFORM_EMUL = osx-64
    else
        CONDA_PLATFORM = osx-64
        CONDA_PLATFORM_EMUL =
    endif
    CONDA_EXE = conda
else
    CONDA_PLATFORM = linux-64
    CONDA_PLATFORM_EMUL =
    CONDA_EXE = conda
endif

# =====================
# Variables
# =====================
ENV_PREPROC = preprocessing
ENV_TRAINING = training
ENV_KAPTIVE = kaptive3api

INF_DATA_ROOT := $(abspath $(ROOT_DIR)/../databases/inference)

UNIQUE_ENVS = $(ENV_PREPROC) $(ENV_TRAINING) $(ENV_KAPTIVE)

DVC_CONFIG_LOCAL = .dvc/config.local

HMM_YAML := ml_assets/config/hmm_data_v1.0.0.setup.yaml
KAPTIVE_YAML := ml_assets/config/kaptive3api_v1.0.0.setup.yaml

# =====================
# Help
# =====================
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make setup-all"
	@echo "  make hooks"
	@echo "  make test"
	@echo "  make clean"

# =====================
# Helpers
# =====================

# Choose platform (handle kaptive emulation only on mac)
define get_platform
$(if $(and $(filter $(ENV_KAPTIVE),$1),$(CONDA_PLATFORM_EMUL)),$(CONDA_PLATFORM_EMUL),$(CONDA_PLATFORM))
endef

# Run inside env
define conda_run
	CONDA_SUBDIR=$(call get_platform,$1) \
	conda run -n $1 $2
endef

# Create/update env
define ensure_env
	@echo "Ensuring $1 environment ($(call get_platform,$1))..."
	@if conda env list | awk '{print $$1}' | grep -qx "$1"; then \
		echo "$1 exists → updating"; \
		$(CONDA_EXE) env update -f conda_envs/$1.yml -n $1; \
	else \
		echo "$1 missing → creating"; \
		$(CONDA_EXE) env create -f conda_envs/$1.yml -n $1; \
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
		$(call conda_run,$(ENV_PREPROC),pip install -e packages/$$pkg); \
	done

.PHONY: setup-training-pkgs
setup-training-pkgs:
	$(call ensure_env,$(ENV_TRAINING))
	@for pkg in training dataloaders models; do \
		$(call conda_run,$(ENV_TRAINING),pip install -e packages/$$pkg); \
	done

.PHONY: setup-kaptive-pkgs
setup-kaptive-pkgs:
	$(call ensure_env,$(ENV_KAPTIVE))
	$(call conda_run,$(ENV_KAPTIVE),pip install -e packages/kaptive3api)

# =====================
# Hooks & Secrets
# =====================
.PHONY: hooks
hooks:
	@$(foreach env,$(UNIQUE_ENVS), \
		echo "Installing hooks in $(env)..."; \
		$(call conda_run,$(env),pre-commit install); \
		$(call conda_run,$(env),pre-commit install --hook-type pre-push); \
	)

.PHONY: secrets-init
secrets-init:
	$(call conda_run,$(ENV_PREPROC),pip install detect-secrets)
	$(call conda_run,$(ENV_PREPROC),detect-secrets scan > .secrets.baseline)

# =====================
# Data & DVC
# =====================
.PHONY: setup-external-data
setup-external-data:
	PYTHONPATH=$(shell pwd)/packages $(call conda_run,$(ENV_PREPROC),python -m external_db_provisioning.hmm_data_setup --config $(HMM_YAML))
	PYTHONPATH=$(shell pwd)/packages $(call conda_run,$(ENV_PREPROC),python -m external_db_provisioning.kaptive_data_setup --config $(KAPTIVE_YAML))

.PHONY: check-dvc-config
check-dvc-config:
	@if [ ! -f "$(DVC_CONFIG_LOCAL)" ] || ! grep -q "gdrive_client_id" "$(DVC_CONFIG_LOCAL)"; then \
		echo "ERROR: Missing $(DVC_CONFIG_LOCAL) or OAuth credentials."; \
		exit 1; \
	fi

.PHONY: dvc-pull
dvc-pull:
	$(call conda_run,$(ENV_PREPROC),dvc pull)

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
# Snapshot
# =====================
MODEL ?= TransformerX_v1
EPOCH ?= 50

.PHONY: snapshot
snapshot:
	@echo "🚀 Snapshot: $(MODEL) @ $(EPOCH)"
	PYTHONPATH=$(shell pwd)/packages $(call conda_run,$(ENV_TRAINING),python packages/utils/scripts/snapshot_inference.py --model $(MODEL) --epoch $(EPOCH))

# =====================
# Inference
# =====================
.PHONY: run-inference
run-inference:
	@if [ ! -d "$(INF_DATA_ROOT)" ]; then \
		echo "ERROR: Inference directory not found at $(INF_DATA_ROOT)"; \
		exit 1; \
	fi
	@$(MAKE) -C inference all

# =====================
# Debug
# =====================
.PHONY: debug-platform
debug-platform:
	@echo "OS: $(UNAME_S)"
	@echo "ARCH: $(UNAME_M)"
	@echo "CONDA_PLATFORM: $(CONDA_PLATFORM)"
	@echo "CONDA_PLATFORM_EMUL: $(CONDA_PLATFORM_EMUL)"
	@echo "CONDA_EXE: $(CONDA_EXE)"

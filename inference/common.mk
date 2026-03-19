ROOT_DIR ?= $(shell pwd | sed 's/\/inference//')
export ROOT_DIR := $(strip $(ROOT_DIR))

# Define the data root based on the clean ROOT_DIR
INF_DATA_ROOT := $(abspath $(ROOT_DIR)/../databases/inference)
INF_WORKSPACE := $(INF_DATA_ROOT)/workspace

# 1. Base Paths
SNAPSHOT_NAME := default_run_ep61_20260319
SNAPSHOT_DIR  := $(INF_DATA_ROOT)/$(SNAPSHOT_NAME)

# 2. Infrastructure Paths (Global Constants)
QUERY_PHAGE   := $(INF_DATA_ROOT)/sequences/phage
QUERY_HOST    := $(INF_DATA_ROOT)/sequences/host
INF_OUT       := $(INF_DATA_ROOT)/output
RUNTIME_CFG   := $(SNAPSHOT_DIR)/runtime_config
CODE_HASH_TXT := $(INF_DATA_ROOT)/code_hash.txt

# Conda Runners (Absolute path to packages)
CUR_PKGS        := PYTHONPATH=$(ROOT_DIR)/packages
CONDA_RUN_PREP  := $(CUR_PKGS) conda run -n preprocessing
CONDA_RUN_KAP   := $(CUR_PKGS) conda run -n kaptive3api
CONDA_RUN_TRAIN := $(CUR_PKGS) conda run -n training

# Source Code
PKG_DIR := $(ROOT_DIR)/packages
WRKFLOW_DIR := $(ROOT_DIR)/workflows

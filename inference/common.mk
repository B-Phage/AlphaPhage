ROOT_DIR ?= $(shell pwd | sed 's/\/inference//')
export ROOT_DIR := $(strip $(ROOT_DIR))

# Define the data root based on the clean ROOT_DIR
INF_DATA_ROOT := $(abspath $(ROOT_DIR)/../databases/inference)
INF_WORKSPACE := $(INF_DATA_ROOT)/workspace

# 1. Base Paths
SNAPSHOT_NAME := default_run_ep61_20260319
SNAPSHOT_DIR  := $(INF_DATA_ROOT)/$(SNAPSHOT_NAME)
# Point specifically to the versioned YAML inside the snapshot
VOCAB_FILE    := $(SNAPSHOT_DIR)/vocab/vocab_v0.0.1.yaml

# Aggregation params
PHAGE_TABLE_NAME := phage_variants_v1
BAC_TABLE_NAME   := bacteria_variants_v1
FEATURE_DB       := $(INF_DATA_ROOT)/data/processed/duckdb/features.duckdb

SEROTYPE_TABLE_NAME := bacteria_serotypes_v1
PHAGE_REF_INFO 		:= $(SNAPSHOT_DIR)/reference/vector_info/phage_vec_1_info.tsv
HOST_REF_INFO		:= $(SNAPSHOT_DIR)/reference/vector_info/host_vec_1_info.tsv


# Output
INT_MATRIX_TSV := $(INF_WORKSPACE)/interactions/inference_matrix.tsv
MODEL_FEATURE_DATA_NAME := inference_set_v1

# 2. Infrastructure Paths (Global Constants)
QUERY_PHAGE   := $(INF_DATA_ROOT)/data/sequence_data/phage
QUERY_HOST    := $(INF_DATA_ROOT)/data/sequence_data/bacteria
INF_OUT       := $(INF_DATA_ROOT)/output
RUNTIME_CFG   := $(SNAPSHOT_DIR)/runtime_config
CODE_HASH_TXT := $(INF_DATA_ROOT)/code_hash.txt

# Conda Runners (Absolute path to packages)
CUR_PKGS        := PYTHONPATH=$(ROOT_DIR)/packages
CONDA_RUN_PREP  := $(CUR_PKGS) conda run -n preprocessing
CONDA_RUN_KAPTIVE   := $(CUR_PKGS) conda run -n kaptive3api
CONDA_RUN_TRAIN := $(CUR_PKGS) conda run -n training

# Source Code
PKG_DIR := $(ROOT_DIR)/packages
WRKFLOW_DIR := $(ROOT_DIR)/workflows

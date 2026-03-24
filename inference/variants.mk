include common.mk

# This file represents the 7-stage variant preprocessing pipeline
.PHONY: all_stages run_prodigal_gv run_hmm_scan run_hmm_summary run_create_fastas run_msa_generation run_inference_mapping run_aggregate

# Ensure DOMAIN is passed from the parent. If not, this might fail.
# We use $(filter ...) to check the domain string
ifeq ($(DOMAIN),phage)
TABLE_NAME := $(PHAGE_TABLE_NAME)
VOCAB_KEY := vocab_clusters_single
else
TABLE_NAME := $(BAC_TABLE_NAME)
VOCAB_KEY := vocab_clusters_single
DOMAIN := bac
endif

# Helper for conditional force flag
FORCE_FLAG := $(if $(FORCE),--force,)

# Define a local workspace for Stage 3 temp files (HMM IDs)
# We nest this in the external HASH_DIR to keep the repo clean
STAGE3_WORK_DIR := $(HASH_DIR)/stage_3_temp

# We extract the cluster name from the config to match the folder structure
# This shell command pulls the value from the YAML
CLUSTER_NAME := $(shell grep 'msa_cluster_name:' $(CFG) | awk '{print $$2}')

# Dynamically extract the output dir from the config
MAPPED_DIR := $(abspath $(shell grep 'mapped_output_dir:' $(CFG) | awk '{print $$2}'))

all_stages: run_prodigal_gv run_hmm_scan run_hmm_summary run_create_fastas run_msa_generation run_inference_mapping run_aggregate

# 1. Prodigal: Extract proteins from query
run_prodigal_gv:
	@echo "🧬 [Stage 1] Running Prodigal..."
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/run_prodigal.py \
		--config $(CFG) \
		--code_hash $(CODE_HASH_TXT) \
		--hash_store $(HASH_DIR) \
		$(FORCE_FLAG)

# 2. HMM Scan: Map proteins to HMM families
run_hmm_scan:
	@echo "🔍 [Stage 2] HMM Scanning Query Proteins..."
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/run_hmm_scan_pipeline.py \
		--config $(CFG) \
		--code_hash $(CODE_HASH_TXT) \
		--hash_store $(HASH_DIR) \
		$(FORCE_FLAG)

# 2.5 Build Summary: Aggregate scan results into DuckDB
run_hmm_summary:
	@echo "📊 [Stage 2.5] Building HMM Summary Table..."
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/hmm_parquet_to_summary_tbl.py \
		--config $(CFG)

# 3. HMM Fasta Export: Prepare files for MSA
run_create_fastas:
	@echo "📂 [Stage 3] Exporting HMM Fastas..."
	@mkdir -p $(STAGE3_WORK_DIR)
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/create_hmm_fastas.py \
		--config $(CFG) \
		--hash_store $(HASH_DIR) \
		--hmm_id_dir $(STAGE3_WORK_DIR)/hmm_ids \
		$(FORCE_FLAG)

# 4. MSA Generation: Align sequence hits to HMM profiles
run_msa_generation:
	@echo "🧬 [Stage 4] Generating MSAs (CSA format)..."
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/create_msas.py \
		--config $(CFG) \
		--code_hash $(CODE_HASH_TXT) \
		--hash_store $(HASH_DIR) \
		$(if $(FORCE),--force,)

# 5. Inference Mapping: Assign query alignments to training variants
run_inference_mapping:
	@echo "🧬 [Stage 5] Mapping Queries to Cluster Variants..."
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/inference_msa_cluster_mapping.py \
		--config $(CFG) \
		--code_hash $(CODE_HASH_TXT) \
		--hash_store $(HASH_DIR) \
		--vocab $(VOCAB_FILE) \
		$(if $(FORCE),--force,)

# 6. Aggregation: Combine individual HMM parquets into a global Feature Table
run_aggregate:
	@echo "🗄️ [Stage 6] Aggregating Variants for $(DOMAIN)..."
	$(CONDA_RUN_PREP) python -m variants.long_table_builder \
		-i $(MAPPED_DIR) \
		-d $(FEATURE_DB) \
		-t $(TABLE_NAME) \
		-s $(VOCAB_KEY) \
		-c $(VOCAB_FILE) \
		-mp $(FEATURE_DB)_$(DOMAIN)_manifest.json

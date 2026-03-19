include common.mk

# This file represents the 7-stage variant preprocessing pipeline
.PHONY: all_stages run_prodigal_gv run_hmm_inference run_create_fastas

# Helper for conditional force flag
FORCE_FLAG := $(if $(FORCE),--force,)

# Define a local workspace for Stage 3 temp files (HMM IDs)
# We nest this in the external HASH_DIR to keep the repo clean
STAGE3_WORK_DIR := $(HASH_DIR)/stage_3_temp

all_stages: run_prodigal_gv run_hmm_inference run_create_fastas

# 1. Prodigal: Extract proteins from query
run_prodigal_gv:
	@echo "🧬 [Stage 1] Running Prodigal..."
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/run_prodigal.py \
		--config $(CFG) \
		--code_hash $(CODE_HASH_TXT) \
		--hash_store $(HASH_DIR) \
		$(FORCE_FLAG)

# 2. HMM Scan: Map proteins to HMM families
run_hmm_inference:
	@echo "🔍 [Stage 2] HMM Scanning Query Proteins..."
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/run_hmm_scan_pipeline.py \
		--config $(CFG) \
		--code_hash $(CODE_HASH_TXT) \
		--hash_store $(HASH_DIR) \
		$(FORCE_FLAG)

# 3. HMM Fasta Export: Prepare files for MSA
run_create_fastas:
	@echo "📂 [Stage 3] Exporting HMM Fastas..."
	@mkdir -p $(STAGE3_WORK_DIR)
	$(CONDA_RUN_PREP) python $(WRKFLOW_DIR)/create_hmm_fastas.py \
		--config $(CFG) \
		--hash_store $(HASH_DIR) \
		--hmm_id_dir $(STAGE3_WORK_DIR)/hmm_ids \
		$(FORCE_FLAG)

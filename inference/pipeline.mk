# --- Safety Gate: Drift Detection ---

.PHONY: check-drift
check-drift:
	@echo "🔍 Checking for code-snapshot drift..."
	@python -c ' \
	import json, subprocess, sys; \
	meta = json.load(open("$(SNAPSHOT)/metadata.json")); \
	current_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(); \
	if meta["git_sha"] != current_sha: \
		print(f"\n⚠️  WARNING: DRIFT DETECTED"); \
		print(f"   Snapshot Code: {meta[\"git_sha\"][:8]} (Branch: {meta.get(\"git_branch\", \"unknown\")})"); \
		print(f"   Current Code : {current_sha[:8]}"); \
		print("   Changes in packages/ may cause incompatible preprocessing results.\n"); \
	else: \
		print("   ✅ Code and Snapshot are in sync.")'

# --- Phage Specifics ---
PHAGE_CFG   := $(RUNTIME_CFG)/phage_preprocessing.yaml
PHAGE_HASH  := $(HASH_DIR)/phage

# --- Host Specifics ---
HOST_CFG    := $(RUNTIME_CFG)/bac_preprocessing.yaml
HOST_HASH   := $(HASH_DIR)/bacteria

.PHONY: run_phage run_host all

all: run_phage run_host

# We define a "Function" to run the stages
define run_pipeline
	@echo "--- Starting Pipeline for $(1) ---"
	$(MAKE) -f stages.mk CFG=$(2) H_STORE=$(3) all_stages
endef

run_phage:
	$(call run_pipeline,PHAGE,$(PHAGE_CFG),$(PHAGE_HASH))

run_host:
	$(call run_pipeline,HOST,$(HOST_CFG),$(HOST_HASH))


# # --- Rules ---

# .PHONY: run_hmm_inference
# run_hmm_inference:
# 	@echo "Starting HMM Scan for Inference Snapshot..."
# 	python3 $(SRC_DIR)/hmm_workflow.py \
# 		--config $(INF_CONFIG) \
# 		--code_hash $(CODE_HASH) \
# 		--hash_store $(HASH_STORE)

include common.mk

# --- Hash Stores ---
HASH_DIR_PHAGE := $(INF_DATA_ROOT)/hashes/phage
HASH_DIR_HOST  := $(INF_DATA_ROOT)/hashes/bacteria


.PHONY: all phage host post-process

# The Master Workflow
all: phage host #post-process

# --- PHAGE BRANCH ---
phage:
	@echo ">>> Running 7-Stage Phage Pipeline"
	$(MAKE) -f variants.mk \
		CFG=$(RUNTIME_CFG)/phage_preprocessing.yaml \
		HASH_DIR=$(INF_WORKSPACE)/hashes/phage \
		all_stages

# --- HOST BRANCH ---
host:
	@echo ">>> Running 7-Stage Host Pipeline + 2 Serotype Stages"
	$(MAKE) -f variants.mk \
		CFG=$(RUNTIME_CFG)/bac_preprocessing.yaml \
		HASH_DIR=$(INF_WORKSPACE)/hashes/bacteria \
		all_stages
# 	$(MAKE) -f host_only.mk \
# 		CFG=$(RUNTIME_CFG)/serotype.yaml \
# 		HASH_DIR=$(INF_WORKSPACE)/hashes/bacteria \
# 		serotype_stages

# # --- FINAL INFERENCE ---
# post-process:
# 	@echo ">>> Running Interaction Extraction & Model Inference"
# 	$(MAKE) -f post_process.mk \
# 		PHAGE_LAKE=$(INF_WORKSPACE)/phage/lake \
# 		HOST_LAKE=$(INF_WORKSPACE)/host/lake \
# 		OUT_DIR=$(INF_DATA_ROOT)/output \
# 		final_stages

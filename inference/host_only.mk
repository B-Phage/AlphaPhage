# host_only.mk
include common.mk

# Domain settings for aggregation
TABLE_NAME := bacteria_serotypes_v1
# We pull the JSON dir from the config
JSON_DIR := $(abspath $(shell grep 'kaptive_dir:' $(CFG) | awk '{print $$2}'))
.PHONY: serotype_stages run_kaptive_ecoli aggregate_serotypes

# The master target now runs both
serotype_stages: run_kaptive_ecoli aggregate_serotypes

run_kaptive_ecoli:
	@echo "🏷️  [Serotype] Running Kaptive for E. coli..."
	$(CONDA_RUN_KAPTIVE) python $(WRKFLOW_DIR)/run_kaptive3api.py \
		--config $(CFG) \
		--code_hash $(CODE_HASH_TXT) \
		--hash_store $(HASH_DIR) \
		$(if $(FORCE),--force,)

# Stage 5.2: Aggregate into DuckDB
aggregate_serotypes:
	@echo "🗄️  [Serotype] Aggregating tokens into DuckDB..."
	$(CONDA_RUN_PREP) python -m representation.typing_aggregator \
		-i $(JSON_DIR) \
		-d $(FEATURE_DB) \
		-t $(TABLE_NAME) \
		-v $(VOCAB_FILE) \
		-m $(FEATURE_DB)_serotypes_manifest.json

# post_process.mk
include common.mk

.PHONY: final_stages build_inference_matrix aggregate_features

# Add your next stages here as you build them (e.g., aggregate_features)
final_stages: build_inference_matrix aggregate_features

build_inference_matrix:
	@echo "🔀 [Inference] Building All-vs-All Interaction Matrix..."
	@# Ensure the directory exists (e.g., .../workspace/interactions/)
	@mkdir -p $(dir $(INT_MATRIX_TSV))
	$(CONDA_RUN_PREP) python -m representation.matrix_builder \
		-p $(QUERY_PHAGE) \
		-b $(QUERY_HOST) \
		-o $(INT_MATRIX_TSV)

aggregate_features: build_inference_matrix
	@echo "📊 [Inference] Building HDF5 Feature Set..."
	$(CONDA_RUN_TRAIN) python -m representation.build_model_data \
		-i $(INT_MATRIX_TSV) \
		-d $(FEATURE_DB) \
		-o $(INF_OUT) \
		-f $(MODEL_FEATURE_DATA_NAME) \
		-v $(VOCAB_FILE) \
		--phage_table $(PHAGE_TABLE_NAME) \
		--host_table $(BAC_TABLE_NAME) \
		--serotype_table $(SEROTYPE_TABLE_NAME) \
		--phage_ref $(PHAGE_REF_INFO) \
		--host_ref $(HOST_REF_INFO) \
		--inference

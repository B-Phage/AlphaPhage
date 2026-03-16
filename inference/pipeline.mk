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

# --- Inference Stages ---

# 1. Prodigal: Extract proteins from query
$(INF_WORKDIR)/prodigal/manifest.json: $(QUERY_PHAGE) | check-drift
	@echo "🧬 Running Prodigal..."
	$(CONDA_RUN) python ../workflows/run_prodigal.py \
		--config $(RUNTIME_CFG)/phage_preprocessing.yaml \
		--input $(QUERY_PHAGE) \
		--output $(INF_WORKDIR)/prodigal/

# 2. HMM Scan
$(INF_WORKDIR)/hmm_results.parquet: $(INF_WORKDIR)/prodigal/manifest.json
	@echo "🔍 Scanning Query Proteins..."
	$(CONDA_RUN) python ../workflows/run_hmm_scan_pipeline.py \
		--config $(RUNTIME_CFG)/phage_preprocessing.yaml \
		--input $(INF_WORKDIR)/prodigal/ \
		--output $@

# 3. Vectorize: The bridge between Query and Snapshot
$(INF_WORKDIR)/query_vector.pt: $(INF_WORKDIR)/hmm_results.parquet $(SNAPSHOT)/reference/features.db
	@echo "📊 Vectorizing query..."
	$(CONDA_RUN) python ../workflows/inference_vectorize.py \
		--db $(SNAPSHOT)/reference/features.db \
		--vocab_info $(SNAPSHOT)/reference/vector_info/ \
		--input $< \
		--output $@

# 4. Predict
$(INF_OUT)/prediction.json: $(INF_WORKDIR)/query_vector.pt $(SNAPSHOT)/model/model_weights.ckpt
	@echo "🧠 Predicting Interaction..."
	$(CONDA_RUN) python ../workflows/predict.py \
		--model $(SNAPSHOT)/model/model_weights.ckpt \
		--input $< \
		--output $@

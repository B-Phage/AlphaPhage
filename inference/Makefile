# --- Configuration ---
SNAPSHOT ?= ../../database/inference/v1_snapshot
QUERY_PHAGE ?= data/queries/test.fasta

# --- Internal Paths ---
INF_WORKDIR = workdir
INF_OUT = output
RUNTIME_CFG = $(SNAPSHOT)/runtime_config
CONDA_RUN = PYTHONPATH=$(shell pwd)/../packages conda run -n training

# --- Standard Targets ---
include pipeline.mk

.PHONY: predict
predict: $(INF_OUT)/prediction.json
	@echo "✅ Inference complete. Results at $(INF_OUT)/prediction.json"

.PHONY: setup
setup:
	mkdir -p $(INF_WORKDIR) $(INF_OUT) data/queries

.PHONY: clean
clean:
	rm -rf $(INF_WORKDIR)/* $(INF_OUT)/*

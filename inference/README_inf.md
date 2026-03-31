# Inference Pipeline 🧬
This directory contains the orchestration logic for running inference on phage-host interaction models. It uses a modular Makefile structure to handle data preprocessing, serotyping, and model execution.

## 🏗 System Architecture
The pipeline follows a Snapshot-Dependency model. It does not run against "latest" code/data blindly; instead, it relies on a specific model snapshot generated from the training pipeline.

## 1. The Snapshot Mechanism
Before running inference, a model must be "signed off" and snapshotted from the root directory:
```
Bash
make snapshot MODEL=my_experiment EPOCH=50
```
This utility bundles the weights, YAML configurations, and vocabulary files into `databases/inference/<SNAPSHOT_NAME>`.

## 2. Configuration (`common.mk`)
To point the pipeline to a specific model, update the SNAPSHOT_NAME in `inference/common.mk`:
```
Makefile
SNAPSHOT_NAME := snapshot_20260330_1500  # Update this to your latest signed-off run
```
This ensures that the `VOCAB_FILE`, `MODEL_CONFIG`, and `CHECKPOINT_PATH` all stay synchronized.

## 🚀 Execution Flow
The pipeline is triggered via `make all` and executes three distinct branches:

### Phase 1: Phage Branch (`variants.mk`)
- Input: Raw phage sequences from $(INF_DATA_ROOT)/data/sequence_data/phage.
- Process: A 7-stage pipeline (defined in `variants.mk`) that performs cleaning, feature extraction, and hashing.

### Phase 2: Host Branch (`variants.mk` & h`ost_only.mk`)
- Preprocessing: Runs the standard 7-stage genomic pipeline.
- Serotyping: Runs additional stages via `host_only.mk` to determine bacterial serotypes (using the kaptive3api conda environment).

### Phase 3: Post-Processing (post_process.mk)
- Interaction Matrix: Constructs the final feature matrix by joining Phage and Host vectors.
- Inference: Loads the `.ckpt` weights and runs the model to produce interaction probabilities.

## 🛠 Prerequisites
### Environment Management
The pipeline automatically handles environment switching via `conda run`. Ensure the following environments exist:

- `preprocessing`: For genomic data manipulation.

- `kaptive3api`: For bacterial serotyping.

- `training`: For running the PyTorch Lightning/Inference scripts.

## Directory Structure
The pipeline expects a specific layout in your `INF_DATA_ROOT`:

```
Plaintext
databases/inference/
├── snapshot_xyz/          # Snapshotted model assets
│   ├── model/             # .ckpt files
│   ├── runtime_config/    # YAML configs for preprocessing
│   └── vocab/             # Tokenizer/Vocab mappings
├── data/
│   └── sequence_data/     # Input .fasta or .fastq files
└── workspace/             # Intermediate hashes and outputs
```

## 📝 Developer Notes
- **Idempotency**: The pipeline uses hash-based tracking. If inputs haven't changed, stages will be skipped. To force a re-run, clear the `$(INF_DATA_ROOT)/hashes` directory.

- **Output Sync**: `MAKEFLAGS += --output-sync=none` is set in the local Makefile to allow real-time log streaming during long genomic stages.

- **Adding Stages**: If adding a new preprocessing step, update `variants.mk` and ensure the all_stages target includes your new logic.

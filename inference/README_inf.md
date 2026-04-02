# Inference Pipeline 🧬
This directory contains the orchestration logic for running phage-host interaction predictions. It is designed as a decoupled inference space that operates independently of the DVC training DAG.

## 🏗 System Architecture
The pipeline follows a Snapshot-Dependency model. To maintain production stability, the inference engine does not track the "latest" research code blindly. Instead, it relies on a specific Model Snapshot promoted from the training environment.

## 1. Model Promotion (The "Snapshot")
Before running inference, a specific model must be "signed off" and deployed to the inference space. This is triggered from the Root Makefile of the repository:
```
Bash
make snapshot MODEL=my_experiment EPOCH=50
```
**What this does:**

Invokes the `snapshot_inference.py` utility via the `training` conda environment.

Extracts the `.ckpt` weights, the specific `runtime_config` YAMLs used during that training run, and the corresponding `vocab` mappings.

Bundles these assets into `databases/inference/snapshot_YYYYMMDD_HHMM`.

## 2. Configuration (`common.mk`)
Once a snapshot is generated, you must "point" the inference pipeline to it. Update the `SNAPSHOT_NAME` in `inference/common.mk`:
```
Makefile
SNAPSHOT_NAME := snapshot_20260330_1500  # Update this to your latest signed-off run
```
This ensures that the `VOCAB_FILE`, `MODEL_CONFIG`, and `CHECKPOINT_PATH` all stay synchronized.

## 🚀 Execution Flow
The pipeline is triggered from the `inference/` directory and executes three distinct phases across multiple Conda environments:
```
Bash
# From the inference/ directory:
make all
```

### Phase 1: Phage Branch (`variants.mk`)
- Input: Raw phage sequences from `$(INF_DATA_ROOT)/data/sequence_data/phage`.
- Process: A 7-stage pipeline (defined in `variants.mk`) that performs cleaning, feature extraction, and hashing.

### Phase 2: Host Branch (`variants.mk` & h`ost_only.mk`)
- Preprocessing: Runs the standard 7-stage genomic pipeline.
- Serotyping: Runs additional stages via `host_only.mk` to determine bacterial serotypes (using the kaptive3api conda environment).

### Phase 3: Post-Processing (post_process.mk)
- Interaction Matrix: Constructs the final feature matrix by joining Phage and Host vectors.
- Model Execution: Loads the `.ckpt` weights and runs the model to produce interaction probabilities.

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
├── snapshot_xyz/          # Promoted Model Assets (created by 'make snapshot')
│   ├── model/             # .ckpt files
│   ├── runtime_config/    # YAML configs preserved from training
│   └── vocab/             # Tokenizer/Vocab mappings
├── data/
│   └── sequence_data/     # Input .fasta or .fastq files
└── workspace/             # Intermediate hashes and outputs
```

## 📝 Developer Notes
- **Idempotency**: The pipeline uses hash-based tracking. If inputs haven't changed, stages will be skipped. To force a re-run, clear the `$(INF_DATA_ROOT)/hashes` directory.

- **Output Sync**: `MAKEFLAGS += --output-sync=none` is set in the local Makefile to allow real-time log streaming during long genomic stages.

- **Adding Stages**: If adding a new preprocessing step, update `variants.mk` and ensure the all_stages target includes your new logic.

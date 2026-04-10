🥇![Coverage main](https://github.com/B-Phage/AlphaPhage/raw/badges2/coverage-main.svg) | 🥇![Coverage dev](https://github.com/B-Phage/AlphaPhage/raw/badges2/coverage-dev.svg) |

# AlphaPhage: Production Machine Learning Model 🧪

This repository houses the **production-grade** implementation of our most robust predictive model. It is engineered for **reliability, repeatability, and maintainability** to ensure successful, stable deployment.

---

## 🎯 Production Quality Aims

We maintain a strict definition of "production-grade" code, which is enforced via Continuous Integration (CI) and code reviews. Contributions are held to the following standards:

* **Multi-Tiered Quality Assurance:**
  - **Unit Test Level:** 100% coverage across all packages (`utils`, `models`, and the various preprocessing packages).
  - **Integration Level:** Validation of critical "real-world" data flows and API interactions with minimal mocking.
  - **Enforcement:** Automated via CI gatekeepers (GitHub Actions) and the `make test-all` suite.
* **Reproducibility & Traceability:** All data pipelines, model training runs, and results must be completely **repeatable** and **versioned** (enforced via **DVC** for data/model versioning and **DVC Live** for metrics).
* **Maintainability & Clarity:** Code must be clean, well-optimized, and thoroughly documented. This includes comprehensive **docstrings**, explanatory **code comments**, and clear documentation for the entire MLOps workflow.
* **Seamless Deployment:** The application must be container-ready and follow a clear, standardized process for building, testing, and pushing to production environments.

**Note on Coverage:** Running `make test` executes our unified test runner which aggregates coverage data across all three Conda environments (Preprocessing, Training, and Kaptive) into a single, comprehensive HTML report found in `htmlcov/`.

---

## 🛠 Getting Started

Please see the **[CONTRIBUTING.md](CONTRIBUTING.md)** file for detailed setup instructions, the full code style guide, commit message conventions, and how to run tests locally.

---

## ⚡ Quick Start

### Set up DVC Data Authentication (REQUIRED) 🔑
This step grants access to the large data files tracked by DVC.
Follow the instructions in **[CONTRIBUTING.md](CONTRIBUTING.md)**, Section 2.3,  to generate your Client ID and Secret.

### Checkout Code, Install Dependencies, DVC Pull & Run Pipeline

```bash
git clone <repo>
cd <repo>
# 1. Setup environments, packages, and pull DVC data and runtime configs
make setup-all

# 2. Run the unit tests (automatically handles env switching)
make test

# 3. Execute the pipeline (requires an active environment)
conda activate preprocessing
dvc repro # relies on `dvc pull` which was run as part of `make setup-all`
```

Alternatively, run infernence on a downloaded or locally produced snapshot.

```bash
# Inference Pipeline
make run-inference
```

---

## 🗂 Repository Structure
```
repo/
├── packages/       # Python modules (preprocessing, training, etc.)
├── ml_assets/      # 📦 Training Space (DVC Tracked)
├── inference/      # 🚀 Inference Space (Snapshotted / "Make" driven)
├── conda_envs/     # Conda environment files
├── workflows/      # Training and infernece workflows
├── tests/          # Unit and Integration test
├── docs/           # Processes, long-form reference, and cross-cutting guides
├── docker/         # Dockerfiles for Docker image based deployment
├── dvc.yaml        # Training DVC pipeline file
└── Makefile        # Setup & pipeline commands
```

## 📦 Packages and Modules

All production code is located within the `packages/` directory. Each package contains a dedicated `README.md` for a targeted overview of its role, contents, and dependencies. They are installed through the `Makefile`, see instrucitons in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

| Package | Description | Details |
| :--- | :--- | :--- |
| `packages/orchestration` | **Workflow Management:** Custom mini-scheduler responsible for dynamic per-file processing, parallel execution, and state checking, called by the coarse DVC stages. | [Read Package README](packages/orchestration/README_orch.md) |
| `packages/utils` | **Core Utilities:** Common code for logging, file handling, and shared data structures. | [Read Package README](packages/utils/README_utils.md) |
| `packages/hmm` | **Preprocessing:** Handles the annotation of protein sets using HMM matching. | [Read Package README](packages/hmm/README_hmm.md) |
| `packages/kaptive3api | Kaptive Integration: Specialized API wrapper (runs via osx-64 emulation).	| [Read Package README](packages/kaptive3api/README_kaptive3api.md) |
| *(... and so on for the rest of our packages)* | | |

## 🚀 Model Lifecycle: From Training to Deployment
AlphaPhage separates the Controlled Training Environment from the Portable Inference Pipeline to ensure that only "signed-off" models reach production.

## 1. Training & Reproducibility (DVC-Controlled)
The training DAG is strictly managed by DVC (Data Version Control). This ensures every model is tied to a specific git commit, dataset version, and hyperparameter set.

- **Immutable Pipeline**: Managed via dvc.yaml. Use dvc repro to execute.

- **Experiment Control**: All "switches" are managed in the root params.yaml. The pipeline automatically handles config staging and path injection.

- **Architecture**: See [docs/PIPELINE.md](docs/PIPELINE.md) for the full DAG visualization.

## 2. Model Promotion (The Snapshot Bridge)
Once a training run is complete and verified, you must "promote" the model to the inference space. The Snapshot Utility extracts the weights, configs, and vocabulary from the DVC-tracked `ml_assets/` and freezes them into the untracked inference directory `../databases/inference/<SNAPSHOT_NAME>`.
```
Bash
# Deploys a specific model version to the inference space
make snapshot MODEL=TransformerX_v1 EPOCH=42
```

📦 Note: Official snapshots are zipped and hosted in the AlphaPhage Model Zoo (Google Drive). Links are provided in the GitHub Release notes for each version tag.

## 3. Inference Pipeline (Make-Driven)
The inference pipeline is designed to be lightweight and decoupled from the DVC training overhead. It uses standard make commands to run predictions on new data using the snapshotted model.

- Workflow: Operates independently of the DVC DAG for faster, more flexible deployment.

- Execution:
```
bash
make run-inference  # Executes the snapshotted model
```

- Detailed Guides: * [docs/RELEASE_MANAGEMENT.md](docs/RELEASE_MANAGEMENT.md): Full specification for Model Sign-off, Deployment Profiles, and the Snapshot process.

  - [inference/README_inf.md](inference/README_inf.md): Technical guide for local CLI usage and variable overrides.

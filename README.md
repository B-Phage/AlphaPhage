🥇![Coverage main](https://github.com/B-Phage/AlphaPhage/raw/badges2/coverage-main.svg) | 🥇![Coverage dev](https://github.com/B-Phage/AlphaPhage/raw/badges2/coverage-dev.svg) |
🥈![Coverage feature_msa_variant_clustering](https://github.com/B-Phage/AlphaPhage/raw/badges2/coverage-feature_msa_variant_clustering.svg) 🥈![Coverage feature_ktypes](https://github.com/B-Phage/AlphaPhage/raw/badges2/coverage-feature_ktypes.svg)

# AlphaPhage: Production Machine Learning Model 🧪

This repository houses the **production-grade** implementation of our most robust predictive model. It is engineered for **reliability, repeatability, and maintainability** to ensure successful, stable deployment.

---

## 🎯 Production Quality Aims

We maintain a strict definition of "production-grade" code, which is enforced via Continuous Integration (CI) and code reviews. Contributions are held to the following standards:

* **100% Code Coverage:** All core logic must be covered by unit and integration tests (enforced via **Unittests** and **Code Coverage** tools).
* **Reproducibility & Traceability:** All data pipelines, model training runs, and results must be completely **repeatable** and **versioned** (enforced via **DVC** for data/model versioning and **DVC Live** for metrics).
* **Maintainability & Clarity:** Code must be clean, well-optimized, and thoroughly documented. This includes comprehensive **docstrings**, explanatory **code comments**, and clear documentation for the entire MLOps workflow.
* **Seamless Deployment:** The application must be container-ready and follow a clear, standardized process for building, testing, and pushing to production environments.

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
git clone <repo>        # checkout code
cd <repo>
make setup-all          # Sets up environments, packages, hooks, pulls DVC data
make test               # Run the unit tests
dvc repro               # Run the preprocessing and training pipeline
```
---

## 🗂 Repository Structure
```
repo/
├── packages/       # Python modules (preprocessing, training, etc.)
├── ml_assets/      # Data, models, experiments tracked with DVC
├── conda_envs/     # Conda environment files
├── workflows/
├── tests/          # Unit and Integration test
├── docs/           # Processes, long-form reference, and cross-cutting guides
├── dvc.yaml
└── Makefile # Setup & pipeline commands
```

## 📦 Packages and Modules

All production code is located within the `packages/` directory. Each package contains a dedicated `README.md` for a targeted overview of its role, contents, and dependencies. They are installed through the `Makefile`, see instrucitons in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

| Package | Description | Details |
| :--- | :--- | :--- |
| `packages/orchestration` | **Workflow Management:** Custom mini-scheduler responsible for dynamic per-file processing, parallel execution, and state checking, called by the coarse DVC stages. | [Read Package README](packages/orchestration/README_orch.md) |
| `packages/utils` | **Core Utilities:** Common code for logging, file handling, and shared data structures. | [Read Package README](packages/utils/README_utils.md) |
| `packages/hmm` | **Preprocessing:** Handles the annotation of protein sets using HMM matching. | [Read Package README](packages/preprocessing/README_hmm.md) |
| *(... and so on for the rest of our packages)* | | |

## 🚀 Execution & Reproducibility

AlphaPhage uses **DVC (Data Version Control)** to manage a complex, parallelised DAG.

* **View Pipeline Architecture**: A full visual representation of our processing streams can be found in [docs/PIPELINE.md](docs/PIPELINE.md).
* **Run the Pipeline**: Use `dvc repro` to execute all out-of-date stages.
* **Change Experiments**: To swap model architectures, preprocessing logic, or dataset versions, follow the [Experiment Control Guide](docs/PIPELINE.md#️-experiment-control-config-staging).

> 💡 **Tip**: All experimental "switches" are managed in the root `params.yaml`. The pipeline automatically handles the staging and path injection for you.

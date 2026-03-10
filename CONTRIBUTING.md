# Contributing to AlphaPhage 🧪

Thank you for contributing to **AlphaPhage**!
This guide provides **all the instructions, standards, and best practices** to ensure code is reliable, reproducible, and production-ready.

---

## 1. Overview

AlphaPhage is a **production-grade ML repository**, focused on reproducibility, maintainability, and clean deployment.
All contributions must adhere to the following principles:

- ✅ **Reliability & Reproducibility:** Code and pipelines must be fully repeatable.
- ✅ **Maintainability & Clarity:** Code should be clean, well-documented, and easy to understand.
- ✅ **Production Readiness:** Any changes should not break CI/CD or pipeline execution.

---

## 2. Initial Setup

Follow these steps to get started with the repo:

### 2.1. Configure SSH for Git Access (Optional but Recommended)

### 2.2. Clone the repository

```bash
git clone <repo-url>
cd <repo-name>
```

### 2.3. Configure DVC Data Authentication (Google Drive)

This step grants access to the project's large data files tracked by DVC.

Before you can authenticate, you need to create the application credentials (Client ID and Secret) used for DVC's OAuth flow.

1. **Access Google Cloud Console:** Navigate to the [alphaphage-dvc](https://console.cloud.google.com/apis/api/drive.googleapis.com/metrics?project=alphaphage-dvc)
2. **Go to __Credentials__:** In the left navigation menu, go to **APIs & Services** $\rightarrow$ **Credentials**.
3. **Create OAuth Client ID:**
   * Click **+ CREATE CREDENTIALS** and select **OAuth client ID**.
   * Set **Application type** to **Desktop app**.
   * Give it a memorable name.
   * Click **CREATE**.
4. **Copy Credentials:** A modal window will appear displaying your new **Client ID** and **Client Secret**.
5. **Update Config:** Copy both values and paste them into your `config.local` file, replacing the placeholders. Since _.dvc/config.local_ is not committed, each user needs one.
  * Copy the template:
```
cp .dvc/config.local.template .dvc/config.local
```
  * Edit _.dvc/config.local_:
```
['remote "gdrive_remote"']
    gdrive_client_id = <YOUR_CLIENT_ID>
    gdrive_client_secret = <YOUR_CLIENT_SECRET>
```
Use the client ID/secret for the OAuth project you’ve enabled:
[alphaphage-dvc](https://console.cloud.google.com/apis/api/drive.googleapis.com/metrics?project=alphaphage-dvc)


### 2.4. Authenticate with Google OAuth
`dvc pull`


Browser will open asking you to log in.

Log in with your Google account.

Google may ask for permissions; if the OAuth project is already verified/enabled, it should succeed.

After successful login, DVC caches the token locally (~~/.config/dvc/cred.json) → no need to repeat.

Each user must authenticate once per machine.

### 2.5. Setup environments, install packages, and pre-commit hooks

```bash
make setup-all
```

This command will:

- Create all Conda environments in _/conda_envs/_

- Install each Python package in editable mode (`pip install -e packages/<module>`)

- Install pre-commit hooks

- Pull DVC-tracked data and models

### 2.6. Activate an environment
```bash
conda activate conda_envs/<environment_name>
```
Replace `<environment_name>` with preprocessing, feature_engineering, training, or inference.

## 3. Repository Structure
```
repo/
├── packages/              # Source code directory containing all Python packages.
│   ├── utils/             # Core utility functions (e.g., logging, common data manipulation).
│   ├── orchestration/     # Custom mini-scheduler logic for sample discovery, parallel execution, and partial reruns.
│   ├── preprocessing/     # Scripts for data cleaning and initial transformation.
│   ├── feature_engineering/ # Logic for creating and selecting model features.
│   ├── training/          # Model training, validation, and serialization logic.
│   └── inference/         # Code for loading models and making predictions.
├── ml_assets/             # Storage for Data Version Control (DVC) tracked files.
│   ├── data/              # Raw and processed datasets (DVC remote storage pointer).
│   ├── models/            # Trained models and checkpoints.
│   ├── vocab/             # Transformer data vocabulary configuration versions
│   └── experiments/       # DVC experiment results, metrics, and parameters.
├── workflows/             # Scripts for DVC. The scripts run a workflow using the orchestration package (snakemake replacements)
├── conda_envs/            # Environment definition files (.yml) for reproducibility.
├── tests/                 # Unit, integration, and functional test suite for all packages.
├── dvc.yaml               # Defines the coarse, stable pipeline stages and serves as the deployment engine for the workflow.
├── Makefile               # Task runner for common operations: install, lint, test, format, and DVC pipeline execution.
├── pyproject.toml         # Defines project metadata, and configurations for tools like Black and isort.
├── .coveragerc            # Configuration file for Python code coverage (coverage.py).
├── .secrets.baseline      # Security baseline file for tracking potential exposed secrets.
└── .pre-commit-config.yaml # Configuration file for Git hooks (e.g., Black, isort, MyPy) to ensure code quality before commit.
```

Place new code in _packages/<module_name>/_.

Track datasets and models with DVC under _ml_assets/_.

Add environment dependencies to _conda_envs/<module_name>.yml_.

## 4. Coding Standards

All code must adhere to production-grade coding standards:

- Formatting: Black, isort

- Linting: ruff / flake8

- Type hints: MyPy

- Docstrings: Google style

- Tests: Unit and integration tests required

Run checks locally:
```bash
conda activate <environment>
pre-commit run --all-files
pytest --cov=packages/<module>

```

## 5. Git Workflow

* **Branch naming:**

  * feature/<short-description>
  * bugfix/<short-description>
  * hotfix/<short-description>

*Note*: The **_badges_** branch is a stand alone branch for storing coverage **_.svg_** images generated in the CI workflow. It is not intended to ever be merged into **_main_**

* **Commit messages:**

  * Use imperative mood
  * Include a short title (<50 chars) and optional detailed body

**Example:**

```pgsql
Add preprocessing function for missing value imputation

- Handles NaNs in numeric and categorical columns
- Added unit tests for edge cases

```
* **Pull requests (PRs):**

  * Run all tests before opening a PR
  * Ensure pre-commit hooks pass
  * CI/CD must pass before merging

## 6. Testing

Run tests per module:

```bash
conda activate <environment>
pytest --cov=packages/<module>
```
Check code coverage:
```bash
pytest --cov=packages/<module> --cov-report term-missing
```

Integration tests should cover full pipeline stages with small test datasets.

## 7. DVC and Pipeline

* Pull data/models: `make dvc-pull`

* Reproduce pipeline: `make dvc-repro`

* Push outputs to remote: `make dvc-push`

* View pipeline graph:
  ```
    dvc dag --dot > pipeline.dot
    dot -Tpng pipeline.dot -o pipeline.png
  ```
  Then open `pipeline.png`

* Clean untracked outputs: `make dvc-clean`

DVCLive metrics:
All metric logging must go to ml_assets/experiments/ as configured in each pipeline stage.

### 7.1. Critical Rules for DVC & Git Coexistence
To prevent data loss and broken pointers, strictly follow these rules:

* *Never manually ignore* `ml_assets/data/` or `ml_assets/models/` in the root `.gitignore`. DVC creates its own internal `.gitignore` files to handle the raw data.
  * If you ignore the entire directory in Git, you will accidentally delete the `.dvc` metadata files from source control, making it impossible for others to pull the data.

* *The `.dvc` files are code.* Treat them like `.py` files. They must be committed to Git so the repo knows which version of the data matches which version of the code.

* *Post-Merge Workflow:* After every `git pull` or `git merge`, you must run `make dvc-pull`. Git updates the pointers; DVC updates the actual files.

| Asset Type | Stored In | Tracked By |
| :--- | :--- | :--- |
| **Source Code** (`.py`, `.yml`) | GitHub | **Git** |
| **Metadata Pointers** (`.dvc`) | GitHub | **Git** |
| **Large Datasets / Weights** | Google Drive | **DVC** |

## 8. Adding New Packages / Modules

1. Create _packages/<module>/_ folder.

2. Add _pyproject.toml_ and a ___init__.py_ folder structure.

3. Create a Conda environment file: _envs/<module>.yml_ (include Python version, 3.10+ if required).

4. Add the package to _Makefile_ targets:

```makefile

setup-<module>:
    conda env create -f envs/<module>.yml -p $(CONDA_ROOT)/<module> || \
        conda env update -f envs/<module>.yml -p $(CONDA_ROOT)/<module>
    conda run -p $(CONDA_ROOT)/<module> pip install -e packages/<module>
```
5. Add pre-commit and DVC integration as needed.

## 9. CI/CD Contributions

* Ensure all CI checks pass locally before pushing.

* The CI workflow includes:

  * Unit and integration tests

  * Code coverage

  * Pre-commit hook validation

  * DVC reproducibility check

## 10. References & Resources

* [DVC Documentation](https://dvc.org/doc)
* [DVCLive Documentation](https://doc.dvc.org/dvclive)
* [Black](https://black.readthedocs.io/en/stable/)
* [isort](https://pycqa.github.io/isort/)
* [MyPy](https://mypy-lang.org/)
* [Pre-commit](https://pre-commit.com/)

## ⚡ Quick Commands Summary

```bash
make setup-all          # Setup environments, packages, pre-commit hooks, DVC pull
make dvc-repro          # Run the full DVC pipeline
make hooks-update       # Update pre-commit hooks
make clean              # Remove all conda environments
```

Thank you for contributing to AlphaPhage! Your work helps maintain a robust, reliable, and production-ready ML system.

## Setup the individual packages in the mono-repo

```
pip install -e ./packages/utils
pip install -e ./packages/preprocessing
```

## Versioning & Releases

We follow (Semantic Versioning)[https://semver.org/].
All changes must go through reviewed Pull Requests.

For details on how releases are prepared and tagged, see:
docs/releasing.md

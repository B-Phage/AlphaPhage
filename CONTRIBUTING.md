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

### Clone the repository

```bash
git clone <repo-url>
cd <repo-name>
```

### Setup environments, install packages, and pre-commit hooks

```bash
make setup-all
```

This command will:

- Create all Conda environments in _/conda_envs/_

- Install each Python package in editable mode (`pip install -e packages/<module>`)

- Install pre-commit hooks

- Pull DVC-tracked data and models

### Activate an environment
```bash
conda activate conda_envs/<environment_name>
```
Replace `<environment_name>` with preprocessing, feature_engineering, training, or inference.

## 3. Repository Structure
```
repo/
├── packages/              # Python modules (preprocessing, feature_engineering, training, inference)
├── ml_assets/             # DVC-tracked data, models, experiments
├── envs/                  # Conda environment definitions per module
├── configs/               # YAML config files for pipeline stages
├── dvc.yaml               # DVC pipeline
├── Makefile               # Setup and pipeline commands
├── pyproject.toml         # Root project metadata (optional)
└── .pre-commit-config.yaml
```
Place new code in _packages/<module_name>/_.

Track datasets and models with DVC under _ml_assets/_.

Add environment dependencies to _envs/<module>.yml_.

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

* View pipeline graph: `make dvc-dag`

* Clean untracked outputs: `make dvc-clean`

DVCLive metrics:
All metric logging must go to ml_assets/experiments/ as configured in each pipeline stage.

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

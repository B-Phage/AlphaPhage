# 🤝 Contributing to AlphaPhage

Welcome! AlphaPhage is a production-grade repository built on principles of reliability, reproducibility, and rigorous testing. This document serves as the guide for all development work, including setup, coding standards, and submission guidelines.

---

## 🚀 1. Setup and Environment

Before contributing, please ensure your development environment is correctly configured.

* **Prerequisites:** List required software (e.g., Python version, Docker).
* **Installation:**
    1.  Clone the repository: `git clone [repository URL]`
    2.  Set up the virtual environment (e.g., `conda create -n alphaphage python=3.x`).
    3.  Install dependencies: `pip install -r requirements.txt`
* **Pre-Commit Hooks:** We use pre-commit hooks to automatically check code for formatting and style issues before a commit is created.
    * Install hooks: `pre-commit install`
    * *Note: Your commit will fail if checks do not pass.*

---

## 🛠 2. Coding Standards and Style

We enforce a strict style guide to ensure code is clean, maintainable, and readable by the entire team.

* **Style Guide:** We adhere to the [e.g., PEP 8 for Python]. We use the **Black** formatter and **Flake8** linter (configured in `pyproject.toml` or `.flake8`).
* **Type Hinting:** All new functions and public methods must include **Python type hints**.
* **Documentation and Docstrings:**
    * Every function, class, and method must have a **comprehensive docstring** (e.g., following the NumPy or Google style).
    * Code comments must be used judiciously to explain **why** code is written a certain way, not **what** the code does.
* **Optimization:** Review performance impact, especially in model training or data processing loops. [Optional: Mention profiling tools to use].

---

## 🧪 3. Testing and Quality Assurance

Contributions will only be merged if they meet our stringent testing requirements.

* **Test Coverage Goal:** **100% code coverage** is mandatory for all new logic.
* **Unit Tests:** New features and bug fixes require corresponding unit tests using the **Unittest** framework. Tests must cover both success and expected failure paths.
* **Running Tests:** Run the full test suite before submitting a pull request: `pytest` or `python -m unittest discover`.
* **Data and Model Versioning (DVC):**
    * Any changes to raw data, processed features, or trained models must be tracked via **DVC**.
    * Ensure all experiments use **DVCLive** to log metrics for traceability.

---

## 📦 4. Submission Guidelines

Follow these steps for submitting your work for review.

1.  **Create a New Branch:** Base your work off the `main` branch: `git checkout -b feature/my-new-feature` or `git checkout -b fix/issue-123`.
2.  **Commit Messages:** We use [e.g., Conventional Commits] for clear, automated versioning.
    * Format: `<type>(<scope>): <description>` (e.g., `feat(model): add new loss function`).
3.  **Create a Pull Request (PR):**
    * PRs must target the `main` branch.
    * The PR description should clearly summarize the changes and link to any relevant issues.
    * Ensure all CI checks (tests, coverage, linting) pass before requesting a review.

---

### 🔑 Initial Setup: Enforcing Quality Standards

1.  **Create/Activate the Root Development Environment:**
    ```bash
    # Assuming you have a top-level environment.yml for dev tools
    conda env create -f environment.yml
    conda activate root-dev
    ```
    *Note: The root-dev environment must contain the 'pre-commit' package.*

2.  **Install the Git Hooks:**
    ```bash
    pre-commit install
    ```
    This command registers the hooks defined in `.pre-commit-config.yaml` with your local Git repository.

**That's it!** From now on, every time you run `git commit`, the pre-commit framework will automatically run all checks against the files you've staged.
#!/bin/bash

# 1. Configuration
ENV_TRAINING="training"
ENV_PREPROC="preprocessing"
ENV_KAPTIVE="kaptive3api"

# 2. Setup paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export COVERAGE_FILE="$PROJECT_ROOT/.coverage"
export LOCAL_PYTHONPATH="$PROJECT_ROOT/packages:$PROJECT_ROOT/workflows"
export KMP_DUPLICATE_LIB_OK=TRUE

echo "📍 Running from: $PROJECT_ROOT"

# 3. Clean slate
echo "🧼 Cleaning old data..."
rm -f .coverage .coverage.*

# Helper function to run tests via pytest with coverage
run_tests() {
    local env_name="$1"
    local test_targets="$2"

    echo "--------------------------------------------------------"
    echo "📋 Environment: $env_name"
    echo "--------------------------------------------------------"

    # -v: Detailed list of test names and results
    # -p no:cov: Disables the internal pytest-cov plugin to stop those "No data" warnings
    # --tb=short: Keeps the error logs concise if something fails
    conda run -n "$env_name" bash -c "
        export PYTHONPATH=$LOCAL_PYTHONPATH; \
        export KMP_DUPLICATE_LIB_OK=TRUE; \
        COVERAGE_FILE=.coverage.$env_name \
        coverage run -m pytest -v -p no:cov -o 'addopts=' $test_targets"
}

# --- PART 1: TRAINING ---
echo "🏋️ Running Training Tests (Pytest)..."
run_tests "$ENV_TRAINING" "tests/unit/training"

# --- PART 2: PREPROCESSING ---
echo "🧹 Running Preprocessing Unit Tests (Pytest)..."
run_tests "$ENV_PREPROC" "tests/unit/preprocessing"

# --- PART 3: KAPTIVE3API ---
echo "🧬 Running Kaptive3API Unit Tests (Pytest)..."
# This runs in the emulated Intel environment if on Mac
run_tests "$ENV_KAPTIVE" "tests/unit/kaptive3api"

# --- PART 4: COMBINE & REPORT ---
echo -e "\n📊 --- FINAL COMBINED UNIT TEST COVERAGE ---"
# Remap paths and merge
# We use the preprocessing env as the 'reporter' because it has the standard tools
conda run -n "$ENV_PREPROC" coverage combine .coverage.*
conda run -n "$ENV_PREPROC" coverage report -m
conda run -n "$ENV_PREPROC" coverage html -d htmlcov
echo "📈 HTML report generated: $(pwd)/htmlcov/index.html"

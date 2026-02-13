#!/bin/bash

# 1. Configuration
ENV_TRAINING="training"
ENV_PREPROC="preprocessing"

# 2. Setup paths
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
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
    conda run -n "$env_name" bash -c "export PYTHONPATH=$LOCAL_PYTHONPATH; \
        export KMP_DUPLICATE_LIB_OK=TRUE; \
        COVERAGE_FILE=.coverage.$env_name coverage run -m pytest -v -p no:cov -o 'addopts=' $test_targets"
}

# --- PART 1: TRAINING ---
echo "🏋️ Running Training Tests (Pytest)..."
TRAIN_FILES="tests/unit/test_transformer_archspecs.py tests/unit/test_transformer_classifier.py tests/unit/test_transformer_module.py tests/unit/test_dataloaders.py"
run_tests "$ENV_TRAINING" "$TRAIN_FILES"

# --- PART 2: CATCH-ALL ---
echo "🧹 Running Remaining Unit Tests (Pytest)..."
EXCLUDE=(tests/unit/test_transformer_archspecs.py tests/unit/test_transformer_classifier.py tests/unit/test_transformer_module.py tests/unit/test_dataloaders.py)
REMAINING_TESTS=$(find tests/unit -name 'test_*.py' $(for f in "${EXCLUDE[@]}"; do echo "! -path $f"; done) | tr '\n' ' ')

if [ -n "$REMAINING_TESTS" ]; then
    run_tests "$ENV_PREPROC" "$REMAINING_TESTS"
fi

# --- PART 3: COMBINE & REPORT ---
echo -e "\n📊 --- FINAL COMBINED UNIT TEST COVERAGE ---"
# Remap paths and merge
conda run -n "$ENV_PREPROC" coverage combine .coverage.*
conda run -n "$ENV_PREPROC" coverage report -m
conda run -n "$ENV_PREPROC" coverage html -d htmlcov
echo "📈 HTML report generated: $(pwd)/htmlcov/index.html"

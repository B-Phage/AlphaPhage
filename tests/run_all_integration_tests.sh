#!/bin/bash

# 1. Configuration
ENV_TRAINING="training"
ENV_PREPROC="preprocessing"
ENV_KAPTIVE="kaptive3api"

# 2. Setup paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure all coverage data is written to unique files for combining later
export COVERAGE_FILE="$PROJECT_ROOT/.coverage"
export LOCAL_PYTHONPATH="$PROJECT_ROOT/packages:$PROJECT_ROOT/workflows"
export KMP_DUPLICATE_LIB_OK=TRUE

echo "📍 Running Integration Tests from: $PROJECT_ROOT"

# 3. Clean slate
echo "🧼 Cleaning old coverage data..."
rm -f .coverage .coverage.*

# Helper function to run tests via pytest with coverage
run_integration_tests() {
    local env_name="$1"
    local test_targets="$2"

    echo "--------------------------------------------------------"
    echo "📋 Environment: $env_name"
    echo "--------------------------------------------------------"

    # We use -v for verbose output and -p no:cov to avoid config conflicts
    # Note: we use 'coverage run --source=packages,workflows' to match the CI requirement
    conda run -n "$env_name" bash -c "
        export PYTHONPATH=$LOCAL_PYTHONPATH; \
        export KMP_DUPLICATE_LIB_OK=TRUE; \
        COVERAGE_FILE=.coverage.int.$env_name \
        coverage run --source=packages,workflows -m pytest -v -p no:cov -o 'addopts=' $test_targets"
}

# --- PART 1: TRAINING INTEGRATION ---
echo "🏋️ Running Training Integration Tests..."
# Target the specific training integration file
run_integration_tests "$ENV_TRAINING" "tests/integration/training"

# --- PART 2: PREPROCESSING INTEGRATION ---
echo "🧹 Running Preprocessing Integration Tests..."
run_integration_tests "$ENV_PREPROC" "tests/integration/preprocessing"

# --- PART 3: KAPTIVE3API INTEGRATION ---
echo "🧬 Running Kaptive3API Integration Tests..."
# This runs integration tests for the emulated Intel package
run_integration_tests "$ENV_KAPTIVE" "tests/integration/kaptive3api"

# --- PART 4: COMBINE & REPORT ---
echo -e "\n📊 --- FINAL COMBINED INTEGRATION COVERAGE ---"
# Combine the integration-specific coverage files
# This will use your .coveragerc [paths] to fix any conda site-packages vs local packages mismatches
# This uses the preprocessing env as the reporter for consistency
conda run -n "$ENV_PREPROC" coverage combine .coverage.int.*
conda run -n "$ENV_PREPROC" coverage report -m
conda run -n "$ENV_PREPROC" coverage html -d htmlcov_int

echo "📈 Integration HTML report generated: $(pwd)/htmlcov_int/index.html"

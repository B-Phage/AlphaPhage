#!/bin/bash
# Generate a covergate badge
# This script supports the CI documentation.

LABEL_PREFIX=$1
BRANCH_NAME=$2
COVERAGE_VALUE=$3
OUTPUT_FILE=$4

# Define the directory of the current script
SCRIPT_DIR="$(dirname "$0")"

# Execute the Python script using its path relative to the current script's directory
python3 "$SCRIPT_DIR/generate_badge.py" "$LABEL_PREFIX" "$BRANCH_NAME" "$COVERAGE_VALUE" --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Badge $OUTPUT_FILE created successfully."
else
    echo "ERROR: Badge generation failed."
    exit 1
fi

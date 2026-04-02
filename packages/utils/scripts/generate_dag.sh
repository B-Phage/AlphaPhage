#!/bin/bash
# Generate the DVC pipeline DAG image.
# This script supports automatic documentation of the docs/PIPELINE.m

# 1. Check if dot (Graphviz) is installed
if ! command -v dot &> /dev/null; then
    echo "⚠️  Warning: Graphviz (dot) not found. Skipping DAG image update."
    exit 0
fi

# 2. Generate the DAG
mkdir -p docs/images
dvc dag --dot > docs/pipeline.dot
dot -Tpng docs/pipeline.dot -o docs/images/pipeline_dag.png
rm docs/pipeline.dot

# 3. Stage the new image
git add docs/images/pipeline_dag.png
echo "✅ DVC DAG visualization updated."

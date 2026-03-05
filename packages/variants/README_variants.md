# variants

## Builds protein variants from Multiple Sequence Alignments (MSAs).

A variant is a cluster of related protein sequences (annotated by the same HMM), grouped using a configurable distance metric and clustering strategy. The resulting tables are used downstream for model training.

## What This Package Does

1. Given an MSA:

2. Standardises amino acid characters using the configured vocabulary

3. Optionally selects alignment columns (e.g. diversity-based)

4. Computes pairwise distances (identity or blosum62)

5. Performs hierarchical clustering

6. Writes:

    * `.mva` → variant feature table

    * `.mcd` → cluster metadata table

## Structure
```
variants/
  build.py          # High-level entrypoint + CLI
  clustering.py     # Hierarchical clustering logic
  distance.py       # Distance matrix calculations
  preprocessing_methods.py  # HMM-specific preprocessing
```

##CLI Example

```
python -m variants.build \
  -f input.aln \
  -o output_dir \
  -c 16 \
  -m identity \
  -r diversity
```

## Scope

This package contains biological domain logic for HMM-based variant construction.

It does not handle:

* Model encoding

* Vocabulary definitions

* Training data assembly

* Orchestration or storage concerns

Those live elsewhere.

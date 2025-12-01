# Orchestrator

Use DVC only for the coarse, stable pipeline stages.
Move all “dynamic per-file logic” to a custom orchestrator.
Let DVC be the deployment engine, not the scheduler.

This gets you the best of both worlds:

1. DVC handles big input/output directories, versioned models, remote storage, and global pipeline consistency.

2. Your orchestrator handles sample discovery, per-file checks, parallelisation, partial reruns, and overrides.

## 1. Role Division: What DVC should do

DVC should handle only the parts where reproducibility matters at the dataset level, not per-file logic.

### ✔ DVC should track:

raw data directories

model parameters

environment / conda lockfile

top-level pipeline steps (e.g., “run all QC”, “run alignment”, “train model”)

### ✔ DVC should provide:

hashing → ensures reproducibility

caching → avoids recomputing heavy stages

remote storage → push/pull to S3, GDrive, etc.

pipeline graph → modular high-level stages

### ❌ DVC should NOT do:

per-file job creation

dynamic wildcard expansion

parallel job scheduling

discovery of new files

partial recomputation logic

This is where Snakemake shines, but DVC intentionally avoids it.

## 🔵 2. Role Division: What your orchestrator should do

Your orchestrator becomes a lightweight mini-scheduler, similar to Snakemake’s rule expansion, but simpler.

**It should:**

✔ scan directories for files that match a pattern

(e.g., *.fastq, sample folders, genome files)

✔ determine which items are “dirty”

Use a mechanism of your choice:

compare modification times

compute checksums

compare against a manifest

compare against a JSON “state database”

✔ schedule only the changed items

Parallel execution via:

concurrent.futures

ray

dask

or Python multiprocessing

✔ allow forced reruns

(e.g., when you release new pipeline code)

✔ produce outputs in a deterministic location

So DVC can track them as a directory without caring about “who touched what”.

✔ optionally update a manifest (inputs → outputs)

This provides reproducibility and incremental correctness.

## Workflow Diagram

          +---------------------+
          |    DVC Pipeline     |
          |   (coarse stages)   |
          +---------------------+
                   |
                   v  (calls)
          +---------------------+
          |   Orchestrator.py   |
          | - scan new files    |
          | - compute hashes    |
          | - parallel jobs     |
          | - force reruns      |
          +---------------------+
                   |
                   v  (per-item jobs)
          +---------------------+
          | process_sample.py   |
          +---------------------+
                   |
                   v
          +---------------------+
          | data/processed/*    |
          +---------------------+

## Folder Structure

ml_assets/
    data/
        workflow/
            hashes/
                dna_to_protein_hashes.json
                prot_to_hmm_hashes.json
            config/
                dna_to_protein.yaml
                prot_to_hmm.yaml

packages/
    orchestration/
        __init__.py
        orchestrator.py
        utils.py

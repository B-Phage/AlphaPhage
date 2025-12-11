#!/usr/bin/env python3
"""
Run Prodigal gene/protein prediction using a specified configuration file.
"""

import argparse
import os
import subprocess
from pathlib import Path

from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_prodigal_config

# -----------------------------
# Command-line args
# -----------------------------
parser = argparse.ArgumentParser(description="Run Prodigal on a set of genomes")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the YAML config file for this run",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Force rerun all genomes regardless of hash",
)
args = parser.parse_args()

# -----------------------------
# Load configuration
# -----------------------------
cfg = get_prodigal_config(override_path=args.config)

GENOME_DIR = Path(cfg.genome_dir)
OUTPUT_DIR = Path(cfg.output_dir)
LST_GENOMES = cfg.lst_genomes
N_JOBS = cfg.n_jobs
CODE_VERSION = cfg.code_version
HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")


# -----------------------------
# Worker function
# -----------------------------
def run_prodigal(input_path: Path, outputs: list):
    protein_file = outputs[0]
    protein_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "prodigal-gv",
        "-p",
        "meta",
        "-i",
        str(input_path),
        "-a",
        str(protein_file),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return outputs


# -----------------------------
# Output function
# -----------------------------
def prodigal_output_func(input_path: Path):
    genome_name = input_path.stem
    protein_file = OUTPUT_DIR / f"{genome_name}.faa"
    return [protein_file]


# -----------------------------
# Prepare inputs
# -----------------------------
if LST_GENOMES:
    genome_files = [GENOME_DIR / f"{g}.fasta" for g in LST_GENOMES]
else:
    genome_files = sorted(GENOME_DIR.glob("*.fasta"))

# -----------------------------
# Run orchestrator
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    orch = Orchestrator(
        workflow_name=f"prodigal_{cfg.get('name', 'default')}",
        hash_store_dir=HASH_STORE_DIR,
        logger_name="prodigal",
    )
    orch.run(
        inputs=genome_files,
        output_func=prodigal_output_func,
        worker=run_prodigal,
        code_id=CODE_VERSION,
        parallelism=N_JOBS,
        force=args.force,
        tqdm_desc=f"Prodigal processing ({cfg.get('name', 'default')})",
    )

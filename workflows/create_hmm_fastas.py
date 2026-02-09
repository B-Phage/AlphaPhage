#!/usr/bin/env python3

import argparse
from pathlib import Path

from hmm.duckdb_io import (
    extract_hmm_sequences_for_msa,
    get_distinct_hmms,
)
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Export per-HMM FASTA files from DuckDB summary table."
)
parser.add_argument("--config", required=True, help="Path to YAML config")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
gen_cfg = get_config(override_path=args.config, params_set="general")
cfg = get_config(override_path=args.config, params_set="hmm_config")
run_cfg = get_config(
    "ml_assets/config/runtime.yaml", params_set="meta"
)  # config that does not affect outputs

DB_PATH = Path(cfg.duckdb)
TABLE_NAME = cfg.summary_tbl
FASTA_ROOT = Path(cfg.hmm_fasta_dir)  # add this to config
HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")
HMM_ID_DIR = Path("ml_assets/data/workflow/hmm_ids")

FASTA_ROOT.mkdir(parents=True, exist_ok=True)
HMM_ID_DIR.mkdir(parents=True, exist_ok=True)

CODE_VERSION = "export_hmm_fastas_v1"  # bump if logic changes
N_JOBS = run_cfg.n_jobs

# ---------------------------------------------------------
# Prepare virtual input files (one per HMM)
# ---------------------------------------------------------
hmms = get_distinct_hmms(DB_PATH, TABLE_NAME)

hmm_id_files = []
for hmm in hmms:
    p = HMM_ID_DIR / f"{hmm}.hmm_id"
    p.write_text(hmm)
    hmm_id_files.append(p)


# ---------------------------------------------------------
# Worker
# ---------------------------------------------------------
def worker(input_path: Path, outputs: list):
    hmm_name = input_path.read_text().strip()
    out_fasta = outputs[0]

    extract_hmm_sequences_for_msa(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        hmm_domain_name=hmm_name,
        out_fasta=out_fasta,
    )


# ---------------------------------------------------------
# Output mapping
# ---------------------------------------------------------
def output_func(input_path: Path):
    hmm_name = input_path.read_text().strip()
    out_fasta = FASTA_ROOT / f"{hmm_name}.fasta"
    return [out_fasta]


# ---------------------------------------------------------
# Run orchestrator
# ---------------------------------------------------------
if __name__ == "__main__":
    orch = Orchestrator(
        workflow_name=f"export_{gen_cfg.get('name', 'test')}_hmm_fastas",
        hash_store_dir=HASH_STORE_DIR,
        logger_name="hmm_fasta_export",
    )

    orch.run(
        inputs=hmm_id_files,
        output_func=output_func,
        worker=worker,
        code_id=CODE_VERSION,
        parallelism=N_JOBS,
        force=args.force,
        tqdm_desc=f"Exporting {gen_cfg.get('name', 'test')} HMM FASTAs",
    )

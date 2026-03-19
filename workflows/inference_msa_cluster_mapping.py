#!/usr/bin/env python3
"""
Workflow: Inference Variant Mapping
Maps unseen aligned protein hits to existing cluster variants
using a stored gallery and distance thresholds.
"""

import argparse
import os
from functools import partial
from pathlib import Path

import pandas as pd
import utils.consol_log as log
from orchestration.orchestrator import Orchestrator
from representation.align_table import process_msa_table
from utils.config_loader import get_config, get_vocab
from utils.hash import compute_combined_code_version, create_manifest
from variants.inference_mapper import InferenceMapper

logger = log.get_logger()

DEFAULT_RUNTIME_PATH = "ml_assets/config/runtime.yaml"  # TO
HASH_STORE_DIR = Path("ml_assets/data/workflow/inference_hashes")

# -----------------------------------------------------------------------------
# Multiprocessing Workers
# -----------------------------------------------------------------------------


def mapping_worker(input_path: Path, outputs: list, gallery_dir: Path, matrix: str, buffer: float):
    """
    Worker to map a single HMM's query MSA against its training gallery.
    """
    hmm_id = input_path.stem
    output_path = outputs[0]

    # 1. Load Query Data
    # Assuming queries are stored in a format readable by pandas (e.g. .csa or .csv)
    # We need: organism_id, hmm_id, sequence_aligned
    vocab = get_vocab()  # TODO: fix to snapshot vocab
    query_df = process_msa_table(input_path, vocab=vocab)

    # 2. Load Gallery (Training Shard)
    gallery_path = gallery_dir / f"{hmm_id}.parquet"
    if not gallery_path.exists():
        logger.warning(f"No gallery found for {hmm_id}. Skipping mapping.")
        return

    gallery_df = pd.read_parquet(gallery_path)

    # 3. Execute Rigorous Mapping
    mapper = InferenceMapper(buffer_factor=buffer)
    results_df = mapper.map_to_variants(
        queries_df=query_df, gallery_df=gallery_df, matrix_type=matrix
    )

    # 4. Save results
    results_df.to_parquet(output_path)


def get_mapping_outputs(input_path: Path, output_dir: Path) -> list[Path]:
    """Expected output is a mapped results parquet for each HMM."""
    return [output_dir / f"{input_path.stem}_mapped.parquet"]


# -----------------------------------------------------------------------------
# Workflow Orchestration
# -----------------------------------------------------------------------------


def run_inference_workflow(config_path: str, code_hash, force: bool = False):
    """
    Orchestrates the Inference Mapping process.
    """
    run_cfg = get_config(DEFAULT_RUNTIME_PATH, params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    inf_cfg = get_config(override_path=config_path, params_set="inference_config")
    var_cfg = get_config(override_path=config_path, params_set="msa_var")

    # Path resolution
    query_input_dir = Path(inf_cfg.query_msa_dir)  # New unseen hits
    gallery_dir = Path(inf_cfg.gallery_shard_dir)  # Training Parquets
    output_dir = Path(inf_cfg.mapped_output_dir)  # Final assignments

    logger.info(f"Mapping Queries from: {query_input_dir}")
    logger.info(f"Using Gallery from: {gallery_dir}")

    # Versioning
    code_version = compute_combined_code_version(
        base_version=gen_cfg.code_version,
        hash_files=code_hash,
        config_items=[str(gallery_dir), var_cfg.dis_mat_method],
    )

    # Setup environment
    query_files = sorted(list(query_input_dir.glob("*.csa")))
    os.makedirs(output_dir, exist_ok=True)

    # Prepare worker
    worker_with_args = partial(
        mapping_worker,
        gallery_dir=gallery_dir,
        matrix=var_cfg.dis_mat_method,
        buffer=inf_cfg.get("buffer_factor", 1.1),
    )

    output_func_with_args = partial(get_mapping_outputs, output_dir=output_dir)

    orch = Orchestrator(
        workflow_name=f"inference_{gen_cfg.get('name', 'test')}",
        hash_store_dir=HASH_STORE_DIR,
        logger_name="inference",
    )

    orch.run(
        inputs=query_files,
        output_func=output_func_with_args,
        worker=worker_with_args,
        code_id=code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc="Mapping queries to variants",
        wipe_dirs=[output_dir] if force else [],
    )

    # Manifest of results
    mapped_results = list(output_dir.glob("*.parquet"))
    create_manifest(mapped_results, output_dir / "inference_manifest.json")
    logger.info(f"✅ Inference mapping complete. {len(mapped_results)} HMMs processed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Map unseen hits to existing variants.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--code_hash", action="append", type=Path, required=True)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    run_inference_workflow(config_path=args.config, code_hash=args.code_hash, force=args.force)


if __name__ == "__main__":
    main()

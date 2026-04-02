#!/usr/bin/env python3
"""
Workflow: Combined Inference Alignment & Mapping
Directly maps raw inference hits to variants by performing joint HMM alignment
and distance-based assignment in a single pass.
"""

import argparse
import pathlib
from functools import partial
from pathlib import Path

import pandas as pd
import utils.consol_log as log
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config, get_vocab
from utils.hash import compute_combined_code_version, create_manifest

# The unified logic we developed
from variants.run_variant_inference import run_variant_inference

logger = log.get_logger()

# --- Path Anchoring ---
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_RUNTIME_PATH = REPO_ROOT / "ml_assets" / "config" / "runtime.yaml"
DEFAULT_HASH_DIR = REPO_ROOT / "ml_assets" / "data" / "workflow" / "inference_hashes"

# -----------------------------------------------------------------------------
# Multiprocessing Worker
# -----------------------------------------------------------------------------


def inference_worker(
    input_fasta: Path,
    outputs: list,
    gallery_dir: Path,
    hmm_dir: Path,
    vocab_config,
    temp_root: Path,
    matrix: str,
    buffer: float,
    clustering_version: str,
):
    """
    Worker to process a single HMM: Fasta -> Alignment -> Mapping -> Parquet.
    """
    hmm_id = input_fasta.stem
    output_path = outputs[0]

    gallery_shard = gallery_dir / f"{hmm_id}.parquet"
    hmm_path = hmm_dir / f"{hmm_id}.hmm"

    # Only process if training data exists for this HMM
    if not gallery_shard.exists() or not hmm_path.exists():
        logger.info(f"⏭️ Skipping {hmm_id}: No training reference found at {gallery_shard}")
        return

    try:
        # Execute the unified pipeline logic
        # task_id ensures file isolation in the shared temp_root
        results_df = run_variant_inference(
            inference_fasta=input_fasta,
            gallery_shard=gallery_shard,
            hmm_path=hmm_path,
            temp_dir=temp_root,
            vocab_config=vocab_config,
            task_id=f"inf_{hmm_id}",
            buffer_factor=buffer,
            matrix_type=matrix,
        )

        if results_df.empty:
            logger.warning(f"⚠️ {hmm_id}: No assignments made.")
            return

        # --- Data Contract Enforcement ---
        # Ensure the columns match exactly what the subsequent rules expect
        results_df["clustering_version"] = clustering_version

        # We ensure sequence_aligned is present (returned by run_variant_inference)
        # Columns: [organism_id, hmm_id, variant_id, sequence_aligned, clustering_version]
        final_cols = [
            "organism_id",
            "hmm_id",
            "variant_id",
            "sequence_aligned",
            "clustering_version",
        ]
        results_df[final_cols].to_parquet(output_path)

    except Exception as e:
        logger.error(f"❌ Failed processing {hmm_id}: {e}")


def get_output_paths(input_path: Path, output_dir: Path) -> list[Path]:
    return [output_dir / f"{input_path.stem}.parquet"]


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def run_inference_pipeline(
    config_path: str, code_hash, vocab_path=None, hash_store_dir=None, force: bool = False
):
    # 1. Load Configs
    run_cfg = get_config(str(DEFAULT_RUNTIME_PATH), params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    inf_cfg = get_config(override_path=config_path, params_set="inference_config")
    var_cfg = get_config(override_path=config_path, params_set="msa_var")
    msa_cfg = get_config(override_path=config_path, params_set="msa_config")
    hmm_cfg = get_config(override_path=config_path, params_set="hmm_config")

    # 2. Paths
    input_dir = Path(hmm_cfg.hmm_fasta_dir)  # Dir containing HMM-named fastas
    output_dir = Path(inf_cfg.mapped_output_dir)
    gallery_dir = Path(inf_cfg.gallery_shard_dir)
    hmm_dir = Path(msa_cfg.split_hmm_dir)
    temp_scratch = Path("/tmp/protein_inference")

    # Gallery check
    # --- Path Validation Hook ---
    logger.info(f"🔍 Validating Gallery Shard Path: {gallery_dir.resolve()}")
    if not gallery_dir.exists():
        logger.error(f"❌ CRITICAL: Gallery directory does not exist: {gallery_dir}")
        raise FileNotFoundError(f"Could not find gallery at {gallery_dir}")

    shard_count = len(list(gallery_dir.glob("*.parquet")))
    logger.info(f"   ✅ Found {shard_count} reference shards in gallery.")

    if shard_count == 0:
        logger.error(f"❌ CRITICAL: Gallery exists but contains 0 .parquet files at {gallery_dir}")
        raise ValueError("Gallery directory is empty. Alignment cannot proceed.")
    # ----------------------------

    vocab = get_vocab(override_path=vocab_path)

    # 3. Versioning
    hash_dir = Path(hash_store_dir) if hash_store_dir else DEFAULT_HASH_DIR
    code_version = compute_combined_code_version(
        base_version=gen_cfg.code_version,
        hash_files=code_hash,
        config_items=[str(gallery_dir), var_cfg.dis_mat_method, inf_cfg.get("buffer_factor", 1.5)],
    )

    # 4. Run
    input_files = sorted(list(input_dir.glob("*.fasta")))
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_scratch.mkdir(parents=True, exist_ok=True)

    worker_func = partial(
        inference_worker,
        gallery_dir=gallery_dir,
        hmm_dir=hmm_dir,
        vocab_config=vocab,
        temp_root=temp_scratch,
        matrix=var_cfg.dis_mat_method,
        buffer=inf_cfg.get("buffer_factor", 1.5),
        clustering_version=var_cfg.msa_cluster_name,
    )

    orch = Orchestrator(
        workflow_name=f"unified_inference_{gen_cfg.name}",
        hash_store_dir=hash_dir,
        logger_name="inference",
    )

    orch.run(
        inputs=input_files,
        output_func=partial(get_output_paths, output_dir=output_dir),
        worker=worker_func,
        code_id=code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc=f"Unified Variant Mapping ({gen_cfg.name})",
        wipe_dirs=[output_dir] if force else [],
    )

    # 5. Manifest
    create_manifest(list(output_dir.glob("*.parquet")), output_dir / "inference_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--code_hash", action="append", type=Path, required=True)
    parser.add_argument("--vocab")
    parser.add_argument("--hash_store")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_inference_pipeline(args.config, args.code_hash, args.vocab, args.hash_store, args.force)

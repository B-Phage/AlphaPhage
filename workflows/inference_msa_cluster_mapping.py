#!/usr/bin/env python3
"""
Workflow: Inference Variant Mapping
Maps unseen aligned protein hits to existing cluster variants using a
dynamic, re-aligned gallery to ensure coordinate synchrony.
"""

import argparse
import pathlib
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
import utils.consol_log as log
from orchestration.orchestrator import Orchestrator
from representation.align_table import process_msa_table
from utils.config_loader import get_config, get_vocab
from utils.hash import compute_combined_code_version, create_manifest
from variants.inference_mapper import InferenceMapper

logger = log.get_logger()

# --- Path Anchoring ---
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_RUNTIME_PATH = REPO_ROOT / "ml_assets" / "config" / "runtime.yaml"
DEFAULT_HASH_DIR = REPO_ROOT / "ml_assets" / "data" / "workflow" / "inference_hashes"

# -----------------------------------------------------------------------------
# Multiprocessing Workers
# -----------------------------------------------------------------------------


def mapping_worker(
    input_path: Path,
    outputs: list,
    gallery_dir: Path,
    matrix: str,
    buffer: float,
    vocab_path: Path,
    clustering_version: str,
):
    """
    Worker to map a single HMM's query MSA against its training gallery.

    This worker utilizes the 'Global Re-Alignment' strategy by loading the
    temporary gallery shard created during the MSA stage, ensuring that
    Exact Matches (Distance 0.0) are correctly identified despite new insertions.

    Args:
        input_path: Path to the .csa MSA file for a specific HMM.
        outputs: List containing the target .parquet path.
        gallery_dir: Directory containing the original reference shards.
        matrix: Substitution matrix/distance metric (e.g., 'identity').
        buffer: Multiplier for the stored radius (e.g., 1.1).
        vocab_path: Path to the YAML vocabulary for token mapping.
        clustering_version: Metadata string for version tracking.
    """
    hmm_id = input_path.stem
    output_path = outputs[0]

    # 1. Load Query Data
    # process_msa_table handles the split-index (.csa format) and sets the organism index
    vocab = get_vocab(override_path=str(vocab_path))
    query_df = process_msa_table(input_path, vocab=vocab)

    if query_df.empty:
        logger.warning(f"⚠️ {hmm_id} query alignment is empty. Skipping.")
        pd.DataFrame(
            columns=[
                "organism_id",
                "hmm_id",
                "variant_id",
                "sequence_aligned",
                "clustering_version",
            ]
        ).to_parquet(output_path)
        return

    # 2. Resolve the Dynamic Gallery
    # We prioritize the re-aligned gallery produced by the Global MSA stage.
    # This file has the same variant_ids but NEW coordinates matching the .csa.
    dynamic_gallery_path = input_path.with_suffix(".gallery_tmp.parquet")

    if dynamic_gallery_path.exists():
        logger.info(f"🔗 {hmm_id}: Using dynamic re-aligned gallery for exact matching.")
        gallery_df = pd.read_parquet(dynamic_gallery_path)
    else:
        # Fallback to static gallery (Note: This may cause UNK/2 drift if coordinates shifted)
        logger.warning(f"⚠️ {hmm_id}: Dynamic gallery not found. Falling back to static reference.")
        static_path = gallery_dir / f"{hmm_id}.parquet"
        if not static_path.exists():
            logger.error(f"❌ {hmm_id}: No reference gallery available.")
            return
        gallery_df = pd.read_parquet(static_path)

    # 3. Prepare query_df for Mapper
    # Reset index to get organism_id as a column and join numeric alignment columns
    query_df = query_df.reset_index().rename(columns={"index": "organism_id"})

    # Identify numeric position columns (0, 1, 2...)
    pos_cols = sorted([c for c in query_df.columns if str(c).isdigit()], key=lambda x: int(x))
    query_df["sequence_aligned"] = query_df[pos_cols].apply(
        lambda x: "".join(x.astype(str)), axis=1
    )

    # 4. Execute Mapping
    # We pass the hmm_id here so the Mapper can include it in the returned DataFrame
    mapper = InferenceMapper(buffer_factor=buffer)
    results_df = mapper.map_to_variants(
        queries_df=cast(pd.DataFrame, query_df[["organism_id", "sequence_aligned"]]),
        gallery_df=gallery_df,
        matrix_type=matrix,
        hmm_id=hmm_id,
    )

    # 5. Metadata and Final Shard Persistence
    results_df["clustering_version"] = clustering_version
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)


def get_mapping_outputs(input_path: Path, output_dir: Path) -> list[Path]:
    """Maps an input .csa file to its target variant shard path."""
    return [output_dir / f"{input_path.stem}.parquet"]


# -----------------------------------------------------------------------------
# Workflow Orchestration
# -----------------------------------------------------------------------------


def run_inference_workflow(
    config_path: str, code_hash, vocab_path=None, hash_store_dir=None, force: bool = False
):
    """
    Orchestrates the Inference Mapping process using the parallel Orchestrator.
    """
    # 1. Load configuration
    run_cfg = get_config(str(DEFAULT_RUNTIME_PATH), params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    inf_cfg = get_config(override_path=config_path, params_set="inference_config")
    var_cfg = get_config(override_path=config_path, params_set="msa_var")

    # 2. Resolve Paths
    query_input_dir = Path(inf_cfg.query_msa_dir)
    output_dir = Path(inf_cfg.mapped_output_dir)

    # Static gallery path (used as fallback or base path)
    raw_gallery_dir = inf_cfg.gallery_shard_dir
    snapshot_root = Path(config_path).parent.parent.resolve()

    if "reference/" in raw_gallery_dir:
        sub_path = raw_gallery_dir.split("reference/")[1]
        gallery_dir = snapshot_root / "reference" / sub_path
    else:
        gallery_dir = Path(raw_gallery_dir).resolve()

    # Vocab resolution
    final_vocab_path = (
        Path(vocab_path) if vocab_path else (REPO_ROOT / "ml_assets/vocab/vocab_v0.0.1.yaml")
    )

    # 3. Code Versioning for cache management
    hash_dir = Path(hash_store_dir) if hash_store_dir else DEFAULT_HASH_DIR
    code_version = compute_combined_code_version(
        base_version=gen_cfg.code_version,
        hash_files=code_hash,
        config_items=[str(gallery_dir), var_cfg.dis_mat_method, inf_cfg.get("buffer_factor", 1.1)],
    )

    # 4. Run Orchestrator
    query_files = sorted(list(query_input_dir.glob("*.csa")))
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_with_args = partial(
        mapping_worker,
        gallery_dir=gallery_dir,
        matrix=var_cfg.dis_mat_method,
        buffer=inf_cfg.get("buffer_factor", 1.1),
        vocab_path=final_vocab_path,
        clustering_version=var_cfg.msa_cluster_name,
    )

    orch = Orchestrator(
        workflow_name=f"inference_{gen_cfg.name}",
        hash_store_dir=hash_dir,
        logger_name="inference",
    )

    orch.run(
        inputs=query_files,
        output_func=partial(get_mapping_outputs, output_dir=output_dir),
        worker=worker_with_args,
        code_id=code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc=f"Mapping variants ({gen_cfg.name})",
        wipe_dirs=[output_dir] if force else [],
    )

    # 5. Final Manifest
    mapped_results = list(output_dir.glob("*.parquet"))
    manifest_path = output_dir / "inference_manifest.json"
    create_manifest(mapped_results, manifest_path)
    logger.info(f"✅ Inference mapping complete. Results at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Map unseen hits to existing variants.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--code_hash", action="append", type=Path, required=True)
    parser.add_argument("--vocab", help="Path to vocab.yaml")
    parser.add_argument("--hash_store", help="Metadata hash directory")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    run_inference_workflow(
        config_path=args.config,
        code_hash=args.code_hash,
        vocab_path=args.vocab,
        hash_store_dir=args.hash_store,
        force=args.force,
    )


if __name__ == "__main__":
    main()

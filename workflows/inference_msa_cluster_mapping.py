#!/usr/bin/env python3
"""
Workflow: Inference Variant Mapping
Maps unseen aligned protein hits to existing cluster variants
using a stored gallery and distance thresholds.
"""

import argparse
import os
import pathlib
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

# --- Path Anchoring ---
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_RUNTIME_PATH = REPO_ROOT / "ml_assets" / "config" / "runtime.yaml"
# Note: we default to a specific inference hash folder
DEFAULT_HASH_DIR = REPO_ROOT / "ml_assets" / "data" / "workflow" / "inference_hashes"

# -----------------------------------------------------------------------------
# Multiprocessing Workers (Top-level for pickle safety)
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
    Enforces a shared coordinate system between Inference and Training.
    """
    hmm_id = input_path.stem
    output_path = outputs[0]

    # 1. Load Query Data
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

    # 2. Handle Missing Gallery (Produce empty lake shard for Aggregator)
    gallery_path = gallery_dir / f"{hmm_id}.parquet"
    if not gallery_path.exists():
        logger.info(f"ℹ️ No gallery for {hmm_id}. Creating empty shard.")
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

    gallery_df = pd.read_parquet(gallery_path)

    # --- COORDINATE ALIGNMENT & RECONSTRUCTION ---
    # We must ensure Query MSA columns match the Training Gallery width exactly.
    # Training width represents the HMM Match States. Any additional cols in Query are insertions.
    gallery_width = len(gallery_df["sequence_aligned"].iloc[0])

    # Identify numeric columns (the MSA positions)
    position_cols = [c for c in query_df.columns if str(c).isdigit()]

    # Select only the columns that exist in the Gallery's coordinate space (0 to width-1)
    # This prevents the 'IndexError' by dropping novel insertions relative to the HMM.
    match_cols = [str(i) for i in range(gallery_width) if str(i) in position_cols]

    # Check if we are missing match states (deletions) - we pad those with gaps '-'
    if len(match_cols) < gallery_width:
        # If query_df is missing some columns that should be match states,
        # we ensure they exist as empty/gap columns.
        for i in range(gallery_width):
            if str(i) not in query_df.columns:
                query_df[str(i)] = "-"
        match_cols = [str(i) for i in range(gallery_width)]

    # Reconstruct 'sequence_aligned' using the fixed Training Coordinate System
    # Sort ensure we aren't joining columns out of order
    match_cols = sorted(match_cols, key=lambda x: int(x))
    query_df["sequence_aligned"] = (
        query_df[match_cols].fillna("-").apply(lambda x: "".join(x.astype(str)), axis=1)
    )

    # Ensure organism_id is present
    if "organism_id" not in query_df.columns:
        # Check standard index names from process_msa_table
        if query_df.index.name in ["query_id", "organism_id"] or query_df.index.name is None:
            query_df = query_df.reset_index().rename(
                columns={"index": "organism_id", "query_id": "organism_id"}
            )

    # Slim down to Mapper input contract
    query_df = query_df[["organism_id", "sequence_aligned"]]

    # 3. Execute Mapping
    mapper = InferenceMapper(buffer_factor=buffer)
    results_df = mapper.map_to_variants(
        queries_df=query_df, gallery_df=gallery_df, matrix_type=matrix
    )

    # 4. ENFORCE VARIANT LAKE DATA CONTRACT
    results_df["hmm_id"] = hmm_id
    results_df["clustering_version"] = clustering_version

    # Save the aligned shard
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)


def get_mapping_outputs(input_path: Path, output_dir: Path) -> list[Path]:
    return [output_dir / f"{input_path.stem}_mapped.parquet"]


# -----------------------------------------------------------------------------
# Workflow Orchestration
# -----------------------------------------------------------------------------


def run_inference_workflow(
    config_path: str, code_hash, vocab_path=None, hash_store_dir=None, force: bool = False
):
    """
    Orchestrates the Inference Mapping process.
    """
    # 1. Load configuration
    run_cfg = get_config(str(DEFAULT_RUNTIME_PATH), params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    inf_cfg = get_config(override_path=config_path, params_set="inference_config")
    var_cfg = get_config(override_path=config_path, params_set="msa_var")

    # 2. Resolve Paths
    query_input_dir = Path(inf_cfg.query_msa_dir)

    # 1. Resolve Snapshot Root (Config is in <snapshot>/runtime_config/file.yaml)
    snapshot_root = Path(config_path).parent.parent.resolve()

    # 2. Get the gallery path from config
    # It likely looks like: "../../databases/inference/reference/phage/..."
    raw_gallery_dir = inf_cfg.gallery_shard_dir

    # 3. Clean the path: remove the relative jumps and the redundant 'databases/inference'
    # We want the part that starts with 'reference/...'
    if "reference/" in raw_gallery_dir:
        sub_path = raw_gallery_dir.split("reference/")[1]
        gallery_dir = snapshot_root / "reference" / sub_path
    else:
        gallery_dir = Path(raw_gallery_dir).resolve()

    logger.info(f"🔎 Final Gallery Path: {gallery_dir}")

    if not gallery_dir.exists() or not list(gallery_dir.glob("*.parquet")):
        raise FileNotFoundError(f"❌ Gallery directory is empty or missing: {gallery_dir}")

    output_dir = Path(inf_cfg.mapped_output_dir)

    # Vocabulary path logic:
    # Use the one inside the snapshot if it exists, else fall back to repo
    final_vocab_path = (
        Path(vocab_path) if vocab_path else (REPO_ROOT / "ml_assets/vocab/vocab_v0.0.1.yaml")
    )

    if not final_vocab_path.exists():
        raise FileNotFoundError(f"❌ Vocab file not found at: {final_vocab_path}")

    # 3. Metadata/Hash Resolution
    hash_dir = Path(hash_store_dir) if hash_store_dir else DEFAULT_HASH_DIR

    code_version = compute_combined_code_version(
        base_version=gen_cfg.code_version,
        hash_files=code_hash,
        config_items=[str(gallery_dir), var_cfg.dis_mat_method, inf_cfg.get("buffer_factor", 1.1)],
    )

    # 4. Setup environment
    query_files = sorted(list(query_input_dir.glob("*.csa")))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Partial function for worker
    cluster_version_str = var_cfg.msa_cluster_name

    worker_with_args = partial(
        mapping_worker,
        gallery_dir=gallery_dir,
        matrix=var_cfg.dis_mat_method,
        buffer=inf_cfg.get("buffer_factor", 1.1),
        vocab_path=final_vocab_path,
        clustering_version=cluster_version_str,  # Pass it here
    )

    output_func_with_args = partial(get_mapping_outputs, output_dir=output_dir)

    # 6. Run Orchestrator
    orch = Orchestrator(
        workflow_name=f"inference_{gen_cfg.get('name', 'test')}",
        hash_store_dir=hash_dir,
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

    # 7. Manifest of results
    mapped_results = list(output_dir.glob("*.parquet"))
    manifest_path = output_dir / "inference_manifest.json"
    create_manifest(mapped_results, manifest_path)
    logger.info(f"✅ Inference mapping complete. Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Map unseen hits to existing variants.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--code_hash", action="append", type=Path, required=True)
    parser.add_argument("--vocab", help="Path to the specific vocab YAML file")
    parser.add_argument("--hash_store", help="Optional override for metadata hashes")
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

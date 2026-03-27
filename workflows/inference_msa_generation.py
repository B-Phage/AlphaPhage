#!/usr/bin/env python3
"""
Workflow: Global Inference MSA Generation
Performs a joint alignment of Training Gallery sequences and Inference Query sequences
to ensure a unified coordinate system for distance-based mapping.
"""

import argparse
import pathlib
from functools import partial
from pathlib import Path
from typing import List, cast

import pandas as pd
import pyhmmer
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config
from utils.consol_log import get_logger
from utils.hash import compute_combined_code_version, create_manifest

logger = get_logger()

# --- Path Anchoring ---
SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_RUNTIME_PATH = REPO_ROOT / "ml_assets" / "config" / "runtime.yaml"
DEFAULT_HASH_DIR = REPO_ROOT / "ml_assets" / "data" / "workflow" / "hashes"

# -----------------------------------------------------------------------------
# Core MSA Logic
# -----------------------------------------------------------------------------


def make_global_alignment_msa(
    query_fasta: Path, hmm_path: Path, gallery_parquet: Path, out_csa_path: Path
) -> Path:
    """
    Jointly aligns Gallery (Training) and Query (Inference) sequences.

    Produces:
        1. {hmm_id}.csa: The aligned query sequences for the aggregator.
        2. {hmm_id}.gallery_tmp.parquet: The re-aligned gallery for the mapper.
    """
    out_csa_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load HMM
    with pyhmmer.plan7.HMMFile(hmm_path) as hmm_file:
        hmm = hmm_file.read()

    alphabet = pyhmmer.easel.Alphabet.amino()

    # 2. Extract Raw Sequences from Gallery Parquet (The Template)
    gallery_df = pd.read_parquet(gallery_parquet)
    all_sequences = []

    for _, row in gallery_df.iterrows():
        # Remove gaps to revert to raw biology
        raw = str(row["sequence_aligned"]).replace("-", "").replace(".", "")
        # Tag names to distinguish them after alignment
        name = f"GALLERY|{row['organism_id']}|{row['variant_id']}".encode()
        all_sequences.append(pyhmmer.easel.TextSequence(name=name, sequence=raw).digitize(alphabet))

    # 3. Load Query Sequences
    with pyhmmer.easel.SequenceFile(str(query_fasta), digital=True) as seqs_file:
        query_block = seqs_file.read_block()
        for seq in query_block:
            seq.name = f"QUERY|{seq.name.decode()}".encode()
            all_sequences.append(seq)

    # 4. Joint Alignment (Incremental Profile Alignment)
    # trim=False + all_consensus_cols=True preserves ALL insertions found in both sets
    full_block = pyhmmer.easel.DigitalSequenceBlock(alphabet, all_sequences)
    msa_raw = pyhmmer.hmmer.hmmalign(hmm, full_block, trim=False, all_consensus_cols=True)
    msa = cast(pyhmmer.plan7.MSA, msa_raw)

    # 5. Process Results
    query_results = []
    gallery_results = []

    for name_bytes, sequence in zip(msa.names, msa.alignment):
        full_name = name_bytes.decode()
        parts = full_name.split("|")
        tag = parts[0]

        if tag == "QUERY":
            # format: seq,organism_id
            query_results.append(f"{sequence},{parts[1]}\n")
        else:
            # format: tag, organism_id, variant_id, sequence
            gallery_results.append(
                {"organism_id": parts[1], "variant_id": parts[2], "sequence_aligned": str(sequence)}
            )

    # 6. Save Outputs
    # Write the .csa file for Stage 5/6
    with open(out_csa_path, "w") as f:
        f.writelines(query_results)

    # Write the temporary re-aligned gallery for the Stage 5 Mapper
    # This file has the same columns as the training gallery but NEW alignments
    # We save radii from the original gallery to ensure distance checks still work
    df_gal_new = pd.DataFrame(gallery_results)
    df_gal_final = df_gal_new.merge(
        gallery_df[["organism_id", "radius"]], on="organism_id", how="left"
    )
    df_gal_final.to_parquet(out_csa_path.with_suffix(".gallery_tmp.parquet"))

    return out_csa_path


# -----------------------------------------------------------------------------
# Multiprocessing Worker
# -----------------------------------------------------------------------------


def msa_inference_worker(input_path: Path, outputs: list, split_hmm_path: Path, gallery_dir: Path):
    """
    Worker that orchestrates the global re-alignment for a single HMM.
    If no gallery exists, the HMM is skipped as it is not part of the training set.
    """
    hmm_id = input_path.stem
    out_csa = outputs[0]
    individual_hmm = split_hmm_path / f"{hmm_id}.hmm"
    gallery_shard = gallery_dir / f"{hmm_id}.parquet"

    # Only process HMMs that have a training reference
    if not gallery_shard.exists():
        logger.info(f"⏭️ Skipping {hmm_id}: No training gallery found.")
        return

    try:
        make_global_alignment_msa(
            query_fasta=input_path,
            hmm_path=individual_hmm,
            gallery_parquet=gallery_shard,
            out_csa_path=out_csa,
        )
    except Exception as e:
        logger.error(f"❌ Failed to align {hmm_id}: {e}")


def get_msa_outputs(input_path: Path, output_dir: Path) -> list[Path]:
    """Maps input fasta to expected .csa output."""
    return [output_dir / f"{input_path.stem}.csa"]


# -----------------------------------------------------------------------------
# Workflow Orchestration
# -----------------------------------------------------------------------------


def run_msa_inference_workflow(
    config_path: str, code_hash, hash_store_dir=None, force: bool = False
):
    run_cfg = get_config(str(DEFAULT_RUNTIME_PATH), params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    inf_cfg = get_config(override_path=config_path, params_set="inference_config")
    msa_cfg = get_config(override_path=config_path, params_set="msa_config")
    hmm_cfg = get_config(override_path=config_path, params_set="hmm_config")

    # Resolve Paths
    hmm_fasta_dir = Path(hmm_cfg.hmm_fasta_dir)
    split_path = Path(msa_cfg.split_hmm_dir)
    output_dir = Path(msa_cfg.msa_csa_dir)
    gallery_dir = Path(inf_cfg.gallery_shard_dir)
    hash_dir = Path(hash_store_dir) if hash_store_dir else DEFAULT_HASH_DIR

    hmm_fasta_files = sorted(hmm_fasta_dir.glob("*.fasta"))
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_args = partial(msa_inference_worker, split_hmm_path=split_path, gallery_dir=gallery_dir)

    orch = Orchestrator(
        workflow_name=f"inf_global_msa_{gen_cfg.name}",
        hash_store_dir=hash_dir,
        logger_name="msa_inf",
    )

    orch.run(
        inputs=hmm_fasta_files,
        output_func=partial(get_msa_outputs, output_dir=output_dir),
        worker=worker_args,
        code_id=gen_cfg.code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc=f"Global MSA Alignment ({gen_cfg.name})",
        wipe_dirs=[output_dir] if force else [],
    )

    # Manifest
    msa_results = list(output_dir.glob("*.csa"))
    manifest_path = Path(str(msa_cfg.msa_csa_dir) + "_manifest.json")
    create_manifest(msa_results, manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--code_hash", action="append", type=Path)
    parser.add_argument("--hash_store")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_msa_inference_workflow(args.config, args.code_hash, args.hash_store, args.force)

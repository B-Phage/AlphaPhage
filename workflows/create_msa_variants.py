#!/usr/bin/env python3
"""
Workflow: Multiple Sequence Alignment to Variant Assignment
Processes HMM hit based MSAs into .mcd and .mva variant files

rule variants:
    input:
        hmm_msa_file = f'{hmm_msa_dir}/{{hmms}}.csa'
    output:
        mcd    = f'{out_dir}/{msa_cluster_name}/{{hmms}}.mcd',
        mva    = f'{out_dir}/{msa_cluster_name}/{{hmms}}.mva'
    params:
        msa_cluster_dir     = f'{out_dir}/{msa_cluster_name} ',
        split_equal_window  = config['split_equal_window'],
        # window_len          = config['window_len']
    shell:
        'python ../../src/variants_build.py ' # This gives more hits
        '--in_file              {input.hmm_msa_file} '
        '--msa_cluster_dir      {params.msa_cluster_dir} '
        '--num_cluster_method   {cluster_method} '
        '--split_equal_window   {params.split_equal_window} '
        '--num_cols_nominal     {window_len} '

"""
#!/usr/bin/env python3
"""
Workflow: MSA Variant Generation
Processes MSA files into cluster feature tables (.mva and .mcd)
using orchestrated parallel processing.
"""

import argparse
import os
from functools import partial
from pathlib import Path

import utils.consol_log as log
from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config, get_vocab
from utils.hash import compute_combined_code_version, create_manifest
from variants.build import build_variants

logger = log.get_logger()

DEFAULT_RUNTIME_PATH = "ml_assets/config/runtime.yaml"
HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")

# -----------------------------------------------------------------------------
# Multiprocessing Workers
# -----------------------------------------------------------------------------


def variant_worker(input_path: Path, outputs: list, vocab, num_clusters: int, matrix: str):
    """
    Executes the variant building for a single MSA file.

    Args:
        input_path: Path to the input MSA (.fasta or .fa) file.
        outputs: List containing target output paths (used here for directory resolution).
        vocab: Vocabulary configuration object.
        num_clusters: Number of clusters for the algorithm.
        matrix: Distance matrix type ('identity' or 'blosum62').
    """
    # build_variants handles its own internal path joining,
    # but we pass the parent directory of our expected outputs.
    msa_cluster_dir = outputs[0].parent

    build_variants(
        in_file=str(input_path),
        msa_cluster_dir=str(msa_cluster_dir),
        vocab=vocab,
        num_clusters=num_clusters,
        matrix=matrix,
        reduction_method=None,  # Set to 'diversity' if needed via config
    )


def get_variant_outputs(input_path: Path, output_dir: Path) -> list[Path]:
    """
    Maps an input MSA file to its expected .mva and .mcd output paths.
    """
    hmm_name = input_path.stem
    return [output_dir / f"{hmm_name}.mva", output_dir / f"{hmm_name}.mcd"]


# -----------------------------------------------------------------------------
# Workflow Orchestration
# -----------------------------------------------------------------------------


def run_variant_workflow(config_path: str, code_hash, force: bool = False):
    """
    Orchestrates the MSA Variant generation process.
    """
    # Load configuration sets
    run_cfg = get_config(DEFAULT_RUNTIME_PATH, params_set="meta")
    gen_cfg = get_config(override_path=config_path, params_set="general")
    msa_cfg = get_config(override_path=config_path, params_set="msa_config")
    var_cfg = get_config(override_path=config_path, params_set="msa_var")
    vocab = get_vocab()

    # Path resolution
    # We take inputs from the directory where the previous MSA step stored them
    msa_input_dir = Path(msa_cfg.msa_csa_dir)
    output_dir = Path(var_cfg.msa_variant_dir) / var_cfg.msa_cluster_name

    logger.info(f"The msa var output dir is {output_dir}")
    logger.info(f"The msa input dir is {msa_input_dir}")

    conf_code_version = gen_cfg.code_version
    code_version = compute_combined_code_version(
        base_version=conf_code_version,
        hash_files=code_hash,
        config_items=[msa_cfg.msa_csa_dir, var_cfg],
    )

    # Setup environment
    msa_files = sorted(list(msa_input_dir.glob("*.csa")))
    os.makedirs(output_dir, exist_ok=True)

    # Prepare worker with fixed arguments
    worker_with_args = partial(
        variant_worker,
        vocab=vocab,
        num_clusters=var_cfg.cluster_method,
        matrix=var_cfg.dis_mat_method,
    )

    output_func_with_args = partial(get_variant_outputs, output_dir=output_dir)

    orch = Orchestrator(
        workflow_name=f"variants_{gen_cfg.get('name', 'test')}",
        hash_store_dir=HASH_STORE_DIR,
        logger_name="variants",
    )

    orch.run(
        inputs=msa_files,
        output_func=output_func_with_args,
        worker=worker_with_args,
        code_id=code_version,
        parallelism=run_cfg.n_jobs,
        force=force,
        tqdm_desc=f"Variant processing ({gen_cfg.get('name', 'test')})",
        wipe_dirs=[output_dir],
    )

    # Generate manifest of the primary variant files (.mva)
    variant_results = list(output_dir.glob("*.mva"))
    MANIFEST_FILE = output_dir.parent / f"{var_cfg.msa_cluster_name}_manifest.json"
    create_manifest(variant_results, MANIFEST_FILE)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate variant tables from MSA files.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--code_hash", action="append", type=Path, required=True, help="Code hash files."
    )
    parser.add_argument("--force", action="store_true", help="Rerun all inputs.")

    args = parser.parse_args(argv)
    run_variant_workflow(config_path=args.config, code_hash=args.code_hash, force=args.force)


if __name__ == "__main__":
    main()

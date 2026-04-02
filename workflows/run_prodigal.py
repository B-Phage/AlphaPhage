#!/usr/bin/env python3
import argparse
import os
import pathlib
import subprocess
from pathlib import Path
from typing import List, Optional

from orchestration.orchestrator import Orchestrator
from utils.config_loader import get_config
from utils.hash import compute_combined_code_version, create_manifest

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
RUNTIME_CONFIG_PATH = REPO_ROOT / "ml_assets" / "config" / "runtime.yaml"


class ProdigalRunner:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def get_output_paths(self, input_path: Path) -> List[Path]:
        """Maps a genome fasta to its expected protein output path."""
        return [self.output_dir / f"{input_path.stem}.fa"]

    def run_prodigal_worker(self, input_path: Path, outputs: List[Path]) -> List[Path]:
        """Executes prodigal-gv on a single genome."""
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


def get_prodigal_inputs(genome_dir: Path, lst_genomes: Optional[List[str]]) -> List[Path]:
    """Retrieves the list of target fasta files."""
    if lst_genomes:
        return [genome_dir / f"{g}.fasta" for g in lst_genomes]
    return sorted(genome_dir.glob("*.fasta"))


def run_workflow(
    config_path: str, hash_files: List[Path], hash_store_dir: str, force: bool = False
):
    """
    Orchestrates Prodigal processing with environment isolation.
    """
    # 1. Load Configuration
    # We load the config from the path provided (Training or Inference path)
    meta_cfg = get_config(
        str(RUNTIME_CONFIG_PATH), params_set="meta"
    )  # config that does not affect outputs
    gen_cfg = get_config(config_path, params_set="general")
    prod_cfg = get_config(config_path, params_set="prodigal_config")

    # 2. Define Paths
    genome_dir = Path(prod_cfg.genome_dir)
    output_dir = Path(prod_cfg.proteins_dir)
    hash_store = Path(hash_store_dir)

    # 3. Compute Code Version
    # This hash includes the code_hash files AND the prodigal_config values
    code_version = compute_combined_code_version(
        base_version=gen_cfg.code_version, hash_files=hash_files, config_items=[prod_cfg]
    )

    # 4. Initialize Components
    runner = ProdigalRunner(output_dir)
    genome_files = get_prodigal_inputs(genome_dir, gen_cfg.get("lst_genomes"))

    orch = Orchestrator(
        workflow_name=f"prodigal_{gen_cfg.get('name', 'prod')}",
        hash_store_dir=hash_store,
        logger_name="prodigal",
    )

    # 5. Execute with Orchestrator logic
    # The orchestrator handles 'wipe_dirs' internally if code_id changes.
    orch.run(
        inputs=genome_files,
        output_func=runner.get_output_paths,
        worker=runner.run_prodigal_worker,
        code_id=code_version,
        parallelism=meta_cfg.get("n_jobs", 1),
        force=force,
        tqdm_desc=f"Prodigal ({gen_cfg.get('name', 'run')})",
        wipe_dirs=[output_dir],  # Explicitly passed for version-change cleaning
    )

    # 6. Manifest Generation
    protein_files = list(output_dir.glob("*.fa"))
    manifest_file = Path(str(output_dir) + "_manifest.json")
    create_manifest(protein_files, manifest_file)

    return manifest_file


# -----------------------------
# CLI Interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated Prodigal Workflow")
    parser.add_argument("--config", type=str, required=True, help="Path to runtime YAML")
    parser.add_argument(
        "--code_hash", action="append", type=Path, required=True, help="Repeatable hash files"
    )
    parser.add_argument(
        "--hash_store",
        type=str,
        default="ml_assets/data/workflow/hashes",
        help="Directory for orchestration hashes",
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild regardless of version")

    args = parser.parse_args()

    run_workflow(
        config_path=args.config,
        hash_files=args.code_hash,
        hash_store_dir=args.hash_store,
        force=args.force,
    )

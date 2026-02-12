import hashlib
import shutil
import subprocess
import time
from pathlib import Path

import pytest

PIPELINE_PATH = Path("workflows/run_hmm_scan_pipeline.py")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def write_test_config(
    tmp_path, proteins_dir, results_dir, data_lake, lst_genomes, min_bit_score=25.0
):
    """
    Write a minimal valid YAML config that matches what the pipeline expects.
    Uses real test HMM and annotations.
    """
    config_path = tmp_path / "config.yaml"
    config_content = f"""
general:
  lst_genomes: {lst_genomes}
  code_version: "test-code-v1"

prodigal_config:
  proteins_dir: "{proteins_dir}"

hmm_config:
  hmm_scan_dir: "{results_dir}"
  data_lake: "{data_lake}"
  hmm_multifile: "tests/fixtures/phrog_126.hmm"
  annotations: "tests/fixtures/phrog_annot_test.tsv"
  hmm_scan_method: "phrogs"
  min_bit_score: {min_bit_score}
"""
    config_path.write_text(config_content)
    return config_path


def run_pipeline(config_path):
    """
    Execute the pipeline script as a subprocess, simulating real-world usage.
    """
    result = subprocess.run(
        [
            f"{shutil.which('python3')}",
            str(PIPELINE_PATH),
            "--config",
            str(config_path),
            "--code_hash",
            str(config_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Pipeline failed with exit code {result.returncode}")


def file_hash(path: Path):
    """
    Compute SHA256 hash of a file.
    """
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_hmm_pipeline_runs_and_creates_outputs(tmp_path):
    """
    Test that the pipeline runs with the small real HMM and protein files,
    and creates expected CSV outputs, DuckDB file, and hash store.
    """
    proteins_path = Path("tests/fixtures")
    results = tmp_path / "results"
    data_lake = tmp_path / "data_lake"
    lst_genomes = ["phrog_126"]

    # Write YAML config
    config_path = write_test_config(tmp_path, proteins_path, results, data_lake, lst_genomes)

    # Run pipeline
    run_pipeline(config_path)

    # Outputs created
    for genome in lst_genomes:
        assert (results / f"{genome}.csv").exists()

    # DuckDB file created
    assert data_lake.exists()

    # Hash store created
    hash_store = Path("ml_assets/data/workflow/hashes")
    assert any(hash_store.glob("hmm_scan_*.json"))


def test_pipeline_skips_unchanged_inputs(tmp_path):
    """
    Test that the pipeline skips rerunning proteins if inputs are unchanged.
    """
    proteins_path = Path("tests/fixtures").resolve()
    results = tmp_path / "results"
    data_lake = tmp_path / "data_lake"
    lst_genomes = ["phrog_126"]

    config_path = write_test_config(tmp_path, proteins_path, results, data_lake, lst_genomes)

    # First run
    run_pipeline(config_path)

    # Capture original CSV hash
    csv_file = results / f"{lst_genomes[0]}.csv"
    hash_before = file_hash(csv_file)

    # Second run (should skip unchanged input)
    run_pipeline(config_path)
    hash_after = file_hash(csv_file)

    # File should not change
    assert hash_before == hash_after


def test_new_protein_triggers_partial_rerun(tmp_path):
    """
    Adding a new protein should trigger a rerun only for the new file.
    """
    proteins_path = tmp_path / "proteins"
    results = tmp_path / "results"
    duckdb = tmp_path / "data_lake"
    proteins_path.mkdir()

    # Copy small real protein files from fixtures
    fixture_dir = Path("tests/fixtures").resolve()
    for genome in ["phrog_126"]:
        shutil.copy(fixture_dir / f"{genome}.fa", proteins_path / f"{genome}.fa")
        shutil.copy(fixture_dir / f"{genome}.fa", proteins_path / f"{genome}_copy.fa")

    # Run pipeline for genome1
    config_path = write_test_config(tmp_path, proteins_path, results, duckdb, ["phrog_126"])
    run_pipeline(config_path)
    assert (results / "phrog_126.csv").exists()

    # Capture hash for genome1
    hash_genome1_before = file_hash(results / "phrog_126.csv")

    # Run pipeline for genome2 (partial rerun)
    config_path = write_test_config(tmp_path, proteins_path, results, duckdb, ["phrog_126_copy"])
    run_pipeline(config_path)
    assert (results / "phrog_126_copy.csv").exists()

    # Check genome1 file was unchanged
    hash_genome1_after = file_hash(results / "phrog_126.csv")
    assert hash_genome1_before == hash_genome1_after


def test_config_change_forces_rerun(tmp_path):
    """
    Changing the config (even with same protein) should force rerun.
    """
    proteins_path = Path("tests/fixtures")
    results = tmp_path / "results"
    duckdb = tmp_path / "data_lake"
    lst_genomes = ["phrog_126"]

    # Initial run
    config_path = write_test_config(
        tmp_path, proteins_path, results, duckdb, lst_genomes, min_bit_score=25
    )
    run_pipeline(config_path)

    csv_file = results / f"{lst_genomes[0]}.csv"

    # Simulate config change (change min_bit_score)
    mtime_before = csv_file.stat().st_mtime
    time.sleep(1)
    config_path = write_test_config(
        tmp_path, proteins_path, results, duckdb, lst_genomes, min_bit_score=50
    )
    run_pipeline(config_path)

    # File should be updated because config hash changed
    mtime_after = csv_file.stat().st_mtime
    assert mtime_after > mtime_before

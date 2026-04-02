import shutil
from pathlib import Path

import pytest
import yaml  # pip install pyyaml
from utils import config_loader

from workflows.create_msas import run_msa_workflow  # adjust import as needed

HASH_STORE_DIR = Path("ml_assets/data/workflow/hashes")


@pytest.mark.integration
class TestMSAWorkflow:
    """Integration tests for MSA workflow with cleanup of orchestrator hashes and temp config."""

    @pytest.fixture(autouse=True)
    def setup_temp_dirs(self, tmp_path):
        """Setup temporary input, split, and output directories and copy test files."""
        self.input_dir = tmp_path / "hmm_inputs"
        self.split_dir = tmp_path / "hmm_split"
        self.output_dir = tmp_path / "msa_outputs"
        self.input_dir.mkdir()
        self.split_dir.mkdir()
        self.output_dir.mkdir()

        # Copy test input files
        src = Path("tests/data/phrog_126.fa")
        shutil.copy("tests/data/phrog_126.fa", self.input_dir / "phrog_126.fasta")
        shutil.copy("tests/data/DUF1244.fa", self.input_dir / "DUF1244.fasta")
        shutil.copy("tests/data/DUF1244.hmm", self.split_dir)
        shutil.copy("tests/data/phrog_126.hmm", self.split_dir)

        # ------------------------------
        # Create a temporary config file
        # ------------------------------
        self.config_file = tmp_path / "test_config.yaml"
        config_data = {
            "self_path": str(self.config_file),
            "general": {
                "code_version": "v1.0.0",
                "name": "test",
            },
            "hmm_config": {
                "hmm_scan_method": "phrogs",
                "min_bit_score": 50,
                "db_fingerprint": "ml_assets/data/artifacts/hmm_phrog_summary.fingerprint",
                "summary_tbl": "hmm_phrog_summary",
                "hmm_fasta_dir": str(self.input_dir),
            },
            "msa_config": {
                "split_hmm_dir": str(self.split_dir),
                "msa_csa_dir": str(self.output_dir),
            },
        }
        with open(self.config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        # Patch config loader to point to temp dirs
        self.orig_get_config = config_loader.get_config

        def get_config_patch(override_path=None, params_set=None):
            cfg = self.orig_get_config(override_path, params_set)
            if params_set == "hmm_config":
                cfg.hmm_fasta_dir = self.input_dir
            if params_set == "msa_config":
                cfg.split_hmm_dir = self.split_dir
                cfg.msa_csa_dir = self.output_dir
            return cfg

        config_loader.get_config = get_config_patch

        self.code_hash = []

        yield

        # Restore original config loader
        config_loader.get_config = self.orig_get_config

        # ------------------------------
        # Cleanup orchestrator hash files
        # ------------------------------
        workflow_name = "msa_test"  # must match run_msa_workflow's orchestrator naming
        hash_file = HASH_STORE_DIR / f"{workflow_name}.json"
        code_hash_file = HASH_STORE_DIR / f"{workflow_name}_code.json"

        for f in [hash_file, code_hash_file]:
            if f.exists():
                f.unlink()

    def test_msa_generation_creates_outputs(self):
        """Check that running the workflow produces .csa outputs."""
        run_msa_workflow(
            config_path=str(self.config_file),
            code_hash=self.code_hash,
            force=True,
        )

        output_files = list(self.output_dir.glob("*.csa"))
        assert len(output_files) == 2
        assert all(f.exists() and f.stat().st_size > 0 for f in output_files)

    def test_rerun_triggered_by_config_change(self):
        """Check that a config change triggers rerun of workflow."""
        # First run
        run_msa_workflow(
            config_path=str(self.config_file),
            code_hash=self.code_hash,
            force=True,
        )
        first_run_files = {f.name for f in self.output_dir.glob("*.csa")}

        # Patch config to simulate a change
        orig_get_config = config_loader.get_config

        def config_changed_patch(override_path=None, params_set=None):
            cfg = orig_get_config(override_path, params_set)
            if params_set == "msa_config":
                cfg.split_hmm_dir = self.split_dir
                cfg.msa_csa_dir = self.output_dir
                cfg.some_param = "new_value"  # arbitrary config change
            return cfg

        config_loader.get_config = config_changed_patch

        try:
            # Second run
            run_msa_workflow(
                config_path=str(self.config_file),
                code_hash=self.code_hash,
                force=True,
            )
            second_run_files = {f.name for f in self.output_dir.glob("*.csa")}
            assert first_run_files == second_run_files
        finally:
            config_loader.get_config = self.orig_get_config

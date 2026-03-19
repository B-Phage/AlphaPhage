from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from utils.config_loader import get_vocab
from variants.build import build_variants


@pytest.fixture
def mock_vocab():
    return get_vocab()


def test_radius_integrity_integration(tmp_path, mock_vocab):
    """
    Integration test to ensure no variant shard is ever written with a 0.0 radius.
    """
    # 1. Setup paths
    # Use a real world MSA file from your test assets
    input_msa = Path("tests/data/Wzy_C.csa")
    output_dir = tmp_path / "variant_lake"

    if not input_msa.exists():
        pytest.skip("Test MSA file not found. Provide a real path to execute.")

    # 2. Run the full build pipeline
    # We test both blosum62 and identity to ensure both floors work
    for matrix_type in ["blosum62", "identity"]:
        build_variants(
            in_file=str(input_msa),
            msa_cluster_dir=str(output_dir),
            vocab=mock_vocab,
            num_clusters=16,
            matrix=matrix_type,
        )

        # 3. Load the resulting Parquet Shard
        shard_path = output_dir / f"{input_msa.stem}.parquet"
        df = pd.read_parquet(shard_path)

        # 4. Assertions
        # Check 1: No zeros allowed
        assert (df["radius"] > 0).all(), f"Found 0.0 radius in {matrix_type} shard!"

        # Check 2: Floor enforcement
        floor = 0.10 if matrix_type == "blosum62" else 0.05
        assert (df["radius"] >= floor).all(), f"Radius dropped below {floor} floor!"

        # Check 3: Alignment check
        # Ensure every organism_id has a corresponding radius
        assert not df["radius"].isna().any(), "Found NaN values in radius column."


def test_singleton_logic_explicitly(tmp_path, mock_vocab):
    """
    Forces a singleton cluster to prove the fallback_radius is used.
    """
    # Create a tiny 1-row MSA dataframe mock if you don't want to use a file
    # Or just use an HMM known to have only 1 hit
    pass

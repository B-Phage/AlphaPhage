from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
from variants.run_variant_inference import run_variant_inference


@pytest.fixture
def mock_params():
    return {
        "inference_fasta": Path("queries.fasta"),
        "gallery_shard": Path("gallery.parquet"),
        "hmm_path": Path("/models/HMM_001.hmm"),
        "temp_dir": Path("/tmp"),
        "vocab_config": MagicMock(),
        "task_id": "task_123",
        "buffer_factor": 1.1,
        "matrix_type": "blosum62",
    }


@patch("variants.run_variant_inference.SeqIO.parse")
@patch("variants.run_variant_inference.pd.read_parquet")
@patch("variants.run_variant_inference.InferencePreProcessor")
@patch("variants.run_variant_inference.InferenceMapper")
@patch("variants.run_variant_inference.Path.exists")
@patch("variants.run_variant_inference.Path.unlink")
def test_run_variant_inference_success(
    mock_unlink, mock_exists, mock_mapper_cls, mock_prep_cls, mock_parquet, mock_seqio, mock_params
):
    # 1. Mock Data Loading
    # Mock SeqIO record
    mock_record = MagicMock()
    mock_record.id = "SEQ_1"
    mock_record.seq = "AAAA"
    mock_seqio.return_value = [mock_record]

    # Mock Gallery Parquet
    mock_parquet.return_value = pd.DataFrame(
        {"organism_id": ["G1"], "sequence_aligned": ["A-A"], "variant_id": ["V1"]}
    )

    # 2. Mock Class Instances
    mock_prep = mock_prep_cls.return_value
    mock_mapper = mock_mapper_cls.return_value

    # Mock PreProcessor output
    mock_prep.prepare_combined_df.return_value = pd.DataFrame({"dummy": ["data"]})
    mock_prep.align_and_map.return_value = pd.DataFrame(
        {
            "organism_id": ["SEQ_1", "G1"],
            "is_inference": [True, False],
            "sequence_aligned": ["AAAA", "AAAA"],
        }
    )

    # Mock Mapper output
    expected_results = pd.DataFrame({"organism_id": ["SEQ_1"], "variant_id": ["V1"]})
    mock_mapper.map_to_variants.return_value = expected_results

    # 3. Setup File Cleanup Mocks
    mock_exists.return_value = True

    # 4. Run Function
    results = run_variant_inference(**mock_params)

    # 5. Assertions
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 1

    # Verify Infrastructure Initialization
    mock_prep_cls.assert_called_once_with(mock_params["hmm_path"], mock_params["vocab_config"])
    mock_mapper_cls.assert_called_once_with(buffer_factor=1.1)

    # Verify Cleanup logic (The 'finally' block)
    # It should try to delete {task_id}_{hmm_id}.fasta and .csa
    assert mock_unlink.call_count == 2
    assert mock_exists.call_count == 2


@patch("variants.run_variant_inference.SeqIO.parse")
@patch("variants.run_variant_inference.pd.read_parquet")
@patch("variants.run_variant_inference.InferencePreProcessor")
@patch("variants.run_variant_inference.Path.exists")
@patch("variants.run_variant_inference.Path.unlink")
def test_run_variant_inference_cleanup_on_failure(
    mock_unlink, mock_exists, mock_prep_cls, mock_parquet, mock_seqio, mock_params
):
    """Ensure that even if the code crashes, the temp files are cleaned up."""
    # Setup mocks to trigger an exception during alignment
    mock_seqio.return_value = [MagicMock(id="S1", seq="A")]
    mock_parquet.return_value = pd.DataFrame({"a": [1]})

    mock_prep = mock_prep_cls.return_value
    mock_prep.prepare_combined_df.side_effect = Exception("Alignment Failed!")

    mock_exists.return_value = True

    # Run and catch exception
    with pytest.raises(Exception, match="Alignment Failed!"):
        run_variant_inference(**mock_params)

    # Verify cleanup still ran
    assert mock_unlink.call_count == 2

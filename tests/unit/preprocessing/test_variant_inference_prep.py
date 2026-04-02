from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from variants.inference_preprocessor import InferencePreProcessor


@pytest.fixture
def mock_vocab():
    mock = MagicMock()
    mock.filler_char = "X"
    return mock


@pytest.fixture
def processor(mock_vocab):
    return InferencePreProcessor(
        hmm_path=Path("/tmp/test.hmm"), vocab_config=mock_vocab, gallery_prefix="gal_"
    )


## --- Unit Tests for Internal Helpers ---


def test_strip_alignment(processor):
    assert processor._strip_alignment("A-B.C_D E") == "ABCDE"
    assert processor._strip_alignment(None) == ""
    assert processor._strip_alignment(np.nan) == ""


def test_clean_organism_id(processor):
    # Test semicolon splitting
    assert processor._clean_organism_id("ESH35;ON528745.1") == "ESH35"
    # Test prefix removal
    assert processor._clean_organism_id("gal_ESH35") == "ESH35"
    # Test combined logic
    assert processor._clean_organism_id("gal_ESH35;extra") == "ESH35"


## --- Stage 1: Logic and Merging ---


def test_prepare_combined_df(processor):
    gallery = pd.DataFrame(
        {"organism_id": ["G1"], "sequence_aligned": ["A-A"], "variant_id": ["V1"]}
    )
    inference = pd.DataFrame({"organism_id": ["I1"], "sequence": ["C-C"]})

    combined = processor.prepare_combined_df(inference, gallery)

    # Check Gallery processing
    gal_row = combined[combined["is_inference"] == False].iloc[0]
    assert gal_row["organism_id"] == "gal_G1"
    assert gal_row["raw_sequence"] == "AA"

    # Check Inference processing
    inf_row = combined[combined["is_inference"] == True].iloc[0]
    assert inf_row["organism_id"] == "I1"
    assert inf_row["raw_sequence"] == "CC"
    assert pd.isna(inf_row["variant_id"])


## --- Stage 2: Alignment Flow ---


@patch("variants.inference_preprocessor.make_msa")
@patch("variants.inference_preprocessor.process_msa_table")
@patch("variants.inference_preprocessor.standardize_frame_chars")
def test_align_and_map_flow(mock_std, mock_process, mock_msa, processor):
    # 1. Setup Input Data
    combined_df = pd.DataFrame(
        {
            "organism_id": ["gal_G1", "I1"],
            "raw_sequence": ["AA", "CC"],
            "is_inference": [False, True],
        }
    )

    # 2. Mock the HMM processing chain
    # Mocking the table returned by process_msa_table
    mock_df_raw = pd.DataFrame([["A", "A"], ["C", "C"]], index=["gal_G1", "I1"], columns=[0, 1])
    mock_process.return_value = mock_df_raw

    # Mock standardize_frame_chars returning (df, info)
    mock_std.return_value = (mock_df_raw, None)

    # 3. Execution
    # Use mock_open to avoid writing real files
    with patch("builtins.open", mock_open()) as mocked_file:
        result = processor.align_and_map(
            combined_df, temp_fasta=Path("test.fasta"), temp_csa=Path("test.csa")
        )

    # 4. Assertions
    assert len(result) == 2
    assert "sequence_aligned" in result.columns
    # Verify that IDs were cleaned during the final merge
    assert list(result["organism_id"]) == ["gal_G1", "I1"]
    # Verify merge preserved boolean status
    assert result.loc[result["organism_id"] == "I1", "is_inference"].iloc[0] == True

    # Verify external tools were called
    mock_msa.assert_called_once()
    mock_process.assert_called_once_with("test.csa", vocab=processor.vocab)


def test_align_and_map_merge_failure_handling(processor):
    """Ensure is_inference defaults correctly if IDs don't match perfectly."""
    combined_df = pd.DataFrame(
        {"organism_id": ["ID_MATCH"], "raw_sequence": ["AA"], "is_inference": [True]}
    )

    # Mock alignment results with a non-matching ID
    with (
        patch("variants.inference_preprocessor.make_msa"),
        patch("variants.inference_preprocessor.process_msa_table") as mock_p,
        patch("variants.inference_preprocessor.standardize_frame_chars") as mock_s,
        patch("builtins.open", mock_open()),
    ):

        mock_p.return_value = pd.DataFrame([["A"]], index=["WRONG_ID"])
        mock_s.return_value = (mock_p.return_value, None)

        result = processor.align_and_map(combined_df, Path("f"), Path("c"))

        # The row from alignment that didn't merge should have is_inference = False
        assert result.iloc[0]["is_inference"] == False

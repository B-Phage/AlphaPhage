import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest
from hmm.duckdb_io import write_hmm_results_with_sequences
from hmm.types import Protein


class FakeProtein(Protein):
    def __init__(self, name: str, sequence: str):
        self.name = name.encode()
        self._sequence = sequence

    def textize(self):
        class Seq:
            sequence = self._sequence

        return Seq()


@pytest.fixture
def hmm_scan_df():
    return pd.DataFrame(
        [
            {
                "protein_name": "protA",
                "organism_name": "org1",
                "hmm_domain_name": "phrog_1",
                "annot": "capsid",
                "evalue": 1e-10,
                "score": 100.0,
                "bias": 0.0,
                "length": 120,
                "top_hmm_domain_hit_on_prot": 1,
                "top_protein_match_for_hmm": 1,
                "best_dom_env_start": 5,
                "best_dom_env_end": 110,
                "best_dom_env_length": 105,
            }
        ]
    )


@pytest.fixture
def protein_block():
    return [
        FakeProtein("protA", "MKTAAA"),
        FakeProtein("protB", "GGGCCC"),  # no HMM hit
    ]


def test_creates_table_when_not_exists(hmm_scan_df, protein_block):
    db = MagicMock()
    db.table_exists.return_value = False
    db.con = MagicMock()

    write_hmm_results_with_sequences(
        hmm_scan_df=hmm_scan_df,
        protein_py_easel_obj=protein_block,
        db=db,
        sql_table_name="hmm_results",
    )

    db.table_exists.assert_called_once_with("hmm_results")
    db.con.execute.assert_called_once()

    # Insert must still occur
    db.insert_dataframe.assert_called_once()


def test_outer_join_includes_proteins_without_hits(hmm_scan_df, protein_block):
    db = MagicMock()
    db.table_exists.return_value = True

    write_hmm_results_with_sequences(
        hmm_scan_df=hmm_scan_df,
        protein_py_easel_obj=protein_block,
        db=db,
        sql_table_name="hmm_results",
    )

    # Grab positional argument
    args, kwargs = db.insert_dataframe.call_args
    df = args[1]  # the DataFrame inserted

    assert set(df["protein_name"]) == {"protA", "protB"}

    seqs = dict(zip(df["protein_name"], df["sequence"]))
    assert seqs["protA"] == "MKTAAA"
    assert seqs["protB"] == "GGGCCC"

    # protB had no HMM hit → NaNs expected
    protB_row = df[df["protein_name"] == "protB"].iloc[0]
    assert protB_row["hmm_domain_name"] == "-"


def test_only_expected_hmm_columns_selected(hmm_scan_df, protein_block):
    db = MagicMock()
    db.table_exists.return_value = True

    write_hmm_results_with_sequences(
        hmm_scan_df=hmm_scan_df,
        protein_py_easel_obj=protein_block,
        db=db,
        sql_table_name="hmm_results",
    )

    args, kwargs = db.insert_dataframe.call_args
    df = args[1]  # second positional argument is the DataFrame

    expected_columns = {
        "protein_name",
        "organism_name",
        "hmm_domain_name",
        "annot",
        "evalue",
        "score",
        "bias",
        "length",
        "top_hmm_domain_hit_on_prot",
        "top_protein_match_for_hmm",
        "best_dom_env_start",
        "best_dom_env_end",
        "best_dom_env_length",
        "sequence",
    }

    assert expected_columns.issubset(df.columns)


def test_primary_keys_are_correct(hmm_scan_df, protein_block):
    db = MagicMock()
    db.table_exists.return_value = True

    write_hmm_results_with_sequences(
        hmm_scan_df=hmm_scan_df,
        protein_py_easel_obj=protein_block,
        db=db,
        sql_table_name="hmm_results",
    )

    _, kwargs = db.insert_dataframe.call_args
    assert kwargs["primary_keys"] == [
        "protein_name",
        "organism_name",
        "hmm_domain_name",
    ]


def test_write_hmm_results_with_sequences_empty_dataframe(caplog):
    # GIVEN an empty HMM scan DataFrame
    empty_df = pd.DataFrame(
        columns=[
            "protein_name",
            "organism_name",
            "hmm_domain_name",
            "annot",
            "evalue",
            "score",
            "bias",
            "length",
            "top_hmm_domain_hit_on_prot",
            "top_protein_match_for_hmm",
            "best_dom_env_start",
            "best_dom_env_end",
            "best_dom_env_length",
        ]
    )

    # Mock protein iterable (can be empty)
    mock_protein_iter = []

    # Mock DuckDBClient
    mock_db = MagicMock()
    mock_db.table_exists.return_value = False
    mock_db.insert_dataframe = MagicMock()

    # Capture logs
    with caplog.at_level(logging.WARNING):
        write_hmm_results_with_sequences(
            hmm_scan_df=empty_df,
            protein_py_easel_obj=mock_protein_iter,
            db=mock_db,
            sql_table_name="test_table",
        )

    # THEN: a warning about no hits is logged
    assert any("No HMM hits to insert" in record.message for record in caplog.records)

    # AND: no DB methods are called
    mock_db.table_exists.assert_not_called()
    mock_db.insert_dataframe.assert_not_called()

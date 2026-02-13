import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, cast

import pyhmmer
import utils.consol_log as log
from hmm.annotations import AnnotationProvider
from hmm.duckdb_io import write_hmm_results_with_sequences
from hmm.io import (
    load_hmms_from_file,
    load_single_hmm_per_file,
)
from hmm.search import run_hmm_search
from hmm.types import Protein
from pyinstrument import Profiler
from storage_io.duckdb_client import DuckDBClient

"""Integration fixture: tiny FASTA, real HMM, and annotation CSV."""

# Use real HMM file from data
# hmm_file = Path("tests/data/DUF1244.hmm")
hmm_file = Path("../databases/hmm/pfam/v38.0/Pfam-A.hmm")
# fasta_file = Path("tests/data/DUF1244.fa")
fasta_file = Path("../databases/Gaborieau_Picard/Picard_Ecoli_25941691/B12I13.fasta")

# Annotation CSV matching the HMM domain
# annotation_file = tmp_path / "annotations.csv"
# annotation_file.write_text("hmm_domain_name,annot\nPHROG_126,TestAnnotation\n")
annotation_file = Path("../databases/hmm/pfam/v38.0/Pfam-A.hmm.dat")
tmp_path = Path("tmp_path")
tmp_path.mkdir(exist_ok=True)

small_pfam_test_data = {
    "fasta": fasta_file,
    "hmm": hmm_file,
    "annotation": annotation_file,
    "out_csv": tmp_path / "results.csv",
    "out_db": tmp_path / "results.duckdb",
}

"""Integration test: run CLI on tiny test dataset using a real HMM file."""
# fmt: off
cmd = [
    sys.executable,
    "workflows/cli/hmm_scan.py",
    "-f", str(small_pfam_test_data["fasta"]),
    "-hp", str(small_pfam_test_data["hmm"]),
    "-a", str(small_pfam_test_data["annotation"]),
    "-m", "pfam",
    "-o", str(small_pfam_test_data["out_csv"]),
    "-db", str(small_pfam_test_data["out_db"]),
]
# fmt: off

profiler = Profiler()
profiler.start()

# result = subprocess.run(cmd, capture_output=True, text=True)


hmms = list(load_hmms_from_file(str(small_pfam_test_data["hmm"])))

    # Load proteins
aa = pyhmmer.easel.Alphabet.amino()
with pyhmmer.easel.SequenceFile(str(small_pfam_test_data["fasta"]), digital=True, alphabet=aa) as seqs_file:
    proteins_block = seqs_file.read_block()
    proteins = cast(Iterable[Protein], proteins_block)

organism_name = os.path.splitext(os.path.basename(str(small_pfam_test_data["fasta"])))[0]

# Load annotation provider
annotations = AnnotationProvider('pfam', str(small_pfam_test_data["annotation"]))

# Run HMM search
df_results = run_hmm_search(
    proteins=proteins_block,
    hmms=hmms,
    annotations=annotations,
    organism_name=organism_name,
)


df_results.to_csv(str(small_pfam_test_data["out_csv"]), sep=",", index=False)

# Optionally write to DuckDB
sql_table_name = f"hmm_pfam_annotations"

db = DuckDBClient(str(small_pfam_test_data["out_db"]))
write_hmm_results_with_sequences(
    df_results,
    proteins,
    db,
    sql_table_name,
)


profiler.stop()
profiler.print()

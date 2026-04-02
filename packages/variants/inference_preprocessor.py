from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from hmm.msa import make_msa

# Project-specific imports
from representation.align_table import process_msa_table
from utils.consol_log import get_logger
from variants.preprocessing_methods import standardize_frame_chars

logger = get_logger()


class InferencePreProcessor:
    def __init__(self, hmm_path: Path, vocab_config, gallery_prefix: str = "gal_"):
        """
        Args:
            hmm_path: Path to the HMM model.
            vocab_config: Configuration containing 'filler_char'.
            gallery_prefix: Prefix to isolate training sequences from inference sequences.
        """
        self.hmm_path = hmm_path
        self.vocab = vocab_config
        self.gallery_prefix = gallery_prefix

    def _strip_alignment(self, sequence: str) -> str:
        """Removes ALL non-alphabetical gap/filler characters."""
        if not isinstance(sequence, str):
            return ""
        # Explicitly remove common gap/junk characters: . - _ and spaces
        for char in ["-", ".", "_", " "]:
            sequence = sequence.replace(char, "")
        return sequence

    def _clean_organism_id(self, org_id: str) -> str:
        """
        Extracts 'ESH35' from 'ESH35;ON528745.1_197'.
        Also removes the gallery prefix if it's already present.
        """
        s_id = str(org_id)
        # 1. Remove the internal gallery prefix if we're re-processing
        if s_id.startswith(self.gallery_prefix):
            s_id = s_id.replace(self.gallery_prefix, "", 1)

        # 2. Split by semicolon and take the first part
        # 3. Strip any whitespace or HMMER-injected characters
        return s_id.split(";")[0].strip()

    def prepare_combined_df(
        self, inference_seqs: pd.DataFrame, gallery_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        STAGE 1: Logic-only merge.
        Prepends prefix to gallery IDs to ensure HMM headers are unique.
        """
        # 1. Process Gallery
        gal = gallery_df.copy()
        gal["organism_id"] = gal["organism_id"].apply(self._clean_organism_id)
        # Apply prefix for HMMER isolation
        gal["organism_id"] = self.gallery_prefix + gal["organism_id"]
        gal["raw_sequence"] = gal["sequence_aligned"].apply(self._strip_alignment)
        gal["is_inference"] = False

        # 2. Process Inference
        inf = inference_seqs.copy()
        # Use 'sequence' if raw, or 'sequence_aligned' if coming from a previous MSA
        seq_col = "sequence" if "sequence" in inf.columns else "sequence_aligned"
        inf["raw_sequence"] = inf[seq_col].apply(self._strip_alignment)
        inf["is_inference"] = True
        inf["organism_id"] = inf["organism_id"].apply(self._clean_organism_id)

        # 3. Combine - Inference rows will have NaN for variant_id and radius
        combined = pd.concat([gal, inf], ignore_index=True, sort=False)
        return combined

    def align_and_map(
        self, combined_df: pd.DataFrame, temp_fasta: Path, temp_csa: Path
    ) -> pd.DataFrame:
        """
        STAGE 2: Physical alignment and metadata mapping.
        """
        # 1. Write the FASTA for pyhmmer
        with open(temp_fasta, "w") as f:
            for row in combined_df.itertuples():
                f.write(f">{row.organism_id}\n{row.raw_sequence}\n")

        # 2. Run MSA (Reusing training infrastructure)
        make_msa(temp_fasta, self.hmm_path, temp_csa)

        # 3. Process HMM output
        df_raw_msa = process_msa_table(str(temp_csa), vocab=self.vocab)
        df_std, _ = standardize_frame_chars(df_raw_msa, filler_char=self.vocab.filler_char)

        # Normalize the index coming back from HMMER
        normalized_ids = df_std.index.astype(str).str.split().str[0].str.strip()

        # 4. Reconstruct aligned strings
        alignment_results = pd.DataFrame(
            {"organism_id": normalized_ids, "sequence_aligned": df_std.values.sum(axis=1)}
        )

        # 5. Final Merge
        # We merge back metadata using the prefixed IDs.
        # The result is a unified DF where is_inference tells you what to map.
        final_df = alignment_results.merge(
            combined_df.drop(columns=["sequence_aligned", "raw_sequence"], errors="ignore"),
            on="organism_id",
            how="left",
        )

        # Force is_inference to be boolean, filling any merge failures with False
        final_df["is_inference"] = final_df["is_inference"].fillna(False).astype(bool)

        return final_df

from pathlib import Path

import pandas as pd
from Bio import SeqIO
from utils.consol_log import get_logger
from variants.inference_mapper import InferenceMapper

# Existing logic imports
from variants.inference_preprocessor import InferencePreProcessor

logger = get_logger()


def run_variant_inference(
    inference_fasta: Path,
    gallery_shard: Path,
    hmm_path: Path,
    temp_dir: Path,
    vocab_config,
    task_id: str,
    buffer_factor: float = 1.5,
    matrix_type: str = "identity",
) -> pd.DataFrame:
    """
    Stand-alone functional interface for mapping FASTA sequences to a
    pre-trained protein variant gallery via HMM alignment.
    """
    # 1. Internal Data Loading
    # Assumes a helper or standard bio-loader for FASTA exists
    # If not, use: pd.DataFrame([{'organism_id': s.id, 'sequence': str(s.seq)} for s in SeqIO.parse(...)])

    inference_df = pd.DataFrame(
        [
            {"organism_id": record.id, "sequence": str(record.seq)}
            for record in SeqIO.parse(inference_fasta, "fasta")
        ]
    )

    gallery_df = pd.read_parquet(gallery_shard)
    hmm_id = hmm_path.stem

    # 2. Setup Transient Paths
    temp_fasta = temp_dir / f"{task_id}_{hmm_id}.fasta"
    temp_csa = temp_dir / f"{task_id}_{hmm_id}.csa"

    # 3. Initialize Infrastructure
    preprocessor = InferencePreProcessor(hmm_path, vocab_config)
    mapper = InferenceMapper(buffer_factor=buffer_factor)

    try:
        # 4. Alignment
        logger.info(f"[{task_id}] Aligning {len(inference_df)} sequences for {hmm_id}")
        combined_meta = preprocessor.prepare_combined_df(inference_df, gallery_df)

        aligned_df = preprocessor.align_and_map(
            combined_meta, temp_fasta=temp_fasta, temp_csa=temp_csa
        )

        # Before splitting
        aligned_df["is_inference"] = aligned_df["is_inference"].fillna(False).astype(bool)

        # 5. Mapping
        # Filters to query/gallery logic inside the mapper call
        results_df = mapper.map_to_variants(
            queries_df=aligned_df[aligned_df["is_inference"]],
            gallery_df=aligned_df[~aligned_df["is_inference"]],
            matrix_type=matrix_type,
            hmm_id=hmm_id,
        )

        return results_df

    finally:
        # 6. Cleanup transient files
        for p in [temp_fasta, temp_csa]:
            if p.exists():
                p.unlink()

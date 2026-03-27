# export KMP_DUPLICATE_LIB_OK=TRUE && conda run -n preprocessing python tests/utils/known_variant_stress_test.py

from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from utils.config_loader import get_vocab
from utils.consol_log import get_logger
from variants.run_variant_inference import run_variant_inference

logger = get_logger()

# Import your unified function
# from variants.pipeline import run_variant_inference


def stress_test_consistency(
    gallery_shard_path: Path, hmm_path: Path, vocab_config, temp_dir: Path, sample_size: int = 50
):
    """
    Validates that known training sequences are mapped to their original
    variants during inference.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    # 1. Load Ground Truth
    gt_df = pd.read_parquet(gallery_shard_path)

    # 2. Sample sequences for the test
    # We take a random sample to ensure we cover different clusters
    test_sample = gt_df.sample(min(sample_size, len(gt_df)), random_state=42)

    # 3. Create Mock Inference FASTA
    # We strip gaps because the inference pipeline expects raw sequences
    mock_fasta = temp_dir / "stress_test_input.fasta"
    records = []
    for _, row in test_sample.iterrows():
        raw_seq = row["sequence_aligned"].replace("-", "").replace(".", "")
        records.append(SeqRecord(Seq(raw_seq), id=row["organism_id"], description=""))

    SeqIO.write(records, mock_fasta, "fasta")

    # 4. Run the Inference Pipeline
    logger.info(f"Running inference on {len(test_sample)} known sequences...")
    results_df = run_variant_inference(
        inference_fasta=mock_fasta,
        gallery_shard=gallery_shard_path,
        hmm_path=hmm_path,
        temp_dir=temp_dir,
        vocab_config=vocab_config,
        task_id="stress_test_v1",
        buffer_factor=1.5,
    )

    # 5. Cross-Reference Results
    # Join the inference results back to our ground truth sample
    comparison = results_df.merge(
        test_sample[["organism_id", "variant_id"]],
        on="organism_id",
        suffixes=("_inferred", "_original"),
    )

    # 6. Evaluate Performance
    comparison["is_correct"] = (
        comparison["variant_id_inferred"] == comparison["variant_id_original"]
    )
    accuracy = comparison["is_correct"].mean() * 100

    print(f"\n--- Stress Test Results: {hmm_path.stem} ---")
    print(f"Total Sequences Tested: {len(comparison)}")
    print(f"Assignment Accuracy: {accuracy:.2f}%")

    if accuracy < 100:
        mismatches = comparison[~comparison["is_correct"]]
        print(f"Mismatches detected: {len(mismatches)}")
        print(mismatches[["organism_id", "variant_id_original", "variant_id_inferred"]])
    else:
        print("Success: All inference assignments matched training ground truth.")

    return comparison


if __name__ == "__main__":

    hmm_name = "Wzy_C"
    gallery_example = f"/Users/tamsen/Code/databases/inference/default_run_ep24_20260325/reference/bac/variant_shards/clusters_16_reduction_false_matrix_blosum62/{hmm_name}.parquet"
    hmm_path = f"/Users/tamsen/Code/databases/hmm/pfam/v38.0/split/{hmm_name}.hmm"
    my_vocab_obj = get_vocab()

    stress_test_consistency(
        gallery_shard_path=Path(gallery_example),
        hmm_path=Path(hmm_path),
        vocab_config=my_vocab_obj,
        temp_dir=Path("/tmp/stress_test"),
    )

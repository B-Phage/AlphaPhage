# export KMP_DUPLICATE_LIB_OK=TRUE && conda run -n preprocessing python tests/utils/unk_variant_stress_test.py

from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from utils.config_loader import get_vocab
from utils.consol_log import get_logger
from variants.run_variant_inference import run_variant_inference

logger = get_logger()


def stress_test_novelty_detection(
    gallery_shard_path: Path,
    hmm_path: Path,
    vocab_config,
    temp_dir: Path,
    num_variants_to_remove: int = 1,
    task_id: str = "unk_test",
):
    """
    Validates that sequences from a completely removed variant are correctly
    flagged as 'UNK' during inference.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the Shard
    full_gallery = pd.read_parquet(gallery_shard_path)

    # 2. Select Variants to Remove
    # We pick variants with multiple members to ensure we have a robust query set
    variant_counts = full_gallery["variant_id"].value_counts()
    eligible_vids = variant_counts.index.tolist()

    if len(eligible_vids) <= num_variants_to_remove:
        raise ValueError("Not enough distinct variants in shard to perform removal test.")

    popped_vids = eligible_vids[:num_variants_to_remove]
    logger.info(f"Removing all sequences for Variant IDs: {popped_vids}")

    # 3. Split Data: Pop the entire variant out
    test_queries = full_gallery[full_gallery["variant_id"].isin(popped_vids)].copy()
    test_gallery = full_gallery[~full_gallery["variant_id"].isin(popped_vids)].copy()

    # 4. Create Mock Inference FASTA
    mock_fasta = temp_dir / f"{task_id}_novel_queries.fasta"
    records = []
    for row in test_queries.itertuples():
        # Clean sequence and force string type to avoid Pylance timedelta issues
        raw_val = str(getattr(row, "sequence_aligned", ""))
        clean_seq = raw_val.replace("-", "").replace(".", "").replace("_", "")
        records.append(SeqRecord(Seq(clean_seq), id=str(row.organism_id), description=""))

    SeqIO.write(records, mock_fasta, "fasta")

    # 5. Save the Reduced Gallery Shard
    temp_shard = temp_dir / f"{task_id}_reduced_gallery.parquet"
    test_gallery.to_parquet(temp_shard)

    # 6. Run Inference with STRICT Buffer (1.0)
    # This means the radius is not expanded at all.
    results_df = run_variant_inference(
        inference_fasta=mock_fasta,
        gallery_shard=temp_shard,
        hmm_path=hmm_path,
        temp_dir=temp_dir,
        vocab_config=vocab_config,
        task_id=task_id,
        buffer_factor=1.0,  # Force strict boundary check
    )

    # 7. Evaluation
    comparison = results_df.merge(
        test_queries[["organism_id", "variant_id"]],
        on="organism_id",
        suffixes=("_assigned", "_ground_truth"),
    )

    # Success here means the assigned variant_id is "UNK"
    comparison["correctly_flagged_unk"] = comparison["variant_id_assigned"] == "UNK"
    unk_rate = comparison["correctly_flagged_unk"].mean() * 100

    print(f"\n--- Novelty Stress Test: {hmm_path.stem} ---")
    print(f"Sequences from removed variants tested: {len(test_queries)}")
    print(f"Novelty Detection Rate (UNK): {unk_rate:.2f}%")

    if unk_rate < 100:
        leaks = comparison[comparison["variant_id_assigned"] != "UNK"]
        print(
            f"⚠️ LEAKAGE: {len(leaks)} sequences were assigned to existing clusters instead of UNK."
        )
        print("Example Leaks (Top 5):")
        print(
            leaks[
                ["organism_id", "variant_id_ground_truth", "variant_id_assigned", "distance"]
            ].head()
        )

    return comparison


if __name__ == "__main__":

    hmm_name = "Wzy_C"
    gallery_example = f"/Users/tamsen/Code/databases/inference/default_run_ep24_20260325/reference/bac/variant_shards/clusters_16_reduction_false_matrix_blosum62/{hmm_name}.parquet"
    hmm_path = f"/Users/tamsen/Code/databases/hmm/pfam/v38.0/split/{hmm_name}.hmm"
    my_vocab_obj = get_vocab()
    num_variants_to_remove = 2

    stress_test_novelty_detection(
        gallery_shard_path=Path(gallery_example),
        hmm_path=Path(hmm_path),
        vocab_config=my_vocab_obj,
        temp_dir=Path("/tmp/stress_test"),
        num_variants_to_remove=num_variants_to_remove,
    )

# export KMP_DUPLICATE_LIB_OK=TRUE && conda run -n preprocessing python tests/utils/similar_variant_stress_test.py

from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from utils.config_loader import get_vocab
from utils.consol_log import get_logger
from variants.run_variant_inference import run_variant_inference

logger = get_logger()


def stress_test_proximal_mapping(
    gallery_shard_path: Path,
    hmm_path: Path,
    vocab_config,
    temp_dir: Path,
    task_id: str = "proximal_test",
):
    """
    Tests if sequences can be correctly mapped to their original variant
    even when their exact match is removed from the gallery.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the Shard
    full_gallery = pd.read_parquet(gallery_shard_path)

    # 2. Identify Non-Unique Variants (Clusters with > 1 sequence)
    # We need a sibling to exist in the gallery for the mapping to work
    variant_counts = full_gallery["variant_id"].value_counts()
    multi_member_variants = variant_counts[variant_counts > 1].index.tolist()

    if not multi_member_variants:
        raise ValueError(
            "Shard contains only single-sequence clusters. Cannot perform proximal test."
        )

    # 3. Split the Data
    # We take ONE sequence from each multi-member variant to be our 'Inference' set
    # The REST stays in the gallery.
    inference_rows = []
    reduced_gallery_rows = []

    for vid, group in full_gallery.groupby("variant_id"):
        if vid in multi_member_variants:
            # Pop the first one for inference
            inference_rows.append(group.iloc[0])
            # Keep the others for the gallery
            reduced_gallery_rows.append(group.iloc[1:])
        else:
            # Single member clusters just stay in the gallery
            reduced_gallery_rows.append(group)

    test_queries = pd.DataFrame(inference_rows)
    test_gallery = pd.concat(reduced_gallery_rows, ignore_index=True)

    # 4. Create Mock Inference FASTA
    mock_fasta = temp_dir / f"{task_id}_queries.fasta"
    records = []
    for row in test_queries.itertuples():
        # Use our improved stripper from the previous step
        clean_seq = row.sequence_aligned.replace("-", "").replace(".", "").replace("_", "")
        records.append(SeqRecord(Seq(clean_seq), id=row.organism_id, description=""))

    SeqIO.write(records, mock_fasta, "fasta")

    # 5. Run Inference against the REDUCED gallery
    # We save the reduced gallery to a temp parquet to simulate a real shard
    temp_shard = temp_dir / f"{task_id}_reduced_gallery.parquet"
    test_gallery.to_parquet(temp_shard)

    results_df = run_variant_inference(
        inference_fasta=mock_fasta,
        gallery_shard=temp_shard,
        hmm_path=hmm_path,
        temp_dir=temp_dir,
        vocab_config=vocab_config,
        task_id=task_id,
        buffer_factor=1.5,  # 50% buffer
    )

    # 6. Evaluation
    comparison = results_df.merge(
        test_queries[["organism_id", "variant_id"]],
        on="organism_id",
        suffixes=("_inferred", "_original"),
    )

    comparison["is_correct"] = (
        comparison["variant_id_inferred"] == comparison["variant_id_original"]
    )
    accuracy = comparison["is_correct"].mean() * 100

    print(f"\n--- Proximal Stress Test: {hmm_path.stem} ---")
    print(f"Sequences Popped & Tested: {len(test_queries)}")
    print(f"Gallery Size (after removal): {len(test_gallery)}")
    print(f"Success Rate: {accuracy:.2f}%")

    if accuracy < 100:
        failures = comparison[comparison["variant_id_inferred"] == "UNK"]
        mismatches = comparison[
            (comparison["variant_id_inferred"] != "UNK") & (~comparison["is_correct"])
        ]
        print(f"Assigned 'UNK': {len(failures)}")
        print(f"Misassigned to wrong Variant: {len(mismatches)}")

    return comparison


if __name__ == "__main__":

    hmm_name = "Wzy_C"
    gallery_example = f"/Users/tamsen/Code/databases/inference/default_run_ep24_20260325/reference/bac/variant_shards/clusters_16_reduction_false_matrix_blosum62/{hmm_name}.parquet"
    hmm_path = f"/Users/tamsen/Code/databases/hmm/pfam/v38.0/split/{hmm_name}.hmm"
    my_vocab_obj = get_vocab()

    stress_test_proximal_mapping(
        gallery_shard_path=Path(gallery_example),
        hmm_path=Path(hmm_path),
        vocab_config=my_vocab_obj,
        temp_dir=Path("/tmp/stress_test"),
    )

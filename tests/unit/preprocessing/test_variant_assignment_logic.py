import numpy as np
import pytest
from variants.cluster_map import rigorous_assignment


def test_rigorous_assignment_consensus():
    """Matches Logic 2: Top matches reach consensus (2 out of 3)."""
    distances = np.array([[0.1, 0.12, 0.15, 0.8]])
    gallery_ids = np.array(["V1", "V1", "V2", "V3"])
    # V1 has 2 votes in the top 3

    ids, scores = rigorous_assignment(distances, gallery_ids)

    assert ids == ["V1"]
    assert scores == [0.1]


def test_rigorous_assignment_no_consensus_within_threshold():
    """Final Decision fallback: No consensus (1-1-1), but best match is valid."""
    distances = np.array([[0.1, 0.2, 0.3]])
    gallery_ids = np.array(["V1", "V2", "V3"])
    # All unique in top 3, but 0.1 < 0.5 (fallback threshold)

    ids, scores = rigorous_assignment(distances, gallery_ids)

    assert ids == ["V1"]
    assert scores == [0.1]


def test_rigorous_assignment_explicit_threshold_violation():
    """Logic 1: Distance exceeds the provided cluster_thresholds."""
    distances = np.array([[0.25]])
    gallery_ids = np.array(["V1"])
    thresholds = {"V1": 0.2}  # 0.25 > 0.2

    ids, scores = rigorous_assignment(distances, gallery_ids, cluster_thresholds=thresholds)

    assert ids == ["UNKNOWN_NEW_VARIANT"]
    assert scores == [0.25]


def test_rigorous_assignment_fallback_threshold_violation():
    """Logic 1 Fallback: Distance > 0.5 when no threshold dict is provided."""
    distances = np.array([[0.51, 0.6, 0.7]])
    gallery_ids = np.array(["V1", "V2", "V3"])

    ids, scores = rigorous_assignment(distances, gallery_ids)

    assert ids == ["UNKNOWN_NEW_VARIANT"]
    assert scores == [0.51]


def test_rigorous_assignment_small_gallery():
    """Ensures K is capped by the number of available gallery sequences."""
    # Gallery only has 2 sequences, K becomes 2
    distances = np.array([[0.05, 0.06]])
    gallery_ids = np.array(["V1", "V1"])

    ids, scores = rigorous_assignment(distances, gallery_ids)

    assert ids == ["V1"]
    assert len(ids) == 1
    assert scores == [0.05]


def test_rigorous_assignment_multi_query():
    """Verifies loop handles multiple queries correctly."""
    distances = np.array(
        [
            [0.1, 0.1, 0.1, 0.9],  # Query 0: Strong consensus
            [0.9, 0.9, 0.9, 0.9],  # Query 1: Unknown
        ]
    )
    gallery_ids = np.array(["A", "A", "B", "C"])

    ids, scores = rigorous_assignment(distances, gallery_ids)

    assert ids[0] == "A"
    assert ids[1] == "UNKNOWN_NEW_VARIANT"
    assert len(ids) == 2


def test_rigorous_assignment_threshold_present_but_variant_missing():
    """Logic 1: Threshold dict exists, but the best_variant isn't in it (falls to elif)."""
    distances = np.array([[0.1]])
    gallery_ids = np.array(["V_NEW"])
    thresholds = {"V_OLD": 0.1}  # V_NEW not in dict

    ids, scores = rigorous_assignment(distances, gallery_ids, cluster_thresholds=thresholds)

    # Should fall through to 'elif best_dist > 0.5' check
    assert ids == ["V_NEW"]


def test_rigorous_assignment_consensus_overridden_by_quality():
    """Tests that a 1.2 ratio prevents a distant consensus from winning."""
    # Best dist is 0.05 (V1). Consensus is V2 (dist 0.15).
    # Ratio is 3.0 (0.15 / 0.05), which is > 1.2. V1 should win.
    distances = np.array([[0.05, 0.15, 0.16]])
    gallery_ids = np.array(["V1", "V2", "V2"])

    ids, _ = rigorous_assignment(distances, gallery_ids, consensus_ratio=1.2)
    assert ids == ["V1"]


def test_rigorous_assignment_consensus_accepted_within_ratio():
    """Tests that consensus wins if it is within the allowed ratio."""
    # Best dist is 0.10 (V1). Consensus is V2 (dist 0.11).
    # Ratio is 1.1, which is < 1.2. V2 (Consensus) should win.
    distances = np.array([[0.10, 0.11, 0.12]])
    gallery_ids = np.array(["V1", "V2", "V2"])

    ids, _ = rigorous_assignment(distances, gallery_ids, consensus_ratio=1.2)
    assert ids == ["V2"]

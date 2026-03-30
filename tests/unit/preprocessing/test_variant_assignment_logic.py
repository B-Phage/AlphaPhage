import numpy as np
import pytest
from variants.cluster_map import rigorous_assignment

# Corrected to match implementation
UNKNOWN_NEW_VARIANT = "UNK"


def test_rigorous_assignment_consensus_logic():
    """Verify consensus wins when within ratio."""
    # V1 is 0.1, V2 is 0.11 (Consensus). 0.11 < (0.1 * 1.2)
    distances = np.array([[0.1, 0.11, 0.11]])
    gallery_ids = np.array(["V1", "V2", "V2"])
    ids, _ = rigorous_assignment(distances, gallery_ids, consensus_ratio=1.2)
    assert ids == ["V2"]


def test_rigorous_assignment_outlier_wins():
    """Verify 1-NN wins when consensus is too far away."""
    # V1 is 0.1, V2 is 0.2 (Consensus). 0.2 > (0.1 * 1.2)
    distances = np.array([[0.1, 0.2, 0.2]])
    gallery_ids = np.array(["V1", "V2", "V2"])
    ids, _ = rigorous_assignment(distances, gallery_ids, consensus_ratio=1.2)
    assert ids == ["V1"]


def test_rigorous_assignment_fallback_radius():
    """Test the 0.5 fallback when no thresholds provided."""
    distances = np.array([[0.51]])
    gallery_ids = np.array(["V1"])
    ids, _ = rigorous_assignment(distances, gallery_ids, fallback_radius=0.5)
    assert ids == [UNKNOWN_NEW_VARIANT]


def test_rigorous_assignment_exact_match_wins():
    """Ensures exact matches (dist 0) don't get swayed by distant consensus."""
    distances = np.array([[0.0, 0.05, 0.05]])
    gallery_ids = np.array(["V1", "V2", "V2"])
    ids, _ = rigorous_assignment(distances, gallery_ids)
    assert ids == ["V1"]


def test_rigorous_assignment_k_limit():
    """Test behavior when gallery is smaller than K (3)."""
    distances = np.array([[0.1]])
    gallery_ids = np.array(["V1"])
    # Should not crash, should just use K=1
    ids, _ = rigorous_assignment(distances, gallery_ids)
    assert ids == ["V1"]

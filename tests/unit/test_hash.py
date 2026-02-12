import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf
from utils import hash as hash_utils

# ------------------------------
# hash_file
# ------------------------------


def test_hash_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello world")
    h = hash_utils.hash_file(f)
    assert h == hash_utils.hash_string("hello world")


def test_hash_file_large(tmp_path):
    f = tmp_path / "large.txt"
    content = "abc" * 100000  # >64KB
    f.write_text(content)
    assert hash_utils.hash_file(f) == hash_utils.hash_string(content)


# ------------------------------
# file_manifest
# ------------------------------


def test_file_manifest_deterministic(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("A")
    b = tmp_path / "b.txt"
    b.write_text("B")
    files = [b, a]  # unsorted
    manifest = hash_utils.file_manifest(files, relative_to=tmp_path)
    keys = list(manifest.keys())
    assert keys == ["a.txt", "b.txt"]  # sorted
    # hash matches file content
    assert manifest["a.txt"] == hash_utils.hash_file(a)
    assert manifest["b.txt"] == hash_utils.hash_file(b)


def test_file_manifest_relative_vs_absolute(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("X")
    manifest_abs = hash_utils.file_manifest([f])
    manifest_rel = hash_utils.file_manifest([f], relative_to=tmp_path)
    assert list(manifest_rel.keys())[0] == "x.txt"
    assert list(manifest_abs.keys())[0].endswith("x.txt")


# ------------------------------
# manifest_from_directory & hash_directory
# ------------------------------


def test_manifest_from_directory_and_hash(tmp_path):
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")

    # Use relative_to=tmp_path to get keys like 'a.txt'
    manifest = hash_utils.manifest_from_directory(tmp_path, relative_to=tmp_path)
    assert "a.txt" in manifest
    assert "b.txt" in manifest
    assert manifest["a.txt"] == hash_utils.hash_file(tmp_path / "a.txt")
    assert manifest["b.txt"] == hash_utils.hash_file(tmp_path / "b.txt")


def _expected_lake_hash(path: Path) -> str:
    manifest = hash_utils.manifest_from_directory(path, "*.parquet")
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def test_hash_lake_single_file(tmp_path: Path):
    # Single parquet file
    df = pd.DataFrame({"col": [1, 2, 3]})
    file_path = tmp_path / "file.parquet"
    df.to_parquet(file_path)

    # Build expected manifest and hash
    manifest = hash_utils.hash_lake(tmp_path)
    expected = _expected_lake_hash(tmp_path)

    assert manifest == expected


def test_hash_directory_pattern(tmp_path):
    (tmp_path / "x.py").write_text("x")
    (tmp_path / "y.txt").write_text("y")
    digest = hash_utils.hash_directory(tmp_path, "*.py")
    assert digest == hash_utils.hash_directory(tmp_path, "*.py")  # deterministic
    digest_txt = hash_utils.hash_directory(tmp_path, "*.txt")
    assert digest != digest_txt


def test_hash_lake_multiple_files_order(tmp_path: Path):
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})

    # reversed names to test sorting
    f1 = tmp_path / "b.parquet"
    f2 = tmp_path / "a.parquet"

    df1.to_parquet(f1)
    df2.to_parquet(f2)

    expected = _expected_lake_hash(tmp_path)

    assert hash_utils.hash_lake(tmp_path) == expected


def test_hash_lake_empty_dir(tmp_path: Path):
    expected = _expected_lake_hash(tmp_path)
    assert hash_utils.hash_lake(tmp_path) == expected


# ------------------------------
# hash_string
# ------------------------------


def test_hash_string():
    s = "abc"
    h = hash_utils.hash_string(s)
    import hashlib

    assert h == hashlib.sha256(s.encode()).hexdigest()


# ------------------------------
# verify_checksum
# ------------------------------


def test_verify_checksum_pass(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("ok")
    hash_val = hash_utils.hash_file(f)
    hash_utils.verify_checksum(f, hash_val)  # should not raise


def test_verify_checksum_fail(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("ok")
    with pytest.raises(ValueError):
        hash_utils.verify_checksum(f, "deadbeef")


# ------------------------------
# create_manifest
# ------------------------------


def test_create_manifest(tmp_path):
    f1 = tmp_path / "f1.txt"
    f1.write_text("foo")
    f2 = tmp_path / "f2.txt"
    f2.write_text("bar")
    out = tmp_path / "manifest.json"
    hash_utils.create_manifest([f1, f2], out, relative_to=tmp_path)
    data = json.loads(out.read_text())
    assert "f1.txt" in data and "f2.txt" in data
    assert data["f1.txt"] == hash_utils.hash_file(f1)


# ------------------------------
# hash_package
# ------------------------------


def test_hash_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f1 = pkg / "a.py"
    f1.write_text("print(1)")
    f2 = pkg / "b.py"
    f2.write_text("print(2)")
    digest1 = hash_utils.hash_package(pkg)
    digest2 = hash_utils.hash_package(pkg)
    assert digest1 == digest2


# ------------------------------
# compute_combined_code_version
# ------------------------------


def test_compute_combined_code_version(tmp_path):
    f1 = tmp_path / "hash1.txt"
    f1.write_text("h1")
    f2 = tmp_path / "hash2.txt"
    f2.write_text("h2")
    base_version = "v1.0"

    # simple config item
    cfg = {"param": 42}

    code_version = hash_utils.compute_combined_code_version(
        base_version=base_version, hash_files=[f1, f2], config_items=[cfg]
    )
    assert isinstance(code_version, str)
    # deterministic: calling again produces same hash
    code_version2 = hash_utils.compute_combined_code_version(
        base_version=base_version, hash_files=[f2, f1], config_items=[cfg]  # unsorted input
    )
    assert code_version == code_version2


def test_compute_combined_code_version_with_path(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("x: 1")
    code_version = hash_utils.compute_combined_code_version(
        base_version="v1", hash_files=[cfg_path], config_items=[cfg_path]
    )
    assert isinstance(code_version, str)


# ------------------------------
# _normalize
# ------------------------------


def test_normalize_paths_and_dicts(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    d = {"p": f, "val": 123}
    normalized = hash_utils._normalize(d)
    assert normalized["p"] == str(f.resolve())
    assert normalized["val"] == 123


def test_normalize_lists_sets():
    items = [3, 1, 2]
    normalized = hash_utils._normalize(items)
    assert normalized == [1, 2, 3]

    s = {3, 2, 1}
    normalized_s = hash_utils._normalize(s)
    assert normalized_s == [1, 2, 3]


# ------------------------------
# Edge cases
# ------------------------------


def test_file_manifest_skips_non_files(tmp_path: Path):
    real_file = tmp_path / "real.txt"
    real_file.write_text("data")

    subdir = tmp_path / "subdir"
    subdir.mkdir()

    missing = tmp_path / "missing.txt"  # does not exist

    manifest = hash_utils.file_manifest([real_file, subdir, missing], relative_to=tmp_path)

    # Only the real file should be present
    assert list(manifest.keys()) == ["real.txt"]


def test_normalize_omegaconf_section():
    cfg = OmegaConf.create(
        {
            "db": {"path": "/data/db", "table": "summary"},
            "list": [3, 1, 2],
        }
    )

    normalized = hash_utils._normalize(cfg)

    # Should now be a plain dict (not OmegaConf)
    assert isinstance(normalized, dict)
    assert normalized["db"]["path"] == "/data/db"
    assert normalized["db"]["table"] == "summary"

    # List should be sorted by string value due to normalization rules
    assert normalized["list"] == [1, 2, 3]

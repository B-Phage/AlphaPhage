import shutil
from pathlib import Path

import pytest
from utils.file_system import clear_dir


def test_creates_directory(tmp_path):
    dir_path = tmp_path / "new_dir"
    assert not dir_path.exists()

    clear_dir(dir_path)

    assert dir_path.exists()
    assert dir_path.is_dir()
    assert list(dir_path.iterdir()) == []


def test_clears_files(tmp_path):
    dir_path = tmp_path / "dir_with_files"
    dir_path.mkdir()
    # create files
    file1 = dir_path / "file1.txt"
    file2 = dir_path / "file2.txt"
    file1.write_text("hello")
    file2.write_text("world")

    clear_dir(dir_path)

    assert dir_path.exists()
    assert list(dir_path.iterdir()) == []


def test_clears_subdirectories(tmp_path):
    dir_path = tmp_path / "dir_with_subdirs"
    subdir1 = dir_path / "sub1"
    subdir2 = dir_path / "sub2"
    subdir1.mkdir(parents=True)
    subdir2.mkdir(parents=True)
    # add files in subdirs
    (subdir1 / "f.txt").write_text("x")
    (subdir2 / "g.txt").write_text("y")

    clear_dir(dir_path)

    assert dir_path.exists()
    assert list(dir_path.iterdir()) == []


def test_clears_mixed_files_and_dirs(tmp_path):
    dir_path = tmp_path / "mixed"
    dir_path.mkdir()
    (dir_path / "file.txt").write_text("abc")
    subdir = dir_path / "subdir"
    subdir.mkdir()
    (subdir / "subfile.txt").write_text("xyz")

    clear_dir(dir_path)

    assert dir_path.exists()
    assert list(dir_path.iterdir()) == []


def test_clears_empty_directory(tmp_path):
    dir_path = tmp_path / "empty_dir"
    dir_path.mkdir()

    clear_dir(dir_path)

    assert dir_path.exists()
    assert list(dir_path.iterdir()) == []


def test_clears_hidden_files(tmp_path):
    dir_path = tmp_path / "hidden_files"
    dir_path.mkdir()
    (dir_path / ".hidden").write_text("secret")

    clear_dir(dir_path)

    assert dir_path.exists()
    assert list(dir_path.iterdir()) == []


def test_path_is_path_object(tmp_path):
    """Ensure the function works if a Path object is passed."""
    dir_path = tmp_path / "some_dir"
    clear_dir(dir_path)  # dir_path is already Path
    assert dir_path.exists()
    assert list(dir_path.iterdir()) == []

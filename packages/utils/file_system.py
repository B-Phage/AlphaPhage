# Shared utils for DVC workflows
import shutil
from pathlib import Path


def clear_dir(dir_path: Path):
    """Empty the contents of a directory, creating it if missing."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

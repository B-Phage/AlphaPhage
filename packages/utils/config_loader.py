from pathlib import Path
from typing import Final

from omegaconf import OmegaConf

PARAMS_PATH: Final[Path] = Path("params.yaml")


def get_prodigal_config(override_path: str | None = None):
    """
    Loads prodigal config from params.yaml or override file
    """
    cfg_path = Path(override_path) if override_path else PARAMS_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    params = OmegaConf.load(cfg_path)
    if "prodigal_config" not in params:
        raise KeyError("prodigal_config section missing in config file")

    return params.prodigal_config

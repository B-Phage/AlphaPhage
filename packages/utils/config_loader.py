"""
This utility is dependent on DVC's params.yml pipeline file where the active vocab is defined.
If you wish to change the active vocab, change it there.

Example usage for this utility:

from packages.utils.config_loader import get_vocab
vocab = get_vocab()
vocab1 = vocab.vocab1
"""

from pathlib import Path
from typing import Final, Optional, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

PARAMS_PATH: Final[Path] = Path("params.yaml")


def _load_params():
    return OmegaConf.load(PARAMS_PATH)


def _load_vocab_file(str_path: str):
    path = Path(str_path)
    if not path.exists():
        raise FileNotFoundError(path)

    return OmegaConf.load(path)


# def get_vocab(test_mode: bool = False, override_path: str | None = None):  #py3.10+ only
def get_vocab(test_mode: bool = False, override_path: Optional[str] = None):
    """
    Loads the vocabulary file referenced in params.yaml
    and returns all vocabularies defined inside that file.

    Supports multi-vocab YAML files (returns nested OmegaConf dict).
    """
    if override_path:
        return _load_vocab_file(override_path)

    params = _load_params()
    vocab_cfg = params.vocab

    if test_mode:
        if "test_file" not in vocab_cfg:
            raise ValueError("params.yaml missing vocab.test_file")
        return _load_vocab_file(vocab_cfg.test_file)

    return _load_vocab_file(vocab_cfg.file)


# def get_config(override_path: str | None = None, params_set: str = "prodigal_config"): #py3.10+ only
def get_config(override_path: Optional[str] = None, params_set: str = "prodigal_config"):
    """
    Loads prodigal config from params.yaml or override file
    """
    cfg_path = Path(override_path) if override_path else PARAMS_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    params = OmegaConf.load(cfg_path)
    if params_set not in params:
        raise KeyError(f"{params_set} section missing in config file {params}")

    return params[params_set]  # type: ignore[reportArgumentType]


# def get_config(override_path: str | None = None, params_set: str = "prodigal_config"): #py3.10+ only
def get_config_generic(
    path: str,
    params_set: Union[str, list, None] = None,
    params_set_2: Union[str, list, None] = None,
):
    """
    Loads prodigal config from params.yaml or override file with the specific config file.
    path: The path of the config file to load.
    params_set: If None, return the entire config. If str, return the section of
        the config corresponding to that key. If list of str, traverse the config
        using the keys in order and return the final section.
    params_set_2: If provided, this is an additional set of keys to traverse
        after the first params_set. This allows for a two-level traversal of the
        config, which can be useful if the config file has a nested structure where
        the desired config section is not directly under the top level.

    """
    cfg_path = Path(path)
    params = OmegaConf.load(cfg_path)
    # 1. Narrow the type: Ensure it's a DictConfig, not a ListConfig
    if isinstance(params, ListConfig):
        raise TypeError(f"Config at {cfg_path} is a YAML list, but a dictionary was expected.")
    if params_set is None:
        # assuming params_set is None, return the entire config
        params1 = params
    elif isinstance(params_set, list):
        # assuming params_set is a list of keys to traverse in order to get to the desired config section
        current_section = params
        for key in params_set:
            if key not in current_section:
                raise KeyError(f"{key} section missing in config file {cfg_path}")
            current_section = current_section[key]
        params1 = current_section

    elif params_set is not None:
        # assuming params_set is a single key corresponding to the desired config section
        if params_set not in params:
            raise KeyError(f"{params_set} section missing in config file {params}")
        params1 = params[params_set]

    # params1 may be a dict or a str, like a path. If it's a path, we want to load
    # a second config from that path and then apply params_set_2 if provided. If
    # it's already a dict, we can just apply params_set_2 if provided.

    if not params_set_2:
        return params1

    else:
        # params1 should be a path
        if isinstance(params1, str):
            cfg_path2 = Path(params1)
            params2 = OmegaConf.load(cfg_path2)

            if isinstance(params2, ListConfig):
                raise TypeError(
                    f"Config at {cfg_path2} is a YAML list, but a dictionary was expected."
                )

            if isinstance(params_set_2, list):
                current_section = params2
                for key in params_set_2:
                    if key not in current_section:
                        raise KeyError(f"{key} section missing in config file {cfg_path2}")
                    current_section = current_section[key]
                params2 = current_section

            elif isinstance(params_set_2, str):
                if params_set_2 not in params2:
                    raise KeyError(f"{params_set_2} section missing in config file {cfg_path2}")

                params2 = params2[params_set_2]

            return params2
        else:
            # Just return the value
            raise TypeError(
                f"params_set_2 was given, implying extending into another config, but \
the type is not str."
            )

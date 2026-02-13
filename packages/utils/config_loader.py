# packages/utils/config_loader.py
"""
This utility is dependent on DVC's params.yml pipleine file where the active vocab is defined.
If you wish to change the active vocab, change it there.

Example usage for this utility:

from packages.utils.config_loader import get_vocab
vocab = get_vocab()
vocab1 = vocab.vocab1
"""

from pathlib import Path
from typing import Final

from omegaconf import OmegaConf

PARAMS_PATH: Final[Path] = Path("params.yaml")


def _load_params():
    return OmegaConf.load(PARAMS_PATH)


def _load_vocab_file(str_path: str):
    path = Path(str_path)
    if not path.exists():
        raise FileNotFoundError(path)

    return OmegaConf.load(path)


def get_vocab(test_mode: bool = False, override_path: str | None = None):
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


def get_prodigal_config(override_path: str | None = None):
    """
    Loads prodigal config from params.yaml or override file
    """
    cfg_path = Path(override_path) if override_path else PARAMS_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    params = OmegaConf.load(cfg_path)
    if "prodigal_config" not in params:
        raise KeyError(f"prodigal_config section missing in config file {params}")

    return params.prodigal_config


def get_kapsid_config(override_path: str | None = None):
    """
    Loads kapsid config from params.yaml or override file
    """
    cfg_par_path = Path(override_path) if override_path else PARAMS_PATH
    if not cfg_par_path.exists():
        raise FileNotFoundError(cfg_par_path)

    params = OmegaConf.load(cfg_par_path)
    if "kapsid" not in params:
        raise KeyError(f"'kapsid' section missing in config file {params}")

    cfg_path = params.kapsid.kapsid_config

    return cfg_path, OmegaConf.load(cfg_path)


# TODO: Test this functionality with the transformer archtecture
# We might want to have the vocab as a dict rather than a DictConfig
# if we want to load a vocab in as a dict
# this would then be a swap out for _load_vocab_file
# def _load_vocab_file_as_dict(str_path: str):
#     path = Path(str_path)
#     if not path.exists():
#         raise FileNotFoundError(path)
#     dict_config_vocab = OmegaConf.load(path)
#     # convert to a standard Python dict
#     converted_vocab = {}
#     # loop through the top-level keys
#     for key, value in dict_config_vocab.items():
#         # Convert the nested DictConfig/ListConfig to a standard Python container.
#         # resolve=True ensures any string interpolations are processed.
#         converted_value = OmegaConf.to_container(
#             value,
#             resolve=True,
#             # structured_config=True is the default, ensuring conversion
#         )
#         # Store the converted standard dict back into the new dictionary
#         converted_vocab[key] = converted_value
#     return converted_vocab

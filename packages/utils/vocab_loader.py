# packages/utils/vocab_loader.py
"""
This utility is dependent on DVC's params.yml pipleine file where the active vocab is defined.
If you wish to change the active vocab, change it there.

Example usage for this utility:

from packages.utils.vocab_loader import get_vocab
vocab = get_vocab()
vocab1 = vocab.vocab1
"""

from functools import lru_cache
from pathlib import Path
from typing import Final

from omegaconf import OmegaConf

PARAMS_PATH: Final[Path] = Path("params.yaml")


# @lru_cache(maxsize=2) # add this if we want cashing
def _load_params():
    return OmegaConf.load(PARAMS_PATH)


# @lru_cache(maxsize=1)
def _load_vocab_file(str_path: str):
    path = Path(str_path)
    if not path.exists():
        raise FileNotFoundError(path)

    return OmegaConf.load(path)


# if we want to load a vocab in as a dict
def _load_vocab_file_as_dict(str_path: str):
    path = Path(str_path)
    if not path.exists():
        raise FileNotFoundError(path)
    dict_config_vocab = OmegaConf.load(path)
    # convert to a standard Python dict
    converted_vocab = {}
    # loop through the top-level keys
    for key, value in dict_config_vocab.items():
        # Convert the nested DictConfig/ListConfig to a standard Python container.
        # resolve=True ensures any string interpolations are processed.
        converted_value = OmegaConf.to_container(
            value,
            resolve=True,
            # structured_config=True is the default, ensuring conversion
        )
        # Store the converted standard dict back into the new dictionary
        converted_vocab[key] = converted_value
    return converted_vocab


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

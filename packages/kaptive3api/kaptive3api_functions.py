import json
import os
import sys
from pathlib import Path
from typing import Any

from kaptive.assembly import typing_pipeline
from kaptive.database import load_database

# from omegaconf import DictConfig, OmegaConf


# -----------------------------
# Output function
# -----------------------------
def kapsid_output_func(input_path: Path, output_dir: Path):
    """Provides the expected output file path(s) for a given input file. The
    Orchestrator uses this to determine where to look for outputs and whether
    they already exist.
    """
    genome_name = Path(input_path).stem
    kaptive_file = os.path.join(output_dir, f"{genome_name}.json")

    return [Path(kaptive_file)]


def run_kapsid(input_path: Path, outputs: list, db_obj: Any):
    """Implements the kapsid.typing_pipeline. The Orchestrator expects a function
    that takes an input path and a list of output paths, and returns the list of
    output paths after processing. The db_obj is passed in via
    Note: The Orchestrator worker requires two arguments - input_path and
    outputs. The db_obj is passed in via functools.partial in main to create a
    bound version of this function that only requires input_path and outputs.
    """
    kaptive_file = outputs[0]
    result = typing_pipeline(input_path, db_obj)
    d = {}
    if result:
        # Request the JSON string
        json_output = result.format("json")
        if isinstance(json_output, str):
            dresult = json.loads(json_output)
        else:
            # Handle cases where format() didn't return a string as expected
            raise TypeError(f"Expected JSON string from format('json'), got {type(json_output)}")

        for key in [
            "sample_name",
            "best_match",
            "confidence",
            "phenotype",
            "problems",
            "percent_identity",
            "percent_coverage",
        ]:
            d.update({key: dresult[key]})

    with open(kaptive_file, "w") as h:
        json.dump(d, h)

    return result

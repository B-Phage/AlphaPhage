#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf


def create_inference_config(
    training_cfg: DictConfig, output_path: str, base_inf_dir: str, buffer_factor: float = 1.1
) -> DictConfig:
    """
    Exports an inference config with injected mapping parameters from a config object.

    Args:
        training_cfg: The OmegaConf DictConfig object (already loaded).
        output_path: Path where the generated inference YAML will be saved.
        buffer_factor: Buffer for the rigorous mapping decision boundary.

    Returns:
        DictConfig: The finalized inference configuration.
    """
    # 1. Deep copy to preserve original settings and resolve interpolations
    inf_dict = OmegaConf.to_container(training_cfg, resolve=True)
    inf = OmegaConf.create(inf_dict)

    # 2. Update General Metadata
    domain_name = inf.general.get("name", "unknown_domain")
    inf.general.name = f"{domain_name}_inference"
    inf.self_path = str(output_path)

    # 3. Resolve the Gallery Path
    # Logic: It uses the msa_variant_dir and msa_cluster_name defined in the config.
    # Note: In the snapshot script, you should override msa_variant_dir to the
    # relative path 'reference/{domain}/variant_shards' before calling this.
    gallery_path = os.path.join(inf.msa_var.msa_variant_dir, inf.msa_var.msa_cluster_name)

    # 5. Inject the new inference-specific block
    inf.inference_config = {
        "gallery_shard_dir": str(gallery_path),
        "mapped_output_dir": f"{base_inf_dir}/{domain_name}/variant_mapping",
        "query_msa_dir": f"{base_inf_dir}/{domain_name}/msa_csa",
        "buffer_factor": buffer_factor,
    }

    # 6. Ensure directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    OmegaConf.save(config=inf, f=output_path)

    return inf


def main():
    parser = argparse.ArgumentParser(
        description="Convert a training/preprocessing YAML config into an inference config."
    )
    parser.add_argument(
        "-t",
        "--training_config",
        type=str,
        required=True,
        help="Path to the original training/preprocessing YAML.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path where the generated inference YAML will be saved.",
    )
    parser.add_argument(
        "-i_dir", "--inf_dir", type=str, required=True, help="Inference processing root path."
    )
    parser.add_argument(
        "-b",
        "--buffer",
        type=float,
        default=1.1,
        help="Buffer factor for the rigorous mapping decision boundary (default: 1.1).",
    )

    args = parser.parse_args()

    try:
        # Load the config from path to pass to the object-based function
        raw_cfg = OmegaConf.load(args.training_config)

        if not isinstance(raw_cfg, DictConfig):
            raise TypeError(
                f"Config at {args.training_config} must be a Mapping (dict), not a List."
            )

        # 3. Explicitly cast for the function call
        loaded_cfg = cast(DictConfig, raw_cfg)

        create_inference_config(
            training_cfg=loaded_cfg,
            output_path=args.output,
            base_inf_dir=args.inf_dir,
            buffer_factor=args.buffer,
        )
        print(f"✅ Success: Inference config written to {args.output}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()

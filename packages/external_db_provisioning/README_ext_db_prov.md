# External DB Provisioning
This package manages the lifecycle, integrity, and post-processing of external biological databases and model files required for protein annotation. It ensures that all remote assets are downloaded, verified via SHA256 checksums, and correctly formatted for the pipeline.

## System Architecture
The package acts as a bridge between remote storage (source URLs) and the local filesystem. It uses a core provisioner logic to handle idempotent downloads, ensuring assets are only re-downloaded if the remote source or checksum has changed.
## Supported Data Types

1. *HMM Data (`hmm_data_setup.py`)*
   Used specifically for Profile HMMs. This setup includes a mandatory post-processing step:
   *Splitting.*
   - *Logic:* Large multi-HMM files are downloaded and then parsed into individual `.hmm` files.
   - *Trigger:* Splitting occurs if the asset is newly updated or if the `split/` directory is missing.
   - *Config Flag:* Requires `can_split: True` in the YAML manifest.

2. *Kaptive/Reference Data (`kaptive_data_setup.py`)*
   Used for static reference databases (e.g., Kaptive).
   - *Logic:* Simple "Download and Verify" flow.
   - *Format:* Supports raw files or compressed archives (`.tar.gz`, `.gz`) via the `source_format` and `inner_path` configuration keys.

## Configuration & Manifests
Assets are defined in YAML manifests located in `ml_assets/config/`. These files act as the "source of truth" for database versions.
```
Database,Primary Config Path,Makefile Target
HMM Profiles,ml_assets/config/hmm_data_v0.0.1.setup.yaml,make setup-hmm-data
Kaptive DB,ml_assets/config/kaptive3api_v0.0.1.setup.yaml,make setup-kaptive-data
```

## Manifest Schema
Each entry in a YAML config should contain:
- `name:` The local filename for the asset.
- `dest_dir`: Where to store the file (usually inside a data lake).
- `source_url`: The remote download link.
- `final_sha256`: The expected hash of the file after extraction/download.
- `source_sha256` (Optional): The hash of the compressed archive.
- `source_format` (Optional): `raw` (default) or `gz.`

## Update Procedure (Destructive Changes)
Updating a version in the YAML config is a breaking change. If you bump a version, you must:
- *Generate Checksums:* Run `sha256sum` on the new files locally before updating the YAML.
- *Validate Integrity:* Run the corresponding make target to ensure the download and (if applicable) splitting finish without error.
- *Downstream Testing:*
  - *HMMs:* Require a complete retrain of downstream variant models.
  - *Kaptive:* Requires regression testing on the annotation outputs.
- *Version Bumping:* Rename the config file (e.g., `v0.0.1` -> `v1.0.0`) and update the `Makefile` variables.

## Usage
While these scripts can be run via Python, it is recommended to use the `Makefile` interface to ensure the correct `conda` environment and `PYTHONPATH` are used:
```
Bash
# Provision HMMs and perform splitting
make setup-hmm-data

# Provision Kaptive reference databases
make setup-kaptive-data
```

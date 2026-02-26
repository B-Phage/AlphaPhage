# kaptive3api

This package wraps the kaptive3 tool for characterising the kapsid of bacteria.
The kaptive code base is found here https://github.com/klebgenomics/Kaptive/tree/master.
The kaptive toobase was originally built for Klebsiella, but can be adapted for
other gram negative bacteria. Some groups have published other species-specific
database files that can be used by kapsid as replacement for the default Klebsiella
database, for example, that for Escherichia coli -
https://github.com/rgladstone/EC-K-typing/tree/main/DB
We have found that the kaptive outputs are an important part of the feature set
of the host for host-phage match prediction on E.coli and its phages.

The current database versions used by our pipelines are explicitly defined by the config here:
[ml_assets/config/kaptive3api_v0.0.1.yaml]
If for some reason the source database for E.coli is corrupted or no longer
available, the HMMs are also stored in our (Google Drive Folder)
[https://drive.google.com/drive/folders/1LL0nKxyBUrwVcstdQ4GWzwNXO2WEpYRV]

If we updated the versions of the kaptive database, the whole training and inference pipeline
outputs would change. Therefore the following is required:
    1. Full end to end testing of
       a. Regression tests
       b. Integration tests
    2. Complete retrain of the model
       1. Including any DVC input updates
       2. A change in important feature names/set for the training and evaluation data
    3. Inference evaluation of critical test sets
       1. The new versions need to improve our models' performance
    4. Update the new kaptive database version/s to the config file
       a. Bump the major version of the file name (E.g. v1.0.0 -> v2.0.0)
       b. Update the file urls
       c. Update the file out path versioning
       d. Update the checksums to ensure correct downloads
           * At the terminal run `sha256sum ../databases/EC-K-typing/*.gbk`
           (the format is irrelevant for the checksum). Then store this in the
           *"source_sha256"* config item.
           * And again run `ha256sum ../databases/EC-K-typing/*.gbk`. Then store
           this in the *"final_sha256"* config item.
    5. Test that the Makefile setup-hmm-data option still functions
    6. The tag version of the main branch will also need to be bumped

## Features
The kaptive tool uses Blast and currently requiring the python 2.9 and run in the
Intel emulation environment.

## Installation
```
make setup-kaptive3api
```
Note that in the Make file, the installation of the environment using
make setup-kaptive3api induces the Intel emulation (--subdir osx-64),
```
conda env create --subdir osx-64 -f conda_envs/$1.yml -n $1;
```
and fixing the environment as the emulation environment by subsequently running
```
conda config --env --set subdir osx-64.
```
The latter step is required to ensure that any future ```conda install``` commands
recognises the Intel environment.

## Config
Location: ml_assets/config/kaptive3api_vx.x.x.yaml
```
general:
  lst_genomes: []        # Optional whitelist of genomes e.g. ['370D', 'AL505', 'ECOR-09']
  code_version: "v0.0.1" # Code version.
  name: "kaptive3api"    # Name.
kaptive_config:
  genome_dir: "ml_assets/data/sequence_data/bacteria"       # Soure of inputs.
  kaptive_dir: "ml_assets/data/processed/bacteria/kaptive"  # Write outputs here.
  orgdb_full_path: "../databases/EC-K-typing/EC-K-typing_group2and3_v3.0.0.gbk"
    # Location of database (in genbank *.gbk format).
```

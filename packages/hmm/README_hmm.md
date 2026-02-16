# Hidden Markov Model (HMM) Package
This package is used in a preprocessing stage for protein annotation based on HMM matching.

This package requires the preprocessing conda enviornmnet to be active.

The integration of the package can be seen in workflows/cli/hmm_scan.py

# Data
The current HM Model verions used by our pipelines are explicitly defined by the config here:
[ml_assets/config/hmm_data_v0.0.1.yaml]
This is used by the Makefile to checkout the correct versions.

If for some reason the source HMM Data is corrupted or no longer available, the HMMs are also stored in our (Google Drive Folder)[https://drive.google.com/drive/folders/1LL0nKxyBUrwVcstdQ4GWzwNXO2WEpYRV]

If we update the versions of the HMM Data. The whole training and inference pipeline outputs will change.
Therefore the following is required:
    1. Full end to end testing of
       a. Regression tests
       b. Integration tests
    2. Complete retrain of the model
       1. Including any DVC input updates
       2. A change in important feature names/set for the training and evaluation data
    3. Inference evaluation of critical test sets
       1. The new versions need to improve our models' performance
    4. Update the new HMM data version/s to the config file
       a. Bump the major version of the file name (E.g. v1.0.0 -> v2.0.0)
       b. Update the file urls
       c. Update the file out path versioning
       d. Update the checksums to ensure correct downloads
           * At the terminal run `sha256sum /path/to/your/hmm_file.tar` the zip format is irrelevant for the checksum. Then store this in the *"source_sha256"* config item
           * And again run `sha256sum /path/to/your/hmm_file.hmm` . Then store this in the *"final_sha256"* config item.
    5. Test that the Makefile setup-hmm-data option still functions
    6. The tag version of the main branch will also need to be bumped

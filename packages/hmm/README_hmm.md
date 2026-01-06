# Hidden Markov Model (HMM) Package
This is a preprocessing stage for protein annotation based on HMM matching.

This package requires the hmmr_env conda enviornmnet to be active.

The integration of package can be seen in workflows/cli/hmm_scan.py

# Data
The HMMs are stored in our Google Drive Folder:

Originals where from the below sources which can be checked periodically for meaninful updates.
The HMM source for bacteria protein annotations is PFAM:
(PFAM Download)["https://www.ebi.ac.uk/interpro/download/Pfam/"]
We need
* Pfam-A models	hmm
* Pfam-A HMM data

The HMM source for phages are the PHROGS from Pharokka Database v1.4.0
(Zenodo)["https://zenodo.org/records/8267900]

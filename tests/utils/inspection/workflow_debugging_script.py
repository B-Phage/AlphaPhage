import subprocess
import sys

"""
Example script to debug into a workflow.

Assumes you have:
* Added breakpoints where you think things are going wrong
* Have activated the correct conda environment

"""
# command = [sys.executable, 'workflows/run_prodigal.py', # sys.executable is debuggable, whereas 'python' is not
#                    '--config', 'ml_assets/config/prodigal_ecoli_v0.0.1.yaml',
#                    '--force'
#                    ]

# result = subprocess.run(command, capture_output=True, text=True)
# print(result)
HMM_PHROG_NAME = "pharokka_v1.4.0_databases.tar.gz"
# HMM_PHROG_GDRIVE_URL = "https://drive.google.com/uc?id=1AtCwZ3ObxwQMav2t9ZoTyvQ-wv2GCNvg"
# https://drive.google.com/file/d/1AtCwZ3ObxwQMav2t9ZoTyvQ-wv2GCNvg/view?usp=drive_link
HMM_PHROG_URL = (
    "https://zenodo.org/records/8267900/files/pharokka_v1.4.0_databases.tar.gz?download=1"
)

HMM_PHROG_SHA256 = "0123456789abcdef..."
HMM_PHROG_DEST_DIR = "../databases/hmm/pharokka_v1.4.0_databases"
HMM_YAML = "ml_assets/config/hmm_data_v0.0.1.yaml"

# Setup workflow debugging
# command = [sys.executable, 'packages/setup_workflows/hmm_data_setup.py', # sys.executable is debuggable, whereas 'python' is not
#                    '--hmm-name', HMM_PHROG_NAME,
#                    '--url', HMM_PHROG_URL,
#                    '--dest-dir', HMM_PHROG_DEST_DIR,
#                    '--expected-sha256', HMM_PHROG_SHA256,
#                    ]

# setup_workflows.hmm_data_setup --yaml-registry $(HMM_YAML)
HMM_NAME = "pfam_annot"

command = [
    sys.executable,
    "packages/setup_workflows/hmm_data_setup.py",  # sys.executable is debuggable, whereas 'python' is not
    "--hmm-name",
    HMM_NAME,
    "--yaml-registry",
    HMM_YAML,
]

result = subprocess.run(command, capture_output=True, text=True)
print(result)

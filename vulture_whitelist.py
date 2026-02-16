"""
Vulture whitelist.

This file declares Python symbols that are invoked externally
(e.g. Makefiles, CLI entrypoints, workflow engines) and therefore
appear unused to static analysis.
"""

from hmm.io import split_hmms as _  # used by the Makefile, specifically the setup-hmm-data option

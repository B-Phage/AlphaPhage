# =============================================================================
# Docker Image: myproject (CI / Reproducible Runtime)
#
# This image is built in CI and is intended to provide a fully reproducible
# runtime environment for development, testing, and downstream packaging.
#
# What this image includes:
#
#   ✅ Base OS:
#      - Ubuntu (latest)
#
#   ✅ Package Management:
#      - micromamba (lightweight conda-compatible package manager)
#
#   ✅ Project Code:
#      - Entire repository copied into /app
#
#   ✅ Conda Environments (via Makefile):
#      - preprocessing
#      - training
#      - kaptive3api
#
#   ✅ Installed Python Packages:
#      - Editable installs of all internal packages under /packages
#      - Installed via `make setup-*-pkgs`
#
#   ✅ System Dependencies:
#      - build-essential, curl, git, etc.
#
#   ✅ Test Capability:
#      - Image is capable of running:
#            make test
#            make test-int
#
# What this image does NOT include:
#
#   ❌ External databases (provisioned separately)
#   ❌ DVC-tracked data (requires credentials and runtime pull)
#   ❌ Model snapshots (added later or mounted at runtime)
#
# Intended usage:
#
#   - CI validation of environments and code
#   - Base image for downstream builds (e.g. adding DBs or models)
#   - Portable runtime for inference workflows
#
# Notes:
#
#   - Conda environments are baked into the image under /opt/conda/envs
#   - The Makefile is used as the single source of truth for environment setup
#   - This image may be large due to multiple conda environments
#
# =============================================================================

FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
RUN mv bin/micromamba /usr/local/bin/micromamba

# Make micromamba available as "conda" for Makefile compatibility
RUN ln -s /usr/local/bin/micromamba /usr/local/bin/conda

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

WORKDIR /app
COPY . /app

# Tell Makefile to use micromamba
ENV CONDA_EXE=micromamba

# Build environments + install packages
RUN make setup-preproc-pkgs setup-training-pkgs setup-kaptive-pkgs

CMD ["bash"]

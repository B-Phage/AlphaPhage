# =============================================================================
# Docker Image: myproject (CI / Reproducible Runtime)
#
# This image uses a multi-stage build to balance CI testing with small
# distribution sizes.
#
# 1. Stage 'skeleton':
#    - Minimal footprint (Ubuntu + micromamba + code)
#    - Intended for distribution/shipping.
#    - User runs `make setup-...` on their own hardware to build envs.
#
# 2. Stage 'tester':
#    - Heavy footprint (Includes all conda environments)
#    - Used by GitHub Actions to verify code and reproducibility.
#
# What this image includes:
#   ✅ Base OS: Ubuntu (latest)
#   ✅ Package Management: micromamba
#   ✅ Project Code: Entire repository copied into /app/code
#
# Intended usage:
#   - Base image for downstream builds or portable inference workflows.
# =============================================================================

# STAGE 1: Skeleton (The "Shipping" Image)
# -----------------------------------------------------------------------------
FROM ubuntu:latest AS skeleton

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
ENV CONDA_EXE=micromamba

# Create a parent directory so that ../databases works relative to the code
WORKDIR /app/code
COPY . /app/code

# Note: In this stage, environments are NOT yet built to keep size small.
CMD ["bash"]


# STAGE 2: Tester (The "CI Verification" Image)
# -----------------------------------------------------------------------------
FROM skeleton AS tester

# ✅ Conda Environments (via Makefile):
#    - preprocessing, training, kaptive3api
# ✅ Installed Python Packages:
#    - Editable installs of all internal packages under /packages
#
# This stage is large and intended for CI testing only.
RUN make setup-preproc-pkgs setup-training-pkgs setup-kaptive-pkgs

# Intended for running:
# make test
# make test-int

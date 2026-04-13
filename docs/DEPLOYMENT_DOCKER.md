# 🐳 Docker-Based Inference Deployment

This document defines the process for packaging a validated AlphaPhage
release into a portable Docker image for inference.

This workflow bridges:
- CI-built base images (code + environments)
- Snapshotted models
- External databases
- End-user distribution

---

## 🧠 Overview

The deployment process consists of:

1. Build & validate base image (CI)
2. Pull image locally
3. Add model snapshot + databases
4. Validate container
5. Package and distribute

---

## 🔄 Workflow

### Step 1: Ensure CI Base Image is Available

Confirm that the GitHub Actions workflow:
`.github/workflows/docker_base_build.yml`

has:
- completed successfully
- passed all tests
- pushed an image to GHCR (GitHub Container Registry)

Example tag:
`ghcr.io/b-phage/alphaphage:ci-latest`
    Note the URL is broken up like this:
    `ghcr.io/<org>/<repo>:ci-<sha>`


---

### Step 2: Authenticate with GHCR

Create a GitHub Personal Access Token (PAT):

- Scope: `read:packages`

Then login:

```bash
docker login ghcr.io -u <YOUR_USERNAME> -p <YOUR_TOKEN>
```

---

### Step 3: Pull Base Image

Run this locally.


```bash
docker pull ghcr.io/b-phage/alphaphage::ci-<sha>
```

This assumes you have docker installed and a few extra GB of disk space ~8GB.

Replace <sha> with the relevant SHA from the most recent CI run log. For example, under the **Tag Docker image** heading you should find something like:

    "FULL_SHA: 9785797982b7e3af3c603fbff270e502a2ebf59d
    SHORT_SHA: 9785797"

---

### Step 4: Create Deployment Image

Create/ update a Dockerfile in "docker/inf":

```dockerfile
FROM ghcr.io/b-phage/alphaphage::ci-<sha>

WORKDIR /app

# Copy from local file system to image file system
COPY databases/ /app/../databases/
COPY inference/TransformerX_v1_ep27_20260401/ /app/../inference/TransformerX_v1_ep27_20260401
```
Open bash and `cd` to the parent directory where
`
├── AlphaPhage/
├── databases/
`

Build with dockerfile:
```bash
docker build -t alphaphage:release -f AlphaPhage/docker/inf/Dockerfile .
```

---

### Step 5: Local Validation

Run container:

```bash
docker run -it alphaphage:release bash
```

**Execute inference workflow and confirm:**

- model loads correctly
- databases resolve
- outputs match expected behaviour

---

### Step 6: Package Image

```bash
docker save alphaphage:release | gzip > alphaphage_release.tar.gz
```

---

Step 7: Upload Artifacts
Upload .tar.gz to Google Drive
Store in Model Zoo
Step 8: Link to Release

In GitHub Release Notes:

Add Google Drive link
Specify:
Docker image version
Base image SHA
Snapshot name
📦 Artifact Structure

A valid deployment includes:

Docker image (code + environments + model + DB)
Snapshot metadata
Release notes linkage
⚠️ Notes
Base images are not distributed directly
Final deployment images are fully self-contained
GHCR is used for build + staging, not end-user delivery

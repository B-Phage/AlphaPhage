# AlphaPhage Inference Setup Guide

## 🛠 Prerequisites

Before starting, ensure you have the following:
- Docker Desktop: Installed and running.
- GitHub Access: Membership in the AlphaPhage organization and a Personal Access Token (PAT) with read:packages permissions.
- Model Zoo Access: Permissions to download from the [AlphaPhage Google Drive](https://drive.google.com/drive/folders/1NvwLZgrRJt85ukkmf1USgLVbbYgiV97Y?usp=drive_link).
- Data: At least one phage and one host sequence for testing.

## 🚀 Step 1: Authentication & Image Retrieval
First, authenticate with the GitHub Container Registry and pull the latest image.

```Bash
# Login to GitHub Container Registry
docker login ghcr.io -u YOUR_GITHUB_USERNAME -p YOUR_PAT

# Pull the AlphaPhage image (Ubuntu latest + code)
docker pull ghcr.io/b-phage/alphaphage:ci-latest
```

## 📂 Step 2: Environment Setup

- **Download the Assets:** Download the DB and Model files from the Google Drive link provided above. The file is called `databases_snapshot_backup.zip`
- **Organize your Directory:** Unzip the contents into your local project folder (e.g., ~/AlphaPhage-Inference).
- **Verify Structure:** Ensure your working directory looks like this:

  -   `/databases/hmm`

  -   `/databases/EC-K-typing`

  -   `/databases/inference`

  -   `docker-compose.yml`

```Bash
# Navigate to your local inference folder
cd path/to/your/folder
```

## 🐳 Step 3: Launching the Container
Use Docker Compose to spin up the environment. This command maps your local volumes to the container automatically.

```
Bash
docker compose -f docker-compose.yml --project-directory . run alphaphage
```

## ⚙️ Step 4: Internal Configuration & Execution

Once you are inside the running container's shell, execute the following commands to initialize the environment and start the inference engine:

**Initialization**
These commands install the necessary preprocessing, training, and Kaptive dependencies.

```Bash
make setup-preproc-pkgs setup-training-pkgs setup-kaptive-pkgs
```

**Run Inference**
Execute the main pipeline:

```Bash
make run-inference
```
Note: If you encounter permission issues during make run-inference, ensure your local folder has read/write permissions enabled for Docker.

**Check the Results**
The ouptup will be in:
- `/databases/inference/output/final_predictions.csv`

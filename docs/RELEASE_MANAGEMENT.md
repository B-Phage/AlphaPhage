# 🚀 Release Management & Model Deployment

This document defines the process for stabilizing, validating, and "signing off" a model for production. A **Release** in AlphaPhage is the synchronized coupling of a specific Git version with a validated Model Snapshot, tailored to a specific **Deployment Goal**.

---

## 📐 1. Versioning Strategy

We follow [Semantic Versioning (SemVer)](https://semver.org/):

* **MAJOR (`X.0.0`):** Breaking changes to the inference API, feature vector schemas, or mandatory database migrations.
* **MINOR (`1.X.0`):** New model architectures, retraining on significant new data subsets, or adding new biological domains.
* **PATCH (`1.1.X`):** Bug fixes, documentation, or internal code optimizations that do not change model behavior.

---

## ⚖️ 2. Model "Sign-off" & Performance Profiles

Rather than applying universal thresholds, model sign-off is based on the **Target Deployment Profile**. Every release must identify which profile it satisfies.

### Deployment Profiles

| Profile | Primary Goal | Typical Metric Focus |
| :--- | :--- | :--- |
| **Gold Standard** | Maximum Reliability | High G-Mean; balanced Sensitivity/Specificity (typically $>0.80$). |
| **Broad Screen** | High Sensitivity | Prioritize Recall ($>0.90$) to ensure no potential hits are missed, accepting lower specificity. |
| **Edge/Mobile** | Compute Efficiency | Minimized model depth/dim; thresholds lowered to favor inference speed and low memory footprint. |
| **Niche Domain** | Subset Accuracy | Optimized for specific biological taxa; metrics calculated specifically on the target subset. |

### The Validation Balance
Regardless of the profile, the relationship between **Sensitivity** and **Specificity** must be documented using the metrics from the `TransformerX` module or equivalent:

* **G-Mean:** The primary indicator of overall model "health" ($G = \sqrt{Sens \times Spec}$).
* **Balanced Penalty (`adj_metric`):** Used to quantify the trade-off between classes. A high penalty is acceptable for "Broad Screen" models but should be minimized for "Gold Standard" releases to ensure the model isn't biased.

---

## 🔄 3. The Release Workflow

### Step 1: Research & Goal Identification
Identify the target biological space or compute constraints.
1. Verify `dev` passes all CI logic gates.
2. Select the training epoch that best satisfies the **Target Deployment Profile**.

### Step 2: Model Promotion (The Snapshot)
Promote the validated model to the inference space. This "locks" the weights and the exact configs (HMM schemas, RoPE settings, etc.) used during that specific run.

```bash
# Example: Promoting a model for a specific biological subset
make snapshot MODEL=phage_group_B_v4 EPOCH=latest NAME=v1.2.0_groupB
```

### Step 3: Local Integration Check
Ensure the snapshotted model resolves correctly in the inference environment before merging. This verifies that the newly promoted assets are compatible with the current codebase.

```bash
cd inference
# Update common.mk SNAPSHOT_NAME := v1.2.0_groupB
make all
```
### Step 4: Merge & Tag
- 1. Create a Pull Request from dev to main.
- 2. PR Requirement: Explicitly state the Deployment Profile and metrics.
- 3. Once merged, tag the main branch:
```
git tag -a v1.2.0 -m "Release v1.2.0: High-Sensitivity screen for Group B"
git push origin v1.2.0
```
## 📦 4. Artifact Storage (Google Drive)
1. Compress: Zip the generated snapshot folder found in `../databases/inference/`.
2. Upload: Store the zip in the official AlphaPhage Model Zoo on Google Drive.
3. Link: Add the direct download link to the GitHub Release Notes for the tag.

## 🛠 5. Branching Model
* main: Production-ready code. Every commit represents a "signed-off" Snapshot.
* dev: Integration space for new features and active training experiments.
* feature/*: Short-lived branches for isolated development.

## 🧪 6. Internal Validation
Post-release, a "Pre-flight" check confirms that the snapshotted model,
when running in the inference 'make' environment, reproduces the
metrics reported during the DVC training phase.

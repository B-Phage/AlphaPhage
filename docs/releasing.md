# Release & Versioning Process

This document describes how we prepare, review, and cut a release.
At this stage, releases are **internal-only** and used for team validation and investor demos.
The process is expected to evolve once we move toward public releases.

## Versioning

We follow Semantic Versioning (SemVer)[https://semver.org/]:

- MAJOR: breaking changes
- MINOR: backwards-compatible features
- PATCH: bug fixes and internal improvements

Individual Python packages should also follow SemVer and may be versioned independently.

## Branching Model

- `main`: stable, released code only
- `dev`: integration branch for completed features
- `feature/*`: short-lived feature branches

## Pull Requests & Review

All changes must be introduced via Pull Requests.

Before merging:
- PRs must be reviewed by at least one other team member
- CI must pass
- Versioning implications should be considered and discussed in the PR

No direct commits to `dev` or `main`.

## Feature Integration Flow

1. Develop features on `feature/*` branches
2. Merge completed feature branches into `dev`
3. Resolve conflicts and stabilize `dev`
4. Close or rebase any remaining open feature branches as needed

## Release Preparation

Once all planned features are merged into `dev` and reviewed:

1. Ensure `dev` is in a releasable state
2. Confirm the target version number
3. Update versioned files:
   - `pyproject.toml`
   - `ml_assets/config/<workflow_name>_<version>.yaml`
   - `params.yaml`
4. Verify consistency across all versioned files

## Release Merge & Tagging

1. Merge `dev` into `main` via Pull Request
2. Ensure the PR is reviewed and approved
3. After merge, tag `main` with the release version
4. Create a GitHub Release (if applicable)

## Validation

After tagging:
- Perform internal validation on the released version
- Designate one or more internal users to test the release in real workflows
- Address any critical issues before proceeding with further development

## Early-Stage Notes

This process is for early internal releases.
We expect to refine and possibly automate parts of it as:
- Release frequency increases
- External users are introduced
- CI/CD and release tooling mature

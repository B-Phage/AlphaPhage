
Update the environment files by activating the specific environment and then exporting it.
For example:

```
conda activate root_dev
conda env export --from-history > conda_envs/root_dev.yml
```

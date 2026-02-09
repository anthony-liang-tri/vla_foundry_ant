# TRI-specific files and READMEs

## Dashboard and Leaderboard

See [leaderboard/README.md](frontend/README.md) for documentation on running the leaderboard.
You can access the leaderboard at:
[http://10.242.10.206:8080/leaderboard/](http://10.242.10.206:8080/leaderboard/)

And the dashboard at:
[http://10.242.10.206:8080/](http://10.242.10.206:8080/)

## Shared Directories
- Clean directory: `s3://tri-ml-datasets-uw2/vla_foundry_datasets/`
    - Please do not add anything here until it has already been tested.
    - More specifically, it would be nice if every folder/file in that bucket has a corresponding script in this repo that can be ran to reproduce that dataset.
    - Eventually we probably want to protect this bucket to avoid overwriting. We're small enough now that it's fine.
- Messy directory for scratch: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/`

---

## Ablation Experiments

See [ablations/README.md](ablations/README.md) for documentation on running ablation experiments.


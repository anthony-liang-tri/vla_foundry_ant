Check the status of the Ray cluster and running jobs.

Steps:
1. Run `ray status` to check cluster health (nodes, resources).
2. Run `ray job list` to show recent/running jobs with their status.
3. If there are running or recently failed jobs, offer to show logs with `ray job logs <job_id>`.
4. Summarize: cluster state, number of nodes, running/pending/failed jobs.

If the cluster appears down, suggest starting it with:
`ray up vla_foundry/config_presets/data/preprocessing/ray_cluster_jeanmercat.yaml`
# Eval Report: Qwen3 5.2B Multitask — Unseen Tasks — 2026-04-16

## Model

| Campaign | Model dir | Checkpoint | S3 Checkpoint |
|----------|-----------|-----------|---------------|
| qwen3_unseen_anzu / qwen3_unseen_oss | qwen3_5_2b | checkpoint_11 | [s3](s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/multitask/mt/qwen3_5_2b/2026_03_24-01_40_07-model_diffusion_policy-lr_5e-05-bsz_1024/) |

## Results — Anzu Sim

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAvocadoFromBowlIntoBin | 10 | 200 | 5.0% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 8 | 200 | 4.0% |
| PutMugInCenterOfTable | 72 | 200 | 36.0% |
| **Unweighted Mean** | | | **15.0%** |

## Results — OSS Sim

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAvocadoFromBowlIntoBin | 8 | 190 | 4.2% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 5 | 200 | 2.5% |
| PutMugInCenterOfTable | 51 | 180 | 28.3% |
| **Unweighted Mean** | | | **11.7%** |

## Notes

- 3 unseen tasks (not in training data)
- Anzu: 600/600 episodes completed
- OSS: 570/600 episodes completed (30 episodes from failed jobs not recovered)
- PutMugInCenterOfTable is the strongest unseen task (~30-36%), the bimanual tasks are much harder (~2-5%)

## S3 Result Paths
- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/multitask/mt/qwen3_5_2b/2026_03_24-01_40_07-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-15_qwen3_unseen_anzu/`
- OSS: `s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/multitask/mt/qwen3_5_2b/2026_03_24-01_40_07-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-15_qwen3_unseen_oss/`

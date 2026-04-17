# Eval Report: SmolVLM Foundry Backbone Sim Subset ckpt11 — 2026-04-17

## Model

| Field | Value |
|-------|-------|
| Model dir | `smolvlm_foundry_backbone_sim_subset_resume` |
| Run name | `2026_04_15-04_33_40-model_diffusion_policy-lr_5e-05-bsz_1024` |
| Checkpoint | `checkpoint_11.pt` (final, 100k steps) |
| EMA | Disabled |
| Architecture | Diffusion policy, SmolVLM Foundry Backbone (1B-equiv), lr=5e-5, bsz=1024 |
| Training | v0.4.3.9 sim subset (42 tasks), 100k steps, flow matching |
| WandB | [run](https://wandb.ai/tri/vla_foundry/runs/AWSBatchdo-not-kill-sim-res-kus5be343efda88300b93e23e4f0dfe6da9-5mnc4l-algo-1) |
| S3 checkpoint | `s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/multitask/mt/smolvlm_foundry_backbone_sim_subset_resume/2026_04_15-04_33_40-model_diffusion_policy-lr_5e-05-bsz_1024/` |

> Note: run originally crashed at step ~48881 (checkpoints 1–4 saved to `smolvlm_foundry_backbone_sim_subset/...`), was resumed on a new SageMaker job that completed all 100k steps and saved checkpoints 5–11 to the `_resume` path.

## Summary

| Metric | Anzu | OSS |
|--------|-----:|----:|
| Seen mean (16 tasks) | **60.5%** | **41.0%** |
| Unseen mean (3 tasks) | 8.2% | 11.7% |
| All 19 mean | 52.2% | 36.4% |
| Weighted overall | 1984/3799 = 52.2% | 1373/3786 = 36.3% |

## Results — Anzu Sim

### Seen tasks (16)

| Task | Succ | Total | Rate |
|------|-----:|------:|-----:|
| BimanualPlaceAppleFromBowlIntoBin | 151 | 200 | 75.5% |
| BimanualPlaceFruitFromBowlIntoBin | 172 | 200 | 86.0% |
| BimanualPutRedBellPepperInBin | 173 | 200 | 86.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 153 | 200 | 76.5% |
| BimanualPutSpatulaOnPlateFromTable | 156 | 200 | 78.0% |
| BimanualStackPlatesOnTableFromDryingRack | 183 | 200 | 91.5% |
| BimanualStoreCerealBoxUnderShelf | 159 | 200 | 79.5% |
| PlaceCupByCoaster | 75 | 200 | 37.5% |
| PushCoasterToCenterOfTable | 126 | 200 | 63.0% |
| PushCoasterToMug | 39 | 199 | 19.6% |
| PutBananaOnSaucer | 23 | 200 | 11.5% |
| PutKiwiInCenterOfTable | 10 | 200 | 5.0% |
| PutMugOnSaucer | 118 | 200 | 59.0% |
| PutSpatulaInUtensilCrock | 137 | 200 | 68.5% |
| TurnCupUpsideDown | 136 | 200 | 68.0% |
| TurnMugRightsideUp | 124 | 200 | 62.0% |
| **Seen Mean (unweighted)** | | | **60.5%** |

### Unseen tasks (3)

| Task | Succ | Total | Rate |
|------|-----:|------:|-----:|
| BimanualPlaceAvocadoFromBowlIntoBin | 40 | 200 | 20.0% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 7 | 200 | 3.5% |
| PutMugInCenterOfTable | 2 | 200 | 1.0% |
| **Unseen Mean (unweighted)** | | | **8.2%** |

## Results — OSS Sim

### Seen tasks (16)

| Task | Succ | Total | Rate |
|------|-----:|------:|-----:|
| BimanualPlaceAppleFromBowlIntoBin | 135 | 200 | 67.5% |
| BimanualPlaceFruitFromBowlIntoBin | 127 | 186 | 68.3% |
| BimanualPutRedBellPepperInBin | 157 | 200 | 78.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 112 | 200 | 56.0% |
| BimanualPutSpatulaOnPlateFromTable | 77 | 200 | 38.5% |
| BimanualStackPlatesOnTableFromDryingRack | 126 | 200 | 63.0% |
| BimanualStoreCerealBoxUnderShelf | 101 | 200 | 50.5% |
| PlaceCupByCoaster | 32 | 200 | 16.0% |
| PushCoasterToCenterOfTable | 48 | 200 | 24.0% |
| PushCoasterToMug | 30 | 200 | 15.0% |
| PutBananaOnSaucer | 19 | 200 | 9.5% |
| PutKiwiInCenterOfTable | 31 | 200 | 15.5% |
| PutMugOnSaucer | 61 | 200 | 30.5% |
| PutSpatulaInUtensilCrock | 62 | 200 | 31.0% |
| TurnCupUpsideDown | 104 | 200 | 52.0% |
| TurnMugRightsideUp | 81 | 200 | 40.5% |
| **Seen Mean (unweighted)** | | | **41.0%** |

### Unseen tasks (3)

| Task | Succ | Total | Rate |
|------|-----:|------:|-----:|
| BimanualPlaceAvocadoFromBowlIntoBin | 39 | 200 | 19.5% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 2 | 200 | 1.0% |
| PutMugInCenterOfTable | 29 | 200 | 14.5% |
| **Unseen Mean (unweighted)** | | | **11.7%** |

## S3 Result Paths

- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/multitask/mt/smolvlm_foundry_backbone_sim_subset_resume/2026_04_15-04_33_40-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-16_smolvlm_bb_sim_ckpt11_anzu/`
- OSS: `s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/multitask/mt/smolvlm_foundry_backbone_sim_subset_resume/2026_04_15-04_33_40-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-16_smolvlm_bb_sim_ckpt11_oss/`

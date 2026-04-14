# Eval Report: dp_simonly_bsz1024_ckpt21 — 2026-04-13

## Model(s)

| Campaign | Model dir | Checkpoint | WandB | S3 Checkpoint |
|----------|-----------|-----------|-------|---------------|
| dp_simonly_bsz1024_ckpt21 | 2026_04_12-10_41_10-model_diffusion_policy-lr_5e-05-bsz_1024 | checkpoint_21 | [run](https://wandb.ai/tri/vla_foundry/runs/AWSBatchvla-kushal2-mt-sim-only6509956a7cc13139b45f64e29f3581b2-rxs96z-algo-7) | [s3](s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_sim_only/2026_04_12-10_41_10-model_diffusion_policy-lr_5e-05-bsz_1024/) |

Sim-only multitask model (trained on 42 tasks from `v0.4.3.9/sim/manifest.jsonl`, evaluated on standard 16).

## Results — Anzu Sim

### dp_simonly_bsz1024_ckpt21 (multi-task, 16 tasks)

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 131 | 200 | 65.5% |
| BimanualPlaceFruitFromBowlIntoBin | 159 | 200 | 79.5% |
| BimanualPutRedBellPepperInBin | 169 | 200 | 84.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 131 | 200 | 65.5% |
| BimanualPutSpatulaOnPlateFromTable | 128 | 200 | 64.0% |
| BimanualStackPlatesOnTableFromDryingRack | 153 | 200 | 76.5% |
| BimanualStoreCerealBoxUnderShelf | 120 | 200 | 60.0% |
| PlaceCupByCoaster | 96 | 200 | 48.0% |
| PushCoasterToCenterOfTable | 122 | 200 | 61.0% |
| PushCoasterToMug | 36 | 193 | 18.7% |
| PutBananaOnSaucer | 25 | 200 | 12.5% |
| PutKiwiInCenterOfTable | 8 | 200 | 4.0% |
| PutMugOnSaucer | 111 | 200 | 55.5% |
| PutSpatulaInUtensilCrock | 123 | 200 | 61.5% |
| TurnCupUpsideDown | 147 | 200 | 73.5% |
| TurnMugRightsideUp | 128 | 200 | 64.0% |
| **Unweighted Mean** | | | **55.9%** |

## Results — OSS Sim

### dp_simonly_bsz1024_ckpt21_oss (multi-task, 16 tasks)

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 123 | 200 | 61.5% |
| BimanualPlaceFruitFromBowlIntoBin | 133 | 200 | 66.5% |
| BimanualPutRedBellPepperInBin | 132 | 200 | 66.0% |
| BimanualPutSpatulaOnPlateFromDryingRack | 119 | 200 | 59.5% |
| BimanualPutSpatulaOnPlateFromTable | 61 | 200 | 30.5% |
| BimanualStackPlatesOnTableFromDryingRack | 121 | 200 | 60.5% |
| BimanualStoreCerealBoxUnderShelf | 82 | 200 | 41.0% |
| PlaceCupByCoaster | 53 | 200 | 26.5% |
| PushCoasterToCenterOfTable | 51 | 200 | 25.5% |
| PushCoasterToMug | 28 | 200 | 14.0% |
| PutBananaOnSaucer | 12 | 200 | 6.0% |
| PutKiwiInCenterOfTable | 23 | 200 | 11.5% |
| PutMugOnSaucer | 54 | 200 | 27.0% |
| PutSpatulaInUtensilCrock | 51 | 200 | 25.5% |
| TurnCupUpsideDown | 110 | 200 | 55.0% |
| TurnMugRightsideUp | 62 | 200 | 31.0% |
| **Unweighted Mean** | | | **38.0%** |

## Comparison vs real-data model (dp_bsz1024_ckpt14)

| Metric | dp_bsz1024_ckpt14 (real) | dp_simonly_bsz1024_ckpt21 (sim) |
|---|---|---|
| Anzu mean | 30.1% | **55.9%** |
| OSS mean | 16.7% | **38.0%** |

Sim-only training nearly doubles success in both sims vs real-data training.

## S3 Result Paths
- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_sim_only/2026_04_12-10_41_10-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-13_dp_simonly_bsz1024_ckpt21/`
- OSS: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_sim_only/2026_04_12-10_41_10-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-13_dp_simonly_bsz1024_ckpt21_oss/`

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
| BimanualPlaceFruitFromBowlIntoBin | 164 | 200 | 82.0% |
| BimanualPutRedBellPepperInBin | 169 | 200 | 84.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 143 | 200 | 71.5% |
| BimanualPutSpatulaOnPlateFromTable | 128 | 200 | 64.0% |
| BimanualStackPlatesOnTableFromDryingRack | 150 | 200 | 75.0% |
| BimanualStoreCerealBoxUnderShelf | 120 | 200 | 60.0% |
| PlaceCupByCoaster | 91 | 197 | 46.2% |
| PushCoasterToCenterOfTable | 121 | 185 | 65.4% |
| PushCoasterToMug | 35 | 174 | 20.1% |
| PutBananaOnSaucer | 25 | 200 | 12.5% |
| PutKiwiInCenterOfTable | 8 | 200 | 4.0% |
| PutMugOnSaucer | 111 | 200 | 55.5% |
| PutSpatulaInUtensilCrock | 123 | 200 | 61.5% |
| TurnCupUpsideDown | 144 | 200 | 72.0% |
| TurnMugRightsideUp | 128 | 200 | 64.0% |
| **Unweighted Mean** | | | **56.5%** |

## Results — OSS Sim

### dp_simonly_bsz1024_ckpt21_oss (multi-task, 16 tasks)

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 123 | 200 | 61.5% |
| BimanualPlaceFruitFromBowlIntoBin | 120 | 188 | 63.8% |
| BimanualPutRedBellPepperInBin | 132 | 200 | 66.0% |
| BimanualPutSpatulaOnPlateFromDryingRack | 98 | 188 | 52.1% |
| BimanualPutSpatulaOnPlateFromTable | 61 | 200 | 30.5% |
| BimanualStackPlatesOnTableFromDryingRack | 108 | 188 | 57.4% |
| BimanualStoreCerealBoxUnderShelf | 82 | 200 | 41.0% |
| PlaceCupByCoaster | 54 | 200 | 27.0% |
| PushCoasterToCenterOfTable | 51 | 200 | 25.5% |
| PushCoasterToMug | 31 | 200 | 15.5% |
| PutBananaOnSaucer | 12 | 200 | 6.0% |
| PutKiwiInCenterOfTable | 23 | 200 | 11.5% |
| PutMugOnSaucer | 54 | 200 | 27.0% |
| PutSpatulaInUtensilCrock | 51 | 200 | 25.5% |
| TurnCupUpsideDown | 103 | 187 | 55.1% |
| TurnMugRightsideUp | 62 | 200 | 31.0% |
| **Unweighted Mean** | | | **37.3%** |

## Comparison vs real-data model (dp_bsz1024_ckpt14)

| Metric | dp_bsz1024_ckpt14 (real) | dp_simonly_bsz1024_ckpt21 (sim) |
|---|---|---|
| Anzu mean | 30.1% | **56.5%** |
| OSS mean | 16.7% | **37.3%** |

Sim-only training nearly doubles success in both sims vs real-data training.

## S3 Result Paths
- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_sim_only/2026_04_12-10_41_10-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-13_dp_simonly_bsz1024_ckpt21/`
- OSS (in progress): `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_sim_only/2026_04_12-10_41_10-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-13_dp_simonly_bsz1024_ckpt21_oss/`

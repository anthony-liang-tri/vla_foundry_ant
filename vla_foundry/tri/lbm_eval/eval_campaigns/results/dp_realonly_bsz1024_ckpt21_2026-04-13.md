# Eval Report: dp_realonly_bsz1024_ckpt21 — 2026-04-13

## Model(s)

| Campaign | Model dir | Checkpoint | WandB | S3 Checkpoint |
|----------|-----------|-----------|-------|---------------|
| dp_realonly_bsz1024_ckpt21 | 2026_04_12-10_51_13-model_diffusion_policy-lr_5e-05-bsz_1024 | checkpoint_21 | [run](https://wandb.ai/tri/vla_foundry/runs/AWSBatchvla-kushal2-mt-real-onl69ddf58454853116bad0ee204e572d8e-41k6ke-algo-8) | [s3](s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_real_only/2026_04_12-10_51_13-model_diffusion_policy-lr_5e-05-bsz_1024/) |

Real-only multitask model (trained on real data from `v0.4.3.9/real/manifest.jsonl`, evaluated on standard 16).

## Results — Anzu Sim

### dp_realonly_bsz1024_ckpt21 (multi-task, 16 tasks)

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 0 | 200 | 0.0% |
| BimanualPlaceFruitFromBowlIntoBin | 0 | 200 | 0.0% |
| BimanualPutRedBellPepperInBin | 0 | 200 | 0.0% |
| BimanualPutSpatulaOnPlateFromDryingRack | 0 | 200 | 0.0% |
| BimanualPutSpatulaOnPlateFromTable | 0 | 200 | 0.0% |
| BimanualStackPlatesOnTableFromDryingRack | 0 | 200 | 0.0% |
| BimanualStoreCerealBoxUnderShelf | 0 | 200 | 0.0% |
| PlaceCupByCoaster | 0 | 200 | 0.0% |
| PushCoasterToCenterOfTable | 1 | 192 | 0.5% |
| PushCoasterToMug | 4 | 159 | 2.5% |
| PutBananaOnSaucer | 0 | 200 | 0.0% |
| PutKiwiInCenterOfTable | 0 | 200 | 0.0% |
| PutMugOnSaucer | 2 | 200 | 1.0% |
| PutSpatulaInUtensilCrock | 0 | 200 | 0.0% |
| TurnCupUpsideDown | 1 | 200 | 0.5% |
| TurnMugRightsideUp | 0 | 200 | 0.0% |
| **Unweighted Mean** | | | **0.3%** |

## Results — OSS Sim

### dp_realonly_bsz1024_ckpt21_oss (multi-task, 16 tasks)

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 0 | 200 | 0.0% |
| BimanualPlaceFruitFromBowlIntoBin | 0 | 200 | 0.0% |
| BimanualPutRedBellPepperInBin | 0 | 200 | 0.0% |
| BimanualPutSpatulaOnPlateFromDryingRack | 0 | 200 | 0.0% |
| BimanualPutSpatulaOnPlateFromTable | 0 | 200 | 0.0% |
| BimanualStackPlatesOnTableFromDryingRack | 0 | 188 | 0.0% |
| BimanualStoreCerealBoxUnderShelf | 0 | 200 | 0.0% |
| PlaceCupByCoaster | 10 | 200 | 5.0% |
| PushCoasterToCenterOfTable | 12 | 175 | 6.9% |
| PushCoasterToMug | 4 | 123 | 3.3% |
| PutBananaOnSaucer | 1 | 200 | 0.5% |
| PutKiwiInCenterOfTable | 21 | 200 | 10.5% |
| PutMugOnSaucer | 0 | 187 | 0.0% |
| PutSpatulaInUtensilCrock | 0 | 200 | 0.0% |
| TurnCupUpsideDown | 0 | 187 | 0.0% |
| TurnMugRightsideUp | 0 | 200 | 0.0% |
| **Unweighted Mean** | | | **1.6%** |

## Comparison: real-only vs sim-only vs mixed (all ckpt21 unless noted)

| Model | Anzu Mean | OSS Mean |
|---|---|---|
| dp_realonly_bsz1024_ckpt21 (real only) | **0.3%** | **1.6%** |
| dp_bsz1024_ckpt14 (real+sim mixed) | 30.1% | 16.7% |
| dp_simonly_bsz1024_ckpt21 (sim only) | 56.5% | 37.3% |

Real-only training produces ~0% success in both sims. The mixed-data model benefits from sim data; sim-only dominates.

## S3 Result Paths
- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_real_only/2026_04_12-10_51_13-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-13_dp_realonly_bsz1024_ckpt21/`
- OSS: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_real_only/2026_04_12-10_51_13-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-13_dp_realonly_bsz1024_ckpt21_oss/`

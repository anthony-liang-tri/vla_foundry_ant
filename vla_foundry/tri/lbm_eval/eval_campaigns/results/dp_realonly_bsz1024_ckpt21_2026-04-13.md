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

_Pending — running now._

## Comparison: real-only vs sim-only vs mixed (all ckpt21, anzu sim)

| Model | Anzu Mean |
|---|---|
| dp_realonly_bsz1024_ckpt21 (real only) | **0.3%** |
| dp_bsz1024_ckpt14 (real+sim mixed) | 30.1% |
| dp_simonly_bsz1024_ckpt21 (sim only) | 56.5% |

Real-only training produces ~0% success in sim eval. The mixed-data model (30.1%) benefits from sim data; sim-only (56.5%) dominates.

## S3 Result Paths
- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_real_only/2026_04_12-10_51_13-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-13_dp_realonly_bsz1024_ckpt21/`
- OSS (in progress): `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/multitask_real_only/2026_04_12-10_51_13-model_diffusion_policy-lr_5e-05-bsz_1024/evaluation/2026-04-13_dp_realonly_bsz1024_ckpt21_oss/`

# Eval Report: kushal_nominal_1b_10k_st — 2026-04-12

## Model(s)

16 single-task models from the `kushal_nominal` 1B VLM backbone ablation, trained for **10k steps** (step 9999). Each model trained on one task with `vlm_foundry_backbone`, bsz_512, lr 5e-05, ws=16 (2 nodes × 8 GPUs).

## Results — OSS Sim

### kushal_nominal_1b_10k_st_oss (16 single-task models, unweighted mean = 38.7%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAppleFromBowlIntoBin | 118 / 200 | 59.0% |
| BimanualPlaceFruitFromBowlIntoBin | 57 / 200 | 28.5% |
| BimanualPutRedBellPepperInBin | 82 / 188 | 43.6% |
| BimanualPutSpatulaOnPlateFromDryingRack | 90 / 188 | 47.9% |
| BimanualPutSpatulaOnPlateFromTable | 48 / 200 | 24.0% |
| BimanualStackPlatesOnTableFromDryingRack | 174 / 200 | 87.0% |
| BimanualStoreCerealBoxUnderShelf | 111 / 200 | 55.5% |
| PlaceCupByCoaster | 79 / 200 | 39.5% |
| PushCoasterToCenterOfTable | 39 / 100 | 39.0% |
| PushCoasterToMug | 2 / 12 | 16.7% |
| PutBananaOnSaucer | 13 / 200 | 6.5% |
| PutKiwiInCenterOfTable | 33 / 200 | 16.5% |
| PutMugOnSaucer | 17 / 124 | 13.7% |
| PutSpatulaInUtensilCrock | 47 / 200 | 23.5% |
| TurnCupUpsideDown | 162 / 200 | 81.0% |
| TurnMugRightsideUp | 74 / 200 | 37.0% |
| **Unweighted Mean** | | **38.7%** |

> Some tasks below 200 demos — jobs still running or failed. PushCoasterToMug particularly low sample count.

## Results — Anzu Sim

*Still running (~78% complete). Will update when done.*

## Comparison: 2k vs 10k steps

| Task | 2k Anzu | 10k Anzu | 2k OSS | 10k OSS |
|------|---------|----------|--------|---------|
| BimanualPlaceApple | 31.0% | pending | 12.5% | **59.0%** |
| BimanualPlaceFruit | 0.0% | pending | 0.0% | **28.5%** |
| BimanualPutRedBellPepper | 18.0% | pending | 2.5% | **43.6%** |
| BimanualPutSpatulaOnPlateFromDryingRack | 31.9% | pending | 20.5% | 47.9% |
| BimanualPutSpatulaOnPlateFromTable | 33.0% | pending | 15.0% | 24.0% |
| BimanualStackPlatesOnTableFromDryingRack | 51.0% | pending | 39.0% | **87.0%** |
| BimanualStoreCerealBoxUnderShelf | 20.5% | pending | 13.5% | 55.5% |
| PlaceCupByCoaster | 30.7% | pending | 21.5% | 39.5% |
| PushCoasterToCenterOfTable | 58.8% | pending | 20.2% | 39.0% |
| PushCoasterToMug | 8.3% | pending | — | 16.7% |
| PutBananaOnSaucer | 7.5% | pending | 5.5% | 6.5% |
| PutKiwiInCenterOfTable | 9.5% | pending | 11.7% | 16.5% |
| PutMugOnSaucer | 8.0% | pending | 3.1% | 13.7% |
| PutSpatulaInUtensilCrock | 34.0% | pending | 5.0% | 23.5% |
| TurnCupUpsideDown | 26.5% | pending | 17.5% | **81.0%** |
| TurnMugRightsideUp | 21.0% | pending | 6.9% | 37.0% |
| **Mean** | 24.4% | pending | 13.0% | **38.7%** |

> 10k step training significantly improves OSS results (13.0% → 38.7% mean, ~3x).

## S3 Result Paths
- OSS: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/<TaskName>_10ksteps/*/evaluation/2026-04-12_kushal_nominal_1b_10k_st_oss/`
- Anzu: `s3://.../<TaskName>_10ksteps/*/evaluation/2026-04-12_kushal_nominal_1b_10k_st_anzu/`

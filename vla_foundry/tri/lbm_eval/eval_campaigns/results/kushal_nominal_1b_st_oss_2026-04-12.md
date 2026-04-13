# Eval Report: kushal_nominal_1b_st_oss — 2026-04-12

## Model(s)

16 single-task models from the `kushal_nominal` 1B VLM backbone ablation. Each model trained on one task with `vlm_foundry_backbone`, bsz_512, lr 5e-05, 2k steps (step 1952).

## Results — OSS Sim

### kushal_nominal_1b_st_oss (16 single-task models, unweighted mean = 13.0%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAppleFromBowlIntoBin | 25 / 200 | 12.5% |
| BimanualPlaceFruitFromBowlIntoBin | 0 / 200 | 0.0% |
| BimanualPutRedBellPepperInBin | 5 / 200 | 2.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 41 / 200 | 20.5% |
| BimanualPutSpatulaOnPlateFromTable | 30 / 200 | 15.0% |
| BimanualStackPlatesOnTableFromDryingRack | 78 / 200 | 39.0% |
| BimanualStoreCerealBoxUnderShelf | 27 / 200 | 13.5% |
| PlaceCupByCoaster | 43 / 200 | 21.5% |
| PushCoasterToCenterOfTable | 20 / 99 | 20.2% |
| PushCoasterToMug | — | pending |
| PutBananaOnSaucer | 11 / 200 | 5.5% |
| PutKiwiInCenterOfTable | 22 / 188 | 11.7% |
| PutMugOnSaucer | 5 / 162 | 3.1% |
| PutSpatulaInUtensilCrock | 10 / 200 | 5.0% |
| TurnCupUpsideDown | 35 / 200 | 17.5% |
| TurnMugRightsideUp | 13 / 188 | 6.9% |
| **Unweighted Mean (15 tasks)** | | **13.0%** |

> Unweighted mean = average of per-task rates (excluding PushCoasterToMug which has no results yet).
> Some tasks below 200 demos due to jobs still running.

## Comparison: Anzu vs OSS

| Task | Anzu | OSS |
|------|------|-----|
| BimanualPlaceAppleFromBowlIntoBin | 31.0% | 12.5% |
| BimanualPlaceFruitFromBowlIntoBin | 0.0% | 0.0% |
| BimanualPutRedBellPepperInBin | 18.0% | 2.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 31.9% | 20.5% |
| BimanualPutSpatulaOnPlateFromTable | 33.0% | 15.0% |
| BimanualStackPlatesOnTableFromDryingRack | 51.0% | 39.0% |
| BimanualStoreCerealBoxUnderShelf | 20.5% | 13.5% |
| PlaceCupByCoaster | 30.7% | 21.5% |
| PushCoasterToCenterOfTable | 58.8% | 20.2% |
| PushCoasterToMug | 8.3% | — |
| PutBananaOnSaucer | 7.5% | 5.5% |
| PutKiwiInCenterOfTable | 9.5% | 11.7% |
| PutMugOnSaucer | 8.0% | 3.1% |
| PutSpatulaInUtensilCrock | 34.0% | 5.0% |
| TurnCupUpsideDown | 26.5% | 17.5% |
| TurnMugRightsideUp | 21.0% | 6.9% |
| **Mean** | **24.4%** | **13.0%** |

## S3 Result Paths
- OSS: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/<TaskName>*/evaluation/2026-04-12_kushal_nominal_1b_st_oss/`

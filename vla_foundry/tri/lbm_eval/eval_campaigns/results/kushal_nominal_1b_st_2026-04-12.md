# Eval Report: kushal_nominal_1b_st — 2026-04-12

## Model(s)

16 single-task models from the `kushal_nominal` 1B VLM backbone ablation. Each model trained on one task with `vlm_foundry_backbone`, bsz_512, lr 5e-05, 2k steps (step 1952).

## Results — Anzu Sim

### kushal_nominal_1b_st (16 single-task models, unweighted mean = 22.4%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAppleFromBowlIntoBin | 0 / 200 | 0.0% |
| BimanualPlaceFruitFromBowlIntoBin | 0 / 190 | 0.0% |
| BimanualPutRedBellPepperInBin | 36 / 200 | 18.0% |
| BimanualPutSpatulaOnPlateFromDryingRack | 60 / 188 | 31.9% |
| BimanualPutSpatulaOnPlateFromTable | 66 / 200 | 33.0% |
| BimanualStackPlatesOnTableFromDryingRack | 102 / 200 | 51.0% |
| BimanualStoreCerealBoxUnderShelf | 41 / 200 | 20.5% |
| PlaceCupByCoaster | 61 / 199 | 30.7% |
| PushCoasterToCenterOfTable | 114 / 194 | 58.8% |
| PushCoasterToMug | 9 / 109 | 8.3% |
| PutBananaOnSaucer | 15 / 200 | 7.5% |
| PutKiwiInCenterOfTable | 19 / 200 | 9.5% |
| PutMugOnSaucer | 16 / 200 | 8.0% |
| PutSpatulaInUtensilCrock | 68 / 200 | 34.0% |
| TurnCupUpsideDown | 53 / 200 | 26.5% |
| TurnMugRightsideUp | 42 / 200 | 21.0% |
| **Unweighted Mean** | | **22.4%** |

> Unweighted mean = average of per-task rates.
> BimanualPlaceApple, BimanualPlaceFruit, BimanualPutRedBellPepper: corrected with 2k step models (originally evaluated with 10k step models by mistake).
> PushCoasterToMug has only 109 demos (some jobs failed).

## S3 Result Paths

Results are spread across 16 checkpoint paths under:
- 13 tasks: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/<TaskName>*/evaluation/2026-04-12_kushal_nominal_1b_st_v2/`
- 3 corrected tasks: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/<TaskName>*/evaluation/2026-04-12_kushal_nominal_1b_2k_fix/`

# Eval Report: kushal_1b_ft_realonly — 2026-04-14

## Model(s)

19 single-task finetuned models from `vla_kushal_nominal_ft_realonly/`. Each model is a 2k step finetune of the kushal_nominal multitask 100k step base model, specialized to one task. Trained on **real-only data** (no sim data).

Includes 16 seen tasks + 3 unseen tasks (BimanualPlaceAvocadoFromBowlIntoBin, BimanualPutSpatulaOnPlateFromUtensilCrock, PutMugInCenterOfTable).

## Results — Anzu Sim

### kushal_1b_ft_realonly_anzu (19 tasks, unweighted mean = 50.9%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAppleFromBowlIntoBin | 122 / 200 | 61.0% |
| BimanualPlaceAvocadoFromBowlIntoBin | 88 / 191 | 46.1% |
| BimanualPlaceFruitFromBowlIntoBin | 110 / 194 | 56.7% |
| BimanualPutRedBellPepperInBin | 146 / 200 | 73.0% |
| BimanualPutSpatulaOnPlateFromDryingRack | 83 / 187 | 44.4% |
| BimanualPutSpatulaOnPlateFromTable | 108 / 200 | 54.0% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 86 / 200 | 43.0% |
| BimanualStackPlatesOnTableFromDryingRack | 162 / 200 | 81.0% |
| BimanualStoreCerealBoxUnderShelf | 110 / 200 | 55.0% |
| PlaceCupByCoaster | 73 / 195 | 37.4% |
| PushCoasterToCenterOfTable | 125 / 200 | 62.5% |
| PushCoasterToMug | 11 / 61 | 18.0% |
| PutBananaOnSaucer | 42 / 191 | 22.0% |
| PutKiwiInCenterOfTable | 29 / 200 | 14.5% |
| PutMugInCenterOfTable | 118 / 196 | 60.2% |
| PutMugOnSaucer | 110 / 200 | 55.0% |
| PutSpatulaInUtensilCrock | 83 / 200 | 41.5% |
| TurnCupUpsideDown | 143 / 180 | 79.4% |
| TurnMugRightsideUp | 125 / 200 | 62.5% |
| **Unweighted Mean** | | **50.9%** |

> PushCoasterToMug (61) and TurnCupUpsideDown (180) still partial.

## Results — OSS Sim

### kushal_1b_ft_realonly_oss (19 tasks, unweighted mean = 27.7%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAppleFromBowlIntoBin | 77 / 192 | 40.1% |
| BimanualPlaceAvocadoFromBowlIntoBin | 13 / 180 | 7.2% |
| BimanualPlaceFruitFromBowlIntoBin | 72 / 200 | 36.0% |
| BimanualPutRedBellPepperInBin | 71 / 172 | 41.3% |
| BimanualPutSpatulaOnPlateFromDryingRack | 59 / 200 | 29.5% |
| BimanualPutSpatulaOnPlateFromTable | 22 / 180 | 12.2% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 34 / 180 | 18.9% |
| BimanualStackPlatesOnTableFromDryingRack | 104 / 175 | 59.4% |
| BimanualStoreCerealBoxUnderShelf | 63 / 200 | 31.5% |
| PlaceCupByCoaster | 36 / 191 | 18.8% |
| PushCoasterToCenterOfTable | 70 / 191 | 36.6% |
| PushCoasterToMug | 16 / 120 | 13.3% |
| PutBananaOnSaucer | 25 / 191 | 13.1% |
| PutKiwiInCenterOfTable | 26 / 180 | 14.4% |
| PutMugInCenterOfTable | 69 / 187 | 36.9% |
| PutMugOnSaucer | 42 / 180 | 23.3% |
| PutSpatulaInUtensilCrock | 16 / 186 | 8.6% |
| TurnCupUpsideDown | 106 / 200 | 53.0% |
| TurnMugRightsideUp | 57 / 174 | 32.8% |
| **Unweighted Mean** | | **27.7%** |

## Comparison: FT (sim+real data) vs FT realonly (16 seen tasks only)

| Task | FT Anzu | FT realonly Anzu |
|------|---------|------------------|
| BimanualPlaceApple | 78.5% | 61.0% |
| BimanualPlaceFruit | 73.5% | 56.7% |
| BimanualPutRedBellPepper | 86.0% | 73.0% |
| BimanualPutSpatulaFromDryingRack | 63.5% | 44.4% |
| BimanualPutSpatulaFromTable | 70.0% | 54.0% |
| BimanualStackPlates | 79.0% | 81.0% |
| BimanualStoreCerealBox | 67.0% | 55.0% |
| PlaceCupByCoaster | 50.0% | 37.4% |
| PushCoasterToCenter | 67.4% | 62.5% |
| PushCoasterToMug | 25.3% | 18.0% |
| PutBananaOnSaucer | 19.5% | 22.0% |
| PutKiwiInCenterOfTable | 16.0% | 14.5% |
| PutMugOnSaucer | 64.5% | 55.0% |
| PutSpatulaInUtensilCrock | 63.0% | 41.5% |
| TurnCupUpsideDown | 82.5% | 79.4% |
| TurnMugRightsideUp | 73.0% | 62.5% |
| **Mean (16 seen)** | **61.2%** | **51.7%** |

> FT with sim+real data (61.2%) outperforms real-only FT (51.7%) by ~10 pts on seen tasks.
> Unseen tasks (avocado 46.1%, spatula_utensilcrock 43.0%, mug_center 60.2%) average 49.8%.

## S3 Result Paths
- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/<TaskName>/*/evaluation/2026-04-14_kushal_1b_ft_realonly_anzu/`

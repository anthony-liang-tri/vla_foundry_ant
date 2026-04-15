# Eval Report: Qwen3 Combined OSS — 2026-04-14

## Models

Combined evaluation of 3 model groups on OSS sim:
1. **Qwen3-FT-v0.4.3.9** — 16 seen tasks, finetuned from Qwen3 multitask
2. **Qwen3-FT-v0.4.3.19** — 3 unseen tasks, finetuned from Qwen3 multitask
3. **NominalConfig-ST-Sim** — 15 tasks, nominal config single-task sim-trained

## Results — OSS Sim

### NominalConfig-ST-Sim (16 tasks, mean = 57.3%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAppleFromBowlIntoBin | 182 / 191 | 95.3% |
| BimanualPlaceFruitFromBowlIntoBin | 70 / 183 | 38.3% |
| BimanualPutRedBellPepperInBin | 150 / 183 | 82.0% |
| BimanualPutSpatulaOnPlateFromDryingRack | 0 / 180 | 0.0% |
| BimanualPutSpatulaOnPlateFromTable | 97 / 192 | 50.5% |
| BimanualStackPlatesOnTableFromDryingRack | 174 / 200 | 87.0% |
| BimanualStoreCerealBoxUnderShelf | 126 / 184 | 68.5% |
| PlaceCupByCoaster | 128 / 191 | 67.0% |
| PushCoasterToCenterOfTable | 104 / 151 | 68.9% |
| PushCoasterToMug | 21 / 71 | 29.6% |
| PutBananaOnSaucer | 43 / 145 | 29.7% |
| PutKiwiInCenterOfTable | 78 / 192 | 40.6% |
| PutMugOnSaucer | 67 / 131 | 51.1% |
| PutSpatulaInUtensilCrock | 82 / 191 | 42.9% |
| TurnCupUpsideDown | 159 / 180 | 88.3% |
| TurnMugRightsideUp | 125 / 163 | 76.7% |
| **Unweighted Mean** | | **57.3%** |

> Note: BimanualPutSpatulaOnPlateFromDryingRack was filled in a follow-up run (2026-04-15). 18 of 20 batches landed (180/200 demos); 3 demos hit Drake NaN sim failures, the rest ran full 30 s without success. Result is solid 0%. The 3 unseen tasks (Avocado, SpatulaUtensilCrock, MugCenter) have no NominalConfig-ST checkpoint — only multitask and FT models were trained on those.

### Qwen3-FT-v0.4.3.9 (16 seen tasks, mean = 44.3%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAppleFromBowlIntoBin | 147 / 200 | 73.5% |
| BimanualPlaceFruitFromBowlIntoBin | 128 / 180 | 71.1% |
| BimanualPutRedBellPepperInBin | 179 / 200 | 89.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 116 / 200 | 58.0% |
| BimanualPutSpatulaOnPlateFromTable | 121 / 200 | 60.5% |
| BimanualStackPlatesOnTableFromDryingRack | 143 / 200 | 71.5% |
| BimanualStoreCerealBoxUnderShelf | 63 / 200 | 31.5% |
| PlaceCupByCoaster | 80 / 200 | 40.0% |
| PushCoasterToCenterOfTable | 12 / 200 | 6.0% |
| PushCoasterToMug | 9 / 175 | 5.1% |
| PutBananaOnSaucer | 96 / 180 | 53.3% |
| PutKiwiInCenterOfTable | 110 / 200 | 55.0% |
| PutMugOnSaucer | 71 / 200 | 35.5% |
| PutSpatulaInUtensilCrock | 77 / 200 | 38.5% |
| TurnCupUpsideDown | 11 / 175 | 6.3% |
| TurnMugRightsideUp | 27 / 200 | 13.5% |
| **Unweighted Mean** | | **44.3%** |

### Qwen3-FT-v0.4.3.19 (3 unseen tasks, mean = 39.5%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAvocadoFromBowlIntoBin | 93 / 150 | 62.0% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 15 / 190 | 7.9% |
| PutMugInCenterOfTable | 75 / 154 | 48.7% |
| **Unweighted Mean** | | **39.5%** |

## S3 Result Paths
- OSS: Under each model's S3 checkpoint path at `evaluation/2026-04-14_qwen3_combined_oss/`

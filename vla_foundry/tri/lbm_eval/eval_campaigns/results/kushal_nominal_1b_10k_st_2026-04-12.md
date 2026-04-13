# Eval Report: kushal_nominal_1b_10k_st — 2026-04-12

## Model(s)

16 single-task models from the `kushal_nominal` 1B VLM backbone ablation, trained for **10k steps** (step 9999). Each model trained on one task with `vlm_foundry_backbone`, bsz_512, lr 5e-05, ws=16 (2 nodes × 8 GPUs).

## Results — Anzu Sim

### kushal_nominal_1b_10k_st_anzu (16 single-task models, unweighted mean = 52.6%)

| Task | Success / Total | Rate |
|------|-----------------|------|
| BimanualPlaceAppleFromBowlIntoBin | 141 / 200 | 70.5% |
| BimanualPlaceFruitFromBowlIntoBin | 70 / 193 | 36.3% |
| BimanualPutRedBellPepperInBin | 139 / 200 | 69.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 105 / 195 | 53.8% |
| BimanualPutSpatulaOnPlateFromTable | 119 / 200 | 59.5% |
| BimanualStackPlatesOnTableFromDryingRack | 180 / 200 | 90.0% |
| BimanualStoreCerealBoxUnderShelf | 111 / 200 | 55.5% |
| PlaceCupByCoaster | 108 / 200 | 54.0% |
| PushCoasterToCenterOfTable | 141 / 192 | 73.4% |
| PushCoasterToMug | 39 / 194 | 20.1% |
| PutBananaOnSaucer | 18 / 200 | 9.0% |
| PutKiwiInCenterOfTable | 23 / 200 | 11.5% |
| PutMugOnSaucer | 50 / 200 | 25.0% |
| PutSpatulaInUtensilCrock | 122 / 196 | 62.2% |
| TurnCupUpsideDown | 176 / 200 | 88.0% |
| TurnMugRightsideUp | 126 / 200 | 63.0% |
| **Unweighted Mean** | | **52.6%** |

## Results — OSS Sim

### kushal_nominal_1b_10k_st_oss (16 single-task models, unweighted mean = 38.3%)

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
| PushCoasterToCenterOfTable | 66 / 187 | 35.3% |
| PushCoasterToMug | 28 / 188 | 14.9% |
| PutBananaOnSaucer | 13 / 200 | 6.5% |
| PutKiwiInCenterOfTable | 33 / 200 | 16.5% |
| PutMugOnSaucer | 25 / 200 | 12.5% |
| PutSpatulaInUtensilCrock | 47 / 200 | 23.5% |
| TurnCupUpsideDown | 162 / 200 | 81.0% |
| TurnMugRightsideUp | 74 / 200 | 37.0% |
| **Unweighted Mean** | | **38.3%** |

## Comparison: 2k vs 10k steps

| Task | 2k Anzu | 10k Anzu | 2k OSS | 10k OSS |
|------|---------|----------|--------|---------|
| BimanualPlaceApple | 31.0% | **70.5%** | 12.5% | **59.0%** |
| BimanualPlaceFruit | 0.0% | **36.3%** | 0.0% | **28.5%** |
| BimanualPutRedBellPepper | 18.0% | **69.5%** | 2.5% | **43.6%** |
| BimanualPutSpatulaFromDryingRack | 31.9% | **53.8%** | 20.5% | **47.9%** |
| BimanualPutSpatulaFromTable | 33.0% | **59.5%** | 15.0% | **24.0%** |
| BimanualStackPlates | 51.0% | **90.0%** | 39.0% | **87.0%** |
| BimanualStoreCerealBox | 20.5% | **55.5%** | 13.5% | **55.5%** |
| PlaceCupByCoaster | 30.7% | **54.0%** | 21.5% | **39.5%** |
| PushCoasterToCenter | 58.8% | **73.4%** | 20.2% | **35.3%** |
| PushCoasterToMug | 8.3% | **20.1%** | — | **14.9%** |
| PutBananaOnSaucer | 7.5% | **9.0%** | 5.5% | **6.5%** |
| PutKiwiInCenterOfTable | 9.5% | **11.5%** | 11.7% | **16.5%** |
| PutMugOnSaucer | 8.0% | **25.0%** | 3.1% | **12.5%** |
| PutSpatulaInUtensilCrock | 34.0% | **62.2%** | 5.0% | **23.5%** |
| TurnCupUpsideDown | 26.5% | **88.0%** | 17.5% | **81.0%** |
| TurnMugRightsideUp | 21.0% | **63.0%** | 6.9% | **37.0%** |
| **Mean** | 24.4% | **52.6%** | 13.0% | **38.3%** |

> 10k step training dramatically improves both sims: Anzu 24.4%→52.6% (2.2x), OSS 13.0%→38.3% (3x).

## S3 Result Paths
- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/<TaskName>_10ksteps/*/evaluation/2026-04-12_kushal_nominal_1b_10k_st_anzu/`
- OSS: `s3://.../evaluation/2026-04-12_kushal_nominal_1b_10k_st_oss/`

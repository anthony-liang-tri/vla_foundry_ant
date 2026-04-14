# Eval Report: kushal_ft_simonly — 2026-04-13

## Model(s)

19 single-task sim-only fine-tuned models from `vla_kushal_nominal_ft_simonly`. Each task has its own model fine-tuned on sim-only data, checkpoint_3.pt (latest, no EMA).

| Task | Run | S3 Checkpoint |
|------|-----|---------------|
| Each task | Latest run per task | `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_simonly/{task}/{latest_run}/` |

## Results — Anzu Sim

> Note: PushCoasterToMug (61/200) and PushCoasterToCenterOfTable (182/200) have incomplete samples due to slow tail jobs. All other tasks have 199-200 samples.

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 148 | 200 | 74.0% |
| BimanualPlaceFruitFromBowlIntoBin | 170 | 200 | 85.0% |
| BimanualPutRedBellPepperInBin | 163 | 200 | 81.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 125 | 199 | 62.8% |
| BimanualPutSpatulaOnPlateFromTable | 138 | 200 | 69.0% |
| BimanualStackPlatesOnTableFromDryingRack | 161 | 200 | 80.5% |
| BimanualStoreCerealBoxUnderShelf | 140 | 200 | 70.0% |
| PlaceCupByCoaster | 87 | 200 | 43.5% |
| PushCoasterToCenterOfTable | 126 | 182 | 69.2% |
| PushCoasterToMug | 13 | 61 | 21.3% |
| PutBananaOnSaucer | 27 | 200 | 13.5% |
| PutKiwiInCenterOfTable | 8 | 200 | 4.0% |
| PutMugOnSaucer | 115 | 200 | 57.5% |
| PutSpatulaInUtensilCrock | 139 | 200 | 69.5% |
| TurnCupUpsideDown | 166 | 200 | 83.0% |
| TurnMugRightsideUp | 134 | 200 | 67.0% |
| **Seen Mean (16 tasks)** | | | **59.5%** |

### Unseen Tasks — Anzu

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAvocadoFromBowlIntoBin | 120 | 200 | 60.0% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 124 | 200 | 62.0% |
| PutMugInCenterOfTable | 118 | 200 | 59.0% |
| **Unseen Mean (3 tasks)** | | | **60.3%** |

## Results — OSS Sim

> Note: BimanualPlaceAppleFromBowlIntoBin (185/200), BimanualPlaceAvocadoFromBowlIntoBin (172/200), PutMugInCenterOfTable (184/200) have slightly incomplete samples.

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 116 | 185 | 62.7% |
| BimanualPlaceFruitFromBowlIntoBin | 132 | 200 | 66.0% |
| BimanualPutRedBellPepperInBin | 148 | 200 | 74.0% |
| BimanualPutSpatulaOnPlateFromDryingRack | 114 | 200 | 57.0% |
| BimanualPutSpatulaOnPlateFromTable | 69 | 200 | 34.5% |
| BimanualStackPlatesOnTableFromDryingRack | 148 | 200 | 74.0% |
| BimanualStoreCerealBoxUnderShelf | 125 | 200 | 62.5% |
| PlaceCupByCoaster | 54 | 200 | 27.0% |
| PushCoasterToCenterOfTable | 60 | 200 | 30.0% |
| PushCoasterToMug | 35 | 200 | 17.5% |
| PutBananaOnSaucer | 13 | 200 | 6.5% |
| PutKiwiInCenterOfTable | 24 | 200 | 12.0% |
| PutMugOnSaucer | 68 | 200 | 34.0% |
| PutSpatulaInUtensilCrock | 62 | 200 | 31.0% |
| TurnCupUpsideDown | 127 | 200 | 63.5% |
| TurnMugRightsideUp | 89 | 200 | 44.5% |
| **Seen Mean (16 tasks)** | | | **43.5%** |

### Unseen Tasks — OSS

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAvocadoFromBowlIntoBin | 66 | 172 | 38.4% |
| BimanualPutSpatulaOnPlateFromUtensilCrock | 61 | 200 | 30.5% |
| PutMugInCenterOfTable | 79 | 184 | 42.9% |
| **Unseen Mean (3 tasks)** | | | **37.3%** |

## Summary

| Metric | Anzu | OSS |
|--------|------|-----|
| Seen Mean (16 tasks) | **59.5%** | **43.5%** |
| Unseen Mean (3 tasks) | **60.3%** | **37.3%** |

## S3 Result Paths
- Anzu: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_simonly/{task}/{run}/evaluation/2026-04-13_kushal_ft_simonly_anzu/`
- OSS: `s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_simonly/{task}/{run}/evaluation/2026-04-13_kushal_ft_simonly_oss/`

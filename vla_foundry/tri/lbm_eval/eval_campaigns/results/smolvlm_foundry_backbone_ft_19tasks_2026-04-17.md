# Eval Report: smolvlm_foundry_backbone_ft (19 tasks) — 2026-04-17

## Model

Single-task fine-tuned SmolVLM foundry-backbone models — one checkpoint per task, 19 total (16 seen + 3 unseen). FT from SmolVLM MT checkpoint_11 on v0.4.3.9 data, lr=5e-6, 1,024,000 samples (2k steps @ bsz 512).

- MT base: `s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/multitask/mt/smolvlm_foundry_backbone_resume/2026_04_15-04_50_00-model_diffusion_policy-lr_5e-05-bsz_1024/checkpoints/checkpoint_11.pt`
- ST FT checkpoints: `s3://tri-ml-datasets-uw2/vla_foundry/model_checkpoints/vla/ablations_v04_3_9/st/<Task>/smolvlm_foundry_backbone_ft/<run_dir>/` (see [task list](../task_lists/smolvlm_foundry_backbone_ft_19tasks.txt))

## Campaigns

| Sim | Campaign | S3 subfolder |
|-----|----------|--------------|
| Anzu | [smolvlm_foundry_backbone_ft_19tasks_anzu](../campaigns/smolvlm_foundry_backbone_ft_19tasks_anzu.yaml) | `evaluation/2026-04-17_smolvlm_foundry_backbone_ft_19tasks_anzu/` |
| OSS  | [smolvlm_foundry_backbone_ft_19tasks_oss](../campaigns/smolvlm_foundry_backbone_ft_19tasks_oss.yaml)   | `evaluation/2026-04-17_smolvlm_foundry_backbone_ft_19tasks_oss/`  |

## Summary

| Sim | All 19 | 16 Seen | 3 Unseen |
|-----|--------|---------|----------|
| Anzu | **60.0%** | 59.6% | 62.5% |
| OSS  | **35.8%** | 37.4% | 27.3% |

## Per-task Results

### ANZU Sim

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 155 | 200 | 77.5% |
| BimanualPlaceFruitFromBowlIntoBin | 154 | 198 | 77.8% |
| BimanualPutRedBellPepperInBin | 163 | 200 | 81.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 141 | 200 | 70.5% |
| BimanualPutSpatulaOnPlateFromTable | 165 | 200 | 82.5% |
| BimanualStackPlatesOnTableFromDryingRack | 160 | 200 | 80.0% |
| BimanualStoreCerealBoxUnderShelf | 127 | 200 | 63.5% |
| PlaceCupByCoaster | 63 | 200 | 31.5% |
| PushCoasterToCenterOfTable | 131 | 200 | 65.5% |
| PushCoasterToMug | 35 | 194 | 18.0% |
| PutBananaOnSaucer | 29 | 200 | 14.5% |
| PutKiwiInCenterOfTable | 9 | 200 | 4.5% |
| PutMugOnSaucer | 137 | 200 | 68.5% |
| PutSpatulaInUtensilCrock | 141 | 198 | 71.2% |
| TurnCupUpsideDown | 145 | 200 | 72.5% |
| TurnMugRightsideUp | 147 | 200 | 73.5% |
| BimanualPlaceAvocadoFromBowlIntoBin (unseen) | 120 | 200 | 60.0% |
| BimanualPutSpatulaOnPlateFromUtensilCrock (unseen) | 109 | 200 | 54.5% |
| PutMugInCenterOfTable (unseen) | 146 | 200 | 73.0% |
| **Mean (19 all)** | | | **60.0%** |
| **Mean (16 seen)** | | | **59.6%** |
| **Mean (3 unseen)** | | | **62.5%** |

### OSS Sim

| Task | Success | Total | Rate |
|------|---------|-------|------|
| BimanualPlaceAppleFromBowlIntoBin | 121 | 200 | 60.5% |
| BimanualPlaceFruitFromBowlIntoBin | 112 | 200 | 56.0% |
| BimanualPutRedBellPepperInBin | 121 | 200 | 60.5% |
| BimanualPutSpatulaOnPlateFromDryingRack | 84 | 200 | 42.0% |
| BimanualPutSpatulaOnPlateFromTable | 67 | 190 | 35.3% |
| BimanualStackPlatesOnTableFromDryingRack | 82 | 200 | 41.0% |
| BimanualStoreCerealBoxUnderShelf | 89 | 190 | 46.8% |
| PlaceCupByCoaster | 34 | 190 | 17.9% |
| PushCoasterToCenterOfTable | 49 | 190 | 25.8% |
| PushCoasterToMug | 30 | 200 | 15.0% |
| PutBananaOnSaucer | 28 | 200 | 14.0% |
| PutKiwiInCenterOfTable | 26 | 200 | 13.0% |
| PutMugOnSaucer | 84 | 200 | 42.0% |
| PutSpatulaInUtensilCrock | 50 | 200 | 25.0% |
| TurnCupUpsideDown | 117 | 200 | 58.5% |
| TurnMugRightsideUp | 91 | 200 | 45.5% |
| BimanualPlaceAvocadoFromBowlIntoBin (unseen) | 38 | 200 | 19.0% |
| BimanualPutSpatulaOnPlateFromUtensilCrock (unseen) | 24 | 200 | 12.0% |
| PutMugInCenterOfTable (unseen) | 102 | 200 | 51.0% |
| **Mean (19 all)** | | | **35.8%** |
| **Mean (16 seen)** | | | **37.4%** |
| **Mean (3 unseen)** | | | **27.3%** |

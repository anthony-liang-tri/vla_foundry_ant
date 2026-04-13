# Eval Report: BimanualPlaceFruitFromBowlIntoBin Apr 13 Re-eval — 2026-04-13

## Model

| Run | Step | S3 Path |
|-----|------|---------|
| 2026_04_13-01_47_52 | 1952 (2k) | s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/BimanualPlaceFruitFromBowlIntoBin/2026_04_13-01_47_52-model_diffusion_policy-lr_5e-05-bsz_512/ |

Re-verification of BimanualPlaceFruitFromBowlIntoBin 2k step single-task model, since previous eval showed 0%.

## Results

| Sim | Success / Total | Rate |
|-----|-----------------|------|
| Anzu | 0 / 177 | 0.0% |
| OSS  | 0 / 170 | 0.0% |

> **Confirmed 0% on both sims.** Same result as the previous Apr 12 2k step run (`2026_04_12-08_27_36`, also 0.0% anzu).
> 2k step training is insufficient for BimanualPlaceFruitFromBowlIntoBin. The 10k step model achieved 36.3% anzu / 28.5% OSS on this task.

## S3 Result Paths
- Anzu: `s3://.../evaluation/2026-04-13_fruit_reeval_apr13_anzu/`
- OSS: `s3://.../evaluation/2026-04-13_fruit_reeval_apr13_oss/`

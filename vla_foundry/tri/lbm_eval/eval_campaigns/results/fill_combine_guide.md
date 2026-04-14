# Fill Campaign — Combine Guide

The `2026-04-14_fill` eval subfolder contains fill results that need to be combined with the original eval subfolders.

After the fill campaign completes, run these S3 copy commands to merge fill results into the originals:

```bash
# FT PushCoasterToMug (anzu: 2026-04-13_kushal_1b_ft_anzu, oss: 2026-04-13_kushal_1b_ft_oss)
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft/PushCoasterToMug/2026_04_13-17_05_16-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft/PushCoasterToMug/2026_04_13-17_05_16-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-13_kushal_1b_ft_anzu/ --recursive --profile sagemaker

# FT PushCoasterToCenterOfTable
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft/PushCoasterToCenterOfTable/2026_04_13-17_05_10-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft/PushCoasterToCenterOfTable/2026_04_13-17_05_10-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-13_kushal_1b_ft_anzu/ --recursive --profile sagemaker

# ST 2k PushCoasterToMug (anzu: 2026-04-12_kushal_nominal_1b_st_v2)
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/PushCoasterToMug/2026_04_12-08_37_26-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/PushCoasterToMug/2026_04_12-08_37_26-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-12_kushal_nominal_1b_st_v2/ --recursive --profile sagemaker

# ST 2k PushCoasterToCenterOfTable
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/PushCoasterToCenterOfTable/2026_04_12-08_37_09-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/PushCoasterToCenterOfTable/2026_04_12-08_37_09-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-12_kushal_nominal_1b_st_v2/ --recursive --profile sagemaker

# ST 2k BimanualPutSpatulaOnPlateFromDryingRack
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/BimanualPutSpatulaOnPlateFromDryingRack/2026_04_12-08_29_07-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/BimanualPutSpatulaOnPlateFromDryingRack/2026_04_12-08_29_07-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-12_kushal_nominal_1b_st_v2/ --recursive --profile sagemaker

# ST 10k PushCoasterToCenterOfTable (anzu: 2026-04-12_kushal_nominal_1b_10k_st_anzu)
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/PushCoasterToCenterOfTable_10ksteps/2026_04_12-18_57_51-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/PushCoasterToCenterOfTable_10ksteps/2026_04_12-18_57_51-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-12_kushal_nominal_1b_10k_st_anzu/ --recursive --profile sagemaker

# ST 10k PushCoasterToMug
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/PushCoasterToMug_10ksteps/2026_04_12-19_05_39-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal/PushCoasterToMug_10ksteps/2026_04_12-19_05_39-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-12_kushal_nominal_1b_10k_st_anzu/ --recursive --profile sagemaker

# FT realonly BimanualPlaceAvocadoFromBowlIntoBin (anzu: 2026-04-14_kushal_1b_ft_realonly_anzu)
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/BimanualPlaceAvocadoFromBowlIntoBin/2026_04_13-18_37_16-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/BimanualPlaceAvocadoFromBowlIntoBin/2026_04_13-18_37_16-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_kushal_1b_ft_realonly_anzu/ --recursive --profile sagemaker

# FT realonly BimanualPlaceFruitFromBowlIntoBin
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/BimanualPlaceFruitFromBowlIntoBin/2026_04_13-18_37_27-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/BimanualPlaceFruitFromBowlIntoBin/2026_04_13-18_37_27-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_kushal_1b_ft_realonly_anzu/ --recursive --profile sagemaker

# FT realonly BimanualPutSpatulaOnPlateFromDryingRack
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/BimanualPutSpatulaOnPlateFromDryingRack/2026_04_13-18_51_40-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/BimanualPutSpatulaOnPlateFromDryingRack/2026_04_13-18_51_40-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_kushal_1b_ft_realonly_anzu/ --recursive --profile sagemaker

# FT realonly PutBananaOnSaucer
aws s3 cp s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/PutBananaOnSaucer/2026_04_13-19_21_30-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_fill/ s3://tri-ml-datasets-uw2/vla_foundry_scratch/models/vla_kushal_nominal_ft_realonly/PutBananaOnSaucer/2026_04_13-19_21_30-model_diffusion_policy-lr_5e-05-bsz_512/evaluation/2026-04-14_kushal_1b_ft_realonly_anzu/ --recursive --profile sagemaker
```

Note: The fill campaign runs all 200 demos but only the missing indices will produce new results. Existing demo indices in the original folder won't be overwritten by the copy since S3 cp won't overwrite existing files with --recursive (demo folders have unique names like demonstration_N/).

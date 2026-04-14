# Examples Overview

The `examples/` directory contains sample bash scripts for the four main workflows in VLA Foundry: **preprocessing**, **training**, **deployment**, and **visualization**. These scripts are ready to run and serve as starting points for your own experiments.

## Directory Structure

```
examples/
  training/                 # Common training routines
    llm_11m.sh              # Minimal LLM training (11M params)
    llm_1b.sh               # LLM training (1B params)
    llm_1b_small.sh         # LLM training (1B params, small config)
    llm_410m_full.sh        # LLM training (410M params, full run)
    vlm_paligemma3b.sh      # VLM training with PaliGemma
    vlm_paligemma3b_hf.sh   # VLM training with HF-backed model
    vlm_paligemma3b_full.sh           # Full-scale VLM training
    vlm_paligemma3b_full_hf.sh        # Full-scale VLM (HF-backed)
    vlm_paligemma3b_full_fromllm.sh   # VLM initialized from LLM
    vlm_smolvlm_full_fromllm.sh       # SmolVLM from LLM checkpoint
    vlm_smolvlm_full_fromllm_correct_sizes.sh  # SmolVLM (corrected sizes)
    diffusion_policy.sh     # Diffusion Policy training
    vla_diffusion_redbellpepper_paligemma2.sh  # VLA diffusion example
    resume.sh               # Finetuning from a pretrained checkpoint
    extended/               # Additional settings
      llm/                  # HF untokenized text training
      diffusion_policy/     # Multi-dataset, augmentation, Unitree G1
      vlm/                  # VLM with SigLIP
      stable_diffusion/     # Stable Diffusion variants
  preprocessing/            # Data preprocessing scripts
    preprocess_robotics_data_lbm.sh     # LBM Spartan to tar shards
    preprocess_robotics_data_lerobot.sh # LeRobot to tar shards
    update_test_assets.sh               # Update test fixture data
    extended/
      preprocess_robotics_data_lerobot_g1.sh  # LeRobot G1 humanoid
      preprocess_robotics_data_mcap_g1.sh     # MCAP ROS 2 recordings
  deployment/               # Model deployment and evaluation
    libero.sh               # LIBERO evaluation
    robocasa.sh             # RoboCasa evaluation
    lbm_eval/               # LBM evaluation suite
      launch_wave_policy.sh         # Launch WAVE policy eval
      launch_inference_policy.sh    # Launch inference policy eval
      download_model.sh             # Download model checkpoint
      download_model_from_wandb.py  # Download model from W&B
      README.md
  visualization/            # Data visualization tools
    visualize_data.sh       # Visualize dataset samples
    visualization_params.yaml  # Visualization config
    README.md
```

## Training

The `training/` folder contains the most common training routines for LLMs, VLMs, and Diffusion Policies. Each script demonstrates a different model type and data modality. The `training/extended/` subfolder has additional configurations for multi-dataset training, image augmentation, and alternative model architectures.

All training scripts use `torchrun` as the launcher and `vla_foundry/main.py` as the entrypoint.

[See annotated training examples](training.md){ .md-button }

## Preprocessing

The `preprocessing/` folder has scripts for converting raw data from various formats (Spartan, LeRobot, MCAP, HF datasets) into the WebDataset tar shard format that VLA Foundry expects at training time.

All preprocessing scripts call `vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py` or related preprocessing utilities.

[See annotated preprocessing examples](preprocessing.md){ .md-button }

## Deployment

The `deployment/` folder contains scripts for deploying trained models to simulation environments (LIBERO, RoboCasa). The `lbm_eval/` subfolder provides tooling for the LBM evaluation suite, including model download helpers and inference launch scripts.

## Visualization

The `visualization/` folder contains scripts for visualizing dataset samples. Use `visualize_data.sh` with the accompanying `visualization_params.yaml` to inspect your data before training.

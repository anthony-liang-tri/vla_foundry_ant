# Examples

This directory contains example scripts and configurations for training, preprocessing, debugging, and deployment of VLA Foundry models.

## Directory Structure

```
examples/
├── training/          # Training scripts for different model types
├── preprocessing/     # Data preprocessing and conversion scripts
├── debug/            # Debugging and demonstration scripts
└── deployment/       # Deployment and evaluation scripts
```

## Training Examples

Ready-to-run training scripts for various model architectures:

### Language Models (LLMs)
- **`llm_11m.sh`** - Small 11M parameter transformer (recommended starting point)
- **`llm_1b.sh`** - 1B parameter transformer model
- **`llm_hf_untokenized.sh`** - Hugging Face transformer with untokenized data
- **`llm_1b_full.sh`** - Full 1B parameter LLM model

### Vision-Language Models (VLMs)
- **`vlm_paligemma3b.sh`** - PaLiGemma 3B VLM model
- **`vlm_paligemma3b_hf.sh`** - Hugging Face PaLiGemma 3B model
- **`vlm_siglip_paligemma3b.sh`** - SigLIP + PaLiGemma 3B combination

### Diffusion Models
- **`stable_diffusion_use_diffusers.sh`** - Stable Diffusion using Diffusers library
- **`stable_diffusion_use_diffusers_flow_matching.sh`** - Flow matching variant
- **`stable_diffusion_no_diffusers.sh`** - Custom implementation without Diffusers

### Policy Models
- **`fake_policy.sh`** - Example policy training script

## Preprocessing Examples

Data preparation and conversion scripts:

- **`preprocess_lbm.sh`** - Convert LBM episodes to tar format for training
- **`preprocess_lbm_tiny.sh`** - Small-scale preprocessing for testing
- **`preprocess_and_push_to_hub.sh`** - Convert LBM data to LeRobot format and push to Hugging Face Hub

For detailed preprocessing documentation, see [vla_foundry/data/scripts/preprocessing/README.md](../vla_foundry/data/scripts/preprocessing/README.md).

## Debug Examples

- **`transformer_embedding_resize_demo.py`** - Demonstrates transformer model creation and embedding resizing

## Deployment Examples

- **`lbm_eval/launch_wave_policy.sh`** - Launch a policy that waves around the robot

## Getting Started

**Note**: All scripts should be run from the repository root directory.

1. **Start simple**: Run `examples/training/llm_11m.sh` to train a small language model
2. **Explore preprocessing**: Try `examples/preprocessing/preprocess_lbm_tiny.sh` for data processing
3. **Debug and understand**: Run `examples/debug/transformer_embedding_resize_demo.py` to see model internals
4. **Scale up**: Modify parameters in the training scripts for your specific use case

For detailed training, data loading, and configuration information, see the main [README.md](../README.md).

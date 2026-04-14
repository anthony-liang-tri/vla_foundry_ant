# Diffusion-Based Models

VLA Foundry includes three diffusion-based architectures: Diffusion Policy for robotics action prediction, Stable Diffusion for image generation, and ManiFlow for consistency flow matching with point cloud inputs.

## Diffusion Policy

**Type key:** `diffusion_policy`
**Source:** `vla_foundry/models/diffusion_policy/diffusion_policy.py`
**Params:** [`DiffusionPolicyParams`](../params/model-params.md#diffusionpolicyparams)

Diffusion Policy predicts robot actions by iteratively denoising a noise vector conditioned on visual and language observations. It is the primary architecture for robotics policy training in VLA Foundry.

### Architecture

```
Camera Images + Language Instruction
              |
              v
    CLIP Backbone (frozen or trainable)
     |                    |
     v                    v
  Visual Tokens      Text Tokens
     |                    |
     +---- Concatenate ---+
              |
              v
     Conditioning Sequence
              |
              +--- Noisy Action Sequence (+ timestep embedding)
              |
              v
        Transformer Denoiser (bidirectional)
              |
              v
        Predicted Clean Actions
```

### Key Design Choices

- **CLIP backbone** for visual-language conditioning. The text encoder can be frozen independently of the image encoder.
- **Bidirectional transformer** as the denoiser (set `is_causal: false` on the transformer sub-component).
- **Flow matching** scheduler by default (`use_flow_matching_scheduler: true`), which provides faster convergence than DDPM.
- **Diffusion step conditioning** via concatenation (`"concat"`) or addition (`"add"`).
- **Action and proprioception dimensions** are automatically derived from `DataParams`.

### Config Preset

```yaml
# config_presets/models/diffusion_policy.yaml
type: diffusion_policy
transformer:
  <<: !include transformer_100m.yaml
  is_causal: false
vision_language_backbone:
  type: clip_backbone
  hf_pretrained: openai/clip-vit-base-patch32
  disable_text: false
noise_scheduler:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  clamp_range: [-3, 3]
use_flow_matching_scheduler: true
```

### Training Job Presets

| Preset | Task | File |
|--------|------|------|
| `diffusion_policy_bellpepper` | LBM BellPepper bimanual task | `training_jobs/diffusion_policy_bellpepper.yaml` |
| `diffusion_policy_lbm1` | LBM1 bimanual manipulation | `training_jobs/diffusion_policy_lbm1.yaml` |
| `diffusion_policy_unitree_g1` | Unitree G1 humanoid | `training_jobs/diffusion_policy_unitree_g1.yaml` |
| `diffusion_policy_humanoid_everyday` | Humanoid everyday tasks | `training_jobs/diffusion_policy_humanoid_everyday.yaml` |
| `diffusion_policy_lerobot_libero` | LeRobot LIBERO benchmark | `training_jobs/diffusion_policy_lerobot_libero.yaml` |

### Usage

```bash
torchrun --nproc_per_node=8 vla_foundry/main.py \
    --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
    --total_train_samples 30_000_000 \
    --num_checkpoints 10 \
    --remote_sync s3://my-bucket/diffusion_policy
```

---

## Stable Diffusion

**Type key:** `stable_diffusion`
**Source:** `vla_foundry/models/diffusion/stable_diffusion.py`
**Params:** [`StableDiffusionParams`](../params/model-params.md#stablediffusionparams)

A text-conditioned latent diffusion model for image generation. Supports classifier-free guidance (CFG).

### Architecture

```
Text Input
    |
    v
CLIP Text Encoder
    |
    v
Text Embeddings ---> UNet Denoiser <--- Noisy Latents + Timestep
                          |
                          v
                   Predicted Noise / Clean Latents
```

### Components

| Component | Params | Description |
|-----------|--------|-------------|
| **UNet** | `UNetParams` | The denoising backbone. Configurable channel counts per resolution level. |
| **Noise Scheduler** | `NoiseSchedulerParams` | DDPM or flow matching noise schedule. |
| **CLIP** | `CLIPHFParams` | Text encoder for conditioning. |

### Config Preset

```yaml
# Example Stable Diffusion configuration
model:
  type: stable_diffusion
  unet:
    type: unet
    in_channels: 3
    out_channels: 3
    time_emb_dim: 256
    text_emb_dim: 512
    channels: [128, 256, 512, 1024]
  noise_scheduler:
    num_timesteps: 1000
    beta_start: 0.0001
    beta_end: 0.02
  clip:
    type: clip_hf
    hf_pretrained: openai/clip-vit-base-patch32
  do_classifier_free_guidance: true
  guidance_scale: 4.0
  dropout_percent: 0.2
```

---

## ManiFlow

**Type key:** `maniflow`
**Source:** `vla_foundry/models/maniflow/maniflow.py`
**Params:** [`ManiFlowParams`](../params/model-params.md#maniflowparams)

ManiFlow is a consistency flow matching model for robotics that operates on point cloud observations. It combines a DP3 point cloud encoder with a DiTX transformer action predictor.

### Architecture

```
Point Cloud + Proprioception
         |
         v
   DP3 Encoder (PointNet)
         |
         v
   Visual Conditioning Features
         |
         +--- Noisy Action Trajectory (+ timestep embedding)
         |
         v
   DiTX Transformer (action predictor)
         |
         v
   Predicted Clean Action Trajectory
```

### Training Strategy

ManiFlow uses a hybrid training objective that combines flow matching and consistency losses within each batch:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `flow_batch_ratio` | `0.75` | 75% of each batch trains with flow matching loss |
| `consistency_batch_ratio` | `0.25` | 25% of each batch trains with consistency loss |

This hybrid approach yields faster inference (fewer denoising steps) while maintaining sample quality.

### Components

#### DP3 Encoder

**Type key:** `dp3_encoder`

Encodes point cloud observations into a fixed-length feature vector using farthest point sampling and PointNet.

```yaml
encoder:
  type: dp3_encoder
  num_fps_points: 4096
  use_pc_color: true
  out_channel: 256
  pointnet_type: pointnet
  pointcloud_encoder_cfg:
    in_channels: 3
    out_channels: 256
    num_points: 4096
    pointwise: true
    use_layernorm: true
    final_norm: layernorm
```

#### DiTX Transformer

**Type key:** `ditx`

A Diffusion Transformer (DiT) variant designed for action prediction, with modality-specific conditioning for diffusion timesteps and visual features.

```yaml
action_predictor:
  type: ditx
  n_layer: 12
  n_head: 8
  n_emb: 768
  qkv_bias: true
  qk_norm: true
  diffusion_timestep_embed_dim: 128
  diffusion_target_t_embed_dim: 128
  visual_cond_len: 4096
```

### Config Preset

```yaml
# config_presets/models/maniflow.yaml
type: maniflow
encoder:
  type: dp3_encoder
  num_fps_points: 4096
  use_pc_color: true
  out_channel: 256
  pointcloud_encoder_cfg:
    in_channels: 3
    out_channels: 256
    num_points: 4096
    pointwise: true
    use_layernorm: true
    final_norm: layernorm
action_predictor:
  type: ditx
  n_layer: 12
  n_head: 8
  n_emb: 768
  qkv_bias: true
  qk_norm: true
flow_batch_ratio: 0.75
consistency_batch_ratio: 0.25
num_inference_steps: 10
```

### Training Job Preset

```yaml
# config_presets/training_jobs/maniflow_example.yaml
model:
  <<: !include ../models/maniflow.yaml

data:
  <<: !include ../data/maniflow.yaml
  dataset_manifest:
    - s3://your-bucket/your-dataset/manifest.jsonl
  dataset_statistics:
    - s3://your-bucket/your-dataset/stats.json
  dataset_modality:
    - robotics
  dataset_weighting:
    - 1.0

hparams:
  <<: !include ../hparams/maniflow.yaml
  per_gpu_batch_size: 64
  global_batch_size: 512

ema:
  enabled: true
  alpha: 0.999
```

---

## Comparison

| Feature | Diffusion Policy | Stable Diffusion | ManiFlow |
|---------|-----------------|------------------|----------|
| **Domain** | Robotics actions | Image generation | Robotics actions |
| **Input** | Camera images + language | Text | Point clouds + proprioception |
| **Denoiser** | Transformer | UNet | DiTX Transformer |
| **Conditioning** | CLIP visual-language | CLIP text | DP3 point cloud features |
| **Scheduler** | Flow matching (default) | DDPM or flow matching | Flow matching + consistency |
| **Output** | Action trajectory | Generated image | Action trajectory |
| **Inference steps** | Configurable | Configurable | ~10 (fast via consistency) |

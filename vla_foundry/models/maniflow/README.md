# ManiFlow

## ⚠️ IMPORTANT: Pre-Release Notice

**This code must be removed before any open source release of vla_foundry.**

This implementation is based on third-party research code that is not licensed for redistribution.

---

## Origin

This implementation is derived from the official ManiFlow policy repository:
- **Source**: https://github.com/geyan21/ManiFlow_Policy
- **Paper**: "ManiFlow: A Consistency Flow Matching Framework for Robotic Manipulation"
- **Authors**: Yan et al.

## Modifications and Adaptations

The original code has been adapted for integration into the vla_foundry framework with the following changes:

### Architecture Changes
1. **Modular Structure**: Refactored into separate files for better organization:
   - `maniflow.py`: Main ManiFlow model and training logic
   - `pointnet_encoder.py`: Point cloud encoder (DP3-style)
   - `transformer.py`: DiT-X transformer backbone
   - `flow_matching.py`: Flow matching and consistency training utilities

2. **Config System**: Integrated with vla_foundry's draccus-based configuration:
   - `ManiFlowParams` in `vla_foundry/params/model_params.py`
   - `DP3EncoderParams` for point cloud encoder configuration
   - `DiTXParams` for transformer configuration

3. **Data Pipeline**: Adapted to work with vla_foundry's robotics data pipeline:
   - Point cloud generation from depth images using `cuda_fps`
   - Integration with vla_foundry's normalization and augmentation systems
   - Support for multi-camera observations

## Configuration

Example configuration files are provided in:
- `vla_foundry/config_presets/models/maniflow.yaml` - Model architecture
- `vla_foundry/config_presets/data/maniflow.yaml` - Data and training config
- `vla_foundry/config_presets/training_jobs/maniflow_example.yaml` - Full training job

## Requirements

This implementation requires:
- Point cloud support (`use_point_cloud: true`)
- CUDA-accelerated Farthest Point Sampling (`cuda_fps` package)
- CLIP vision encoder for image processing
- Dataset with depth images and camera intrinsics/extrinsics

## Before Open Source Release

**Action Items:**
1. Remove this entire directory (`vla_foundry/models/maniflow/`)
2. Remove ManiFlow-related configuration presets
3. Remove point cloud-specific code if not used by other models
4. Remove references in model registry and documentation
5. Ensure no ManiFlow code appears in git history of public repo

**Alternative**: If ManiFlow becomes available under a compatible license, update this notice and add proper attribution.

---

## References

```bibtex
@inproceedings{yan2025maniflow,
  title={{ManiFlow}: A General Robot Manipulation Policy via Consistency Flow Training},
  author={Yan, Ge and Zhu, Jiyue and Deng, Yuquan and Yang, Shiqi and Qiu, Ri-Zhao and Cheng, Xuxin and Memmel, Marius and Krishna, Ranjay and Goyal, Ankit and Wang, Xiaolong and Fox, Dieter},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2025}
}
```

For questions or issues related to this implementation, contact Shun Iwase (siwase@tri.global).

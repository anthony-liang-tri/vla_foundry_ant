import numpy as np
import torch

NEOX_OFFSET = 50432  # neox tokens + slack text tokens
TARGET_MASK_INDIVIDUAL = 50400
TARGET_MASK_LEFT = 50300
SEQ_LEN = 1024
SHARD_SIZE = 8192
CHUNK_SIZE = SEQ_LEN + 1
MAX_CAPTION_TOKENS = 100
IMAGE_PATCH_PAD = 50305
SEQ_LEN_FT = 2048
CHUNK_SIZE_FT = SEQ_LEN_FT + 1

SPECIAL_STRS = ["<|img_patch|>", "<|a|>", "<|/a|>", "<|h|>", "<|/h|>"]
SPECIAL_TOKENS = [50277, 50278, 50279, 50280, 50281]
HUMAN_START = 50280
HUMAN_STOP = 50281
AGENT_START = 50278
AGENT_STOP = 50279

EMB_IMAGE_SIZE = 384
VQGAN_IMAGE_SIZE = 216
NUM_IMAGE_TOKENS = 729

TRIGGER_PROMPTS = {
    "captioning": {
        "text": "\nProvide a short image description. ",
        "neox": np.array([187, 11509, 504, 247, 2159, 2460, 5740, 15, 209]).astype(int),
    },
    "imagegen": {
        "text": "\nGenerate an image for the description. ",
        "neox": np.array([187, 40731, 271, 2460, 323, 253, 5740, 15, 209]).astype(int),
    },
}

TRIGGER_PROMPTS_TORCH = {
    "captioning": {
        "text": "\nProvide a short image description. ",
        "neox": torch.Tensor([187, 11509, 504, 247, 2159, 2460, 5740, 15, 209]),
    },
    "imagegen": {
        "text": "\nGenerate an image for the description. ",
        "neox": torch.Tensor([187, 40731, 271, 2460, 323, 253, 5740, 15, 209]),
    },
}


from torchvision.transforms import (
    Compose,
    Resize,
    InterpolationMode,
    ToTensor,
    Normalize,
)
import timm
import torch
import torch.nn as nn

from enum import IntEnum

class OutputType(IntEnum):
    EMBEDDING = 1
    TOKEN = 2

class TimmFeaturizer(nn.Module):
    def __init__(
        self, timm_string, img_size=EMB_IMAGE_SIZE, override_forward=True
    ) -> None:
        super().__init__()

        def load_model_helper():
            return timm.create_model(
                timm_string, pretrained=True, num_classes=0, img_size=img_size
            )  # .to(torch.bfloat16)


        if torch.distributed.is_initialized():
            # Need to download on only one rank at a time; start with rank 0, then do the rest.
            local_rank = world_info_from_env()[0]
            if local_rank == 0:
                self.model = load_model_helper()
            torch.distributed.barrier()
            if local_rank != 0:
                self.model = load_model_helper()
        else:
            self.model = load_model_helper()

        self.model.eval()
        self.img_size = img_size
        self.embed_dim = self.model.embed_dim

        model_data_cfg = timm.data.resolve_model_data_config(self.model)
        model_data_cfg["input_size"] = (3, img_size, img_size)

        if override_forward:
            self.model.forward = unpack_tuple(
                partial(
                    self.model.get_intermediate_layers,
                    n={len(self.model.blocks) - 2},
                )
            )

    @torch.inference_mode
    def forward(self, x):
        # [batch, num_patches, feature_dim]
        return self.model(x)

class SiglipFeaturizer(TimmFeaturizer):
    def __init__(self, img_size=EMB_IMAGE_SIZE, override_forward=True) -> None:
        super().__init__("vit_so400m_patch14_siglip_384", img_size, override_forward)


class DinoFeaturizer(TimmFeaturizer):
    def __init__(self, img_size=EMB_IMAGE_SIZE, override_forward=True) -> None:
        super().__init__(
            "vit_large_patch14_reg4_dinov2.lvd142m", img_size, override_forward
        )


class MovqganTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = get_movqgan_model("270M", pretrained=True)  # .to(torch.bfloat16)
        # self.model = torch.compile(self.model)
        self.model.eval()

    @torch.inference_mode
    def forward(self, x):
        return torch.flatten(self.model.tokenize(x).long(), start_dim=1) + NEOX_OFFSET


class PatchFeaturizer(nn.Module):
    def __init__(self, width=2048, patch_size=14) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.embed_dim = width

    def forward(self, x):
        return torch.flatten(self.conv1(x).permute(0, 2, 3, 1), start_dim=1, end_dim=2)

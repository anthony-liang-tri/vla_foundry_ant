from torchvision.transforms import (
    Compose,
    Resize,
    InterpolationMode,
    ToTensor,
    Normalize,
)

def get_image_transform(encoder):
    transform_mapper = {
        "siglip": SiglipTransform,
        "dino": DinoTransform,
        "movqgan": MoVQGANTransform,
        "patch": PatchTransform,
    }
    if encoder in transform_mapper:
        return transform_mapper[encoder]
    else:
        raise ValueError(f"{encoder} image transform not supported!")


class BaseTransform:
    def __init__(self, emb_image_size, interpolation_mode):
        self.emb_image_size = emb_image_size
        if interpolation_mode == "bicubic":
            self.interpolation = InterpolationMode.BICUBIC
        else:
            raise ValueError(f"{interpolation_mode} not supported!")

class SiglipTransform(BaseTransform):
    def __init__(self, emb_image_size, interpolation_mode):
        super().__init__(emb_image_size, interpolation_mode)
        self.transform = Compose(
            [
                Resize(
                    size=(self.emb_image_size, self.emb_image_size),
                    interpolation=self.interpolation,
                    max_size=None,
                    antialias=True,
                ),
                lambda x: x.convert("RGB"),
                ToTensor(),
                Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000)),
            ]
        )

class DinoTransform(BaseTransform):
    def __init__(self, emb_image_size, interpolation_mode):
        super().__init__(emb_image_size, interpolation_mode)
        self.transform = Compose(
            [
                Resize(
                    size=(self.emb_image_size, self.emb_image_size),
                    interpolation=self.interpolation,
                    max_size=None,
                    antialias=True,
                ),
                lambda x: x.convert("RGB"),
                ToTensor(),
                Normalize(mean=(0.4850, 0.4560, 0.4060), std=(0.2290, 0.2240, 0.2250)),
            ]
        )

class MoVQGANTransform(BaseTransform):
    def __init__(self, emb_image_size, interpolation_mode):
        super().__init__(emb_image_size, interpolation_mode)
        self.transform = Compose(
            [
                Resize(
                    size=(self.emb_image_size, self.emb_image_size),
                    interpolation=self.interpolation,
                    max_size=None,
                    antialias=True,
                ),
                lambda x: x.convert("RGB"),
                ToTensor(),
                lambda x: 2.0 * x - 1.0,
            ]
        )


class PatchTransform(BaseTransform):
    def __init__(self, emb_image_size, interpolation_mode):
        super().__init__(emb_image_size, interpolation_mode)
        self.transform = Compose(
            [
                Resize(
                    size=(self.emb_image_size, self.emb_image_size),
                    interpolation=self.interpolation,
                    max_size=None,
                    antialias=True,
                ),
                lambda x: x.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
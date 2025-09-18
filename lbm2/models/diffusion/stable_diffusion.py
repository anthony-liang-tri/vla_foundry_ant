import torch
from PIL import Image
from tqdm import tqdm

from lbm2.models.base_model import BaseModel
from lbm2.models.diffusion.noise_scheduler import NoiseScheduler
from lbm2.models.diffusion.unet import UNet
from lbm2.params.model_params import StableDiffusionParams


class StableDiffusion(BaseModel):
    def __init__(self, model_params: StableDiffusionParams, scheduler: NoiseScheduler, unet: UNet):
        super().__init__(model_params)
        self.scheduler = scheduler
        self.unet = unet
        # self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, image, attention_mask, noise):
        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.num_timesteps, (input_ids.shape[0],)).to(image.device)  # [bsz]
        # text_embeddings = self.text_encoder(input_ids).last_hidden_state
        noisy_images = self.scheduler.add_noise(image, noise, timesteps)  # [bsz, channels, h, w]
        predicted_direction = self.unet(noisy_images, timesteps)  # [bsz, channels, h, w]
        return predicted_direction

    @torch.no_grad()
    def generate(self, batch_size, device):
        images = torch.randn(
            batch_size,
            self.model_params.unet.in_channels,
            self.model_params.unet.image_size,
            self.model_params.unet.image_size,
        )
        images = images.to(device)

        for t in tqdm(range(self.scheduler.num_timesteps - 1, 0, -1)):
            # 1. predict noise model_output
            model_output = self.unet(images, t)

            # 2. compute previous image: x_t -> t_t-1
            images = self.scheduler.step(model_output, t, images)

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = [Image.fromarray(image) for image in images]
        return images

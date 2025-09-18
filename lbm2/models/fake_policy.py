import torch

from lbm2.models.base_model import BaseModel
from lbm2.models.model_outputs.llm_output import TransformerOutput
from lbm2.params.model_params import FakePolicyParams


class FakePolicy(BaseModel):
    def __init__(self, model_params: FakePolicyParams):
        super().__init__(model_params)
        # Create a random weight for the model so that it behaves like a model
        self.weights = torch.nn.Parameter(
            torch.randn(
                1,
            )
        )

    def forward(
        self,
        input_ids,
        image,
        actions,
        past_mask=None,
        proprioception=None,
        attention_mask=None,
        use_cache=True,
        future_mask=None,
    ):
        # Use the weight so that backward pass does not fail
        output = torch.randn_like(actions) * self.weights
        loss = torch.nn.functional.mse_loss(output, actions)
        return TransformerOutput(
            logits=output,
            loss=loss,
        )

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        image,
        actions,
        proprioception=None,
        attention_mask=None,
        num_inference_steps=None,
        past_mask=None,
    ):
        return actions

import torch
import torch.nn as nn

from lbm2.params.model_params import ModelParams


class BaseModelMeta(type):
    """Metaclass that automatically calls _post_init after __init__."""

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        instance.__init__(*args, **kwargs)

        # Call _post_init if it exists
        if hasattr(instance, "_post_init"):
            instance._post_init()

        return instance


class BaseModel(nn.Module, metaclass=BaseModelMeta):
    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.model_params = model_params

    def _post_init(self):
        """Called automatically after subclass initialization is complete."""
        if self.model_params.freeze:
            self.freeze_parameters()

    def freeze_parameters(self):
        """Freeze all parameters in the model by setting requires_grad to False."""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError

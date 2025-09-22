import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWithZLoss(CrossEntropyLoss):
    def __init__(
        self,
        eps: float = 1e-4,
        weight: Tensor = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target) + self.eps * torch.square(torch.logsumexp(input, dim=-1)).mean()


def masked_mse_loss(predicted_direction, target_direction, mask=None):
    """
    Compute the flow matching loss with masking.

    Args:
        predicted_direction: Predicted flow direction
        target_direction: Target flow direction from flow_sample
        mask: Mask of valid actions (should be broadcastable to the shape of predicted_direction)

    Returns:
        Masked MSE loss between predicted and target directions
    """
    if mask is None:
        return torch.nn.functional.mse_loss(predicted_direction, target_direction)

    # Compute element-wise squared error
    loss = (predicted_direction - target_direction) ** 2

    # Apply mask (assume mask is 1 for valid, 0 for invalid)
    # Depending on the input strategy (past given in the same sequence or separate),
    # the mask may be shorter or longer than the loss
    seq_len = min(mask.shape[1], loss.shape[1])
    loss = loss[:, -seq_len:] * mask[:, -seq_len:, None]

    # Compute mean over masked elements only
    # Add a small epsilon to avoid division by zero
    dim_shape = loss.shape[-1]
    mean_loss = loss.sum() / (mask.sum() * dim_shape + 1e-8)
    return mean_loss


def _ignore_mask(loss_fn):
    def wrapper(*args, mask=None, **kwargs):
        return loss_fn(*args, **kwargs)

    return wrapper


def get_loss_function(loss_function_type, hparams):
    """
    Get the appropriate loss function for the given type.
    The output loss function should take inputs and targets as arguments and also a mask
    but it doesn't need to use the mask. In that case, the mask can be ignored with the wrapper _ignore_mask.
    """
    if loss_function_type == "cross_entropy":
        if hparams.z_loss_coefficient != 0.0:
            loss = _ignore_mask(CrossEntropyLossWithZLoss(hparams.z_loss_coefficient))
        else:
            loss = _ignore_mask(torch.nn.CrossEntropyLoss())
    elif loss_function_type == "mse":
        loss = _ignore_mask(torch.nn.MSELoss())
    elif loss_function_type == "masked_mse":
        loss = masked_mse_loss
    else:
        raise ValueError(f"Loss function {loss_function_type} not supported.")

    return loss

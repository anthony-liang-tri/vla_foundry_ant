import logging
from torch import optim
from lbm2.file_utils import pt_load
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def create_optimizer(experiment_configs, model):
    named_parameters = list(model.named_parameters())
    no_decay_params = []  # to be potentially used later
    params = [p for n, p in named_parameters if p.requires_grad]

    if experiment_configs.optimizer == "adamw":
        optimizer = optim.AdamW(
            [
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": params, "weight_decay": experiment_configs.wd},
            ],
            lr=experiment_configs.lr,
            betas=(experiment_configs.beta1, experiment_configs.beta2),
            eps=experiment_configs.eps,
        )
    else:
        raise ValueError("Only adamw supported for now")

    return optimizer


def load_optimizer(checkpoint_path, use_fsdp, model, optimizer):
    optimizer_path = checkpoint_path.replace("checkpoint_", "optimizer_")
    optimizer_checkpoint = pt_load(optimizer_path, map_location="cpu")
    if "optimizer" in optimizer_checkpoint:
        osd = optimizer_checkpoint["optimizer"]
        if use_fsdp:
            osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=osd)
        optimizer.load_state_dict(osd)
        logging.info(f"=> resuming optimizer")
    else:
        logging.info(f"=> WARNING: not resuming optimizer.")

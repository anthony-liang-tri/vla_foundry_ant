from datetime import datetime

from lbm2.distributed import broadcast_object


def get_experiment_name(cfg):
    if cfg.name is None:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if cfg.distributed.use_distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(cfg, date_str)
        name = "-".join(
            [
                date_str,
                f"model_{cfg.model.type}",
                f"lr_{cfg.hparams.lr}",
                f"bsz_{cfg.hparams.global_batch_size}",
            ]
        )
    else:
        name = cfg.name

    # sanitize model name for filesystem / uri use
    name.replace("/", "-")
    object.__setattr__(cfg, "name", name)
    return name

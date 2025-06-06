from pathlib import Path
from datetime import datetime
from distributed import broadcast_object

def get_experiment_name(cfg):
    if cfg.name is None:
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if cfg.distributed.use_distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(cfg, date_str)
        cfg.name = "-".join(
            [
                date_str,
                f"model_{cfg.model.model_type}",
                f"lr_{cfg.experiment.lr}",
                f"bsz_{cfg.experiment.global_batch_size}",
            ]
        )
    
    # sanitize model name for filesystem / uri use
    return cfg.name.replace("/", "-")

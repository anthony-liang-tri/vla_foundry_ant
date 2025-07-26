import logging
from dataclasses import dataclass, field
from typing import Type

import draccus

from lbm2.data.utils import epochs_to_samples
from lbm2.file_utils import copy_to_temp_file
from lbm2.params.base_params import BaseParams
from lbm2.params.data_params import DataParams
from lbm2.params.distributed_params import DistributedParams
from lbm2.params.hyper_params import HyperParams
from lbm2.params.model_params import ModelParams


@dataclass(frozen=True)
class TrainExperimentParams(BaseParams):
    # Logging and remote sync
    name: str = field(default=None)
    save_path: str = field(default=None)
    wandb: bool = field(default=True)
    wandb_project_name: str = field(default="lbm2")
    log_every_n_steps: int = field(default=20)
    remote_sync: str = field(default=None)

    # Training
    total_train_samples: int = field(default=None)
    num_epochs: int = field(default=None)
    num_checkpoints: int = field(default=5)

    # Params Subclasses
    data: DataParams = field(default_factory=DataParams)
    distributed: DistributedParams = field(default_factory=DistributedParams)
    hparams: HyperParams = field(default_factory=HyperParams)
    model: ModelParams = field(default_factory=ModelParams)

    def __post_init__(self):
        super().__post_init__()
        self.data.init_shared_attributes(self)
        self.distributed.init_shared_attributes(self)
        self.hparams.init_shared_attributes(self)
        self.model.init_shared_attributes(self)
        if self.num_epochs is not None and self.total_train_samples is not None:
            raise ValueError("Set either num_epochs or total_train_samples, not both.")
        if self.num_epochs is not None:
            logging.info(
                f"Setting total_train_samples based on self.num_epochs={self.num_epochs} epochs. "
                "If you have already set total_train_samples, this will be ignored."
            )
            total_train_samples = epochs_to_samples(self.data.dataset_manifest, self.num_epochs)
            object.__setattr__(self, "total_train_samples", total_train_samples)
        self.check_asserts()

    def check_asserts(self):
        assert self.hparams.global_batch_size % self.distributed.world_size == 0
        assert (
            self.hparams.accum_freq * self.distributed.world_size * self.hparams.per_gpu_batch_size
            == self.hparams.global_batch_size
        )
        assert len(self.data.dataset_manifest) == len(self.data.dataset_modality)
        assert len(self.data.dataset_manifest) == len(self.data.dataset_weighting)
        assert self.total_train_samples is not None
        if self.num_epochs is not None:
            assert self.data.allow_multiple_epochs
        # This causes an error when loading from yaml. Commenting out for now.
        # if self.distributed.fsdp and not self.distributed.use_distributed:
        #     raise ValueError(f"--fsdp can only be specified in distributed mode.")


def load_params_from_yaml(params_class: Type[BaseParams], path: str) -> BaseParams:
    """
    Load a draccus params object from a yaml file with support for s3 paths.

    Warning: If loading from s3, the file will be copied to a temporary file and deleted after loading.
    This does not allow !include statements in the yaml files because those need to be relative to the file.
    Hopefully s3 configs do not have !include statements (they shouldn't).
    """
    # Need to copy to temp file because draccus doesn't support loading from s3.
    if path.startswith("s3"):
        with copy_to_temp_file(path) as temp_path:
            # Load the params
            params = draccus.load(params_class, temp_path)
    else:
        # Load the params from the local file so it can support !include statements.
        params = draccus.load(params_class, path)
    return params


def load_experiment_params_from_yaml(path: str) -> TrainExperimentParams:
    return load_params_from_yaml(TrainExperimentParams, path)


if __name__ == "__main__":
    path = "s3://tri-ml-datasets/scratch/sedrick.keh/sedrick/vlm_paligemma_3b/2025_07_04-01_38_18-model_vlm-lr_0.0001-bsz_128/config.yaml"
    print(load_experiment_params_from_yaml(path))

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Type

import yaml

from vla_foundry.data.utils import epochs_to_samples
from vla_foundry.file_utils import localize_paths, yaml_load
from vla_foundry.params.base_params import BaseParams
from vla_foundry.params.data_params import DataParams  # not from base_data_params so it loads registered params
from vla_foundry.params.distributed_params import DistributedParams
from vla_foundry.params.hyper_params import HyperParams
from vla_foundry.params.model_params import ModelParams


@dataclass(frozen=True)
class TrainExperimentParams(BaseParams):
    """
    Top-level, immutable configuration for a training experiment.
    """

    # -- Logging and remote sync
    # Optional explicit experiment name.
    # If `None`, a name will be generated at runtime (see `vla_foundry.utils.get_experiment_name`).
    name: str = field(default=None)

    # If resolve_configs is True, main.py will print the resolved config and stop.
    # The optional resolve_configs_path field will dump that printed output to {path}/resolved_config.yaml.
    resolve_configs: bool = field(default=False)
    resolve_configs_path: Optional[str] = field(default=None)

    # Optional base directory where the experiment folder is created. If `None`, defaults to `experiments/`.
    save_path: str = field(default=None)
    wandb: bool = field(default=True)
    wandb_entity: str = field(default=os.getenv("WANDB_ENTITY"))
    wandb_project_name: str = field(default="vla_foundry")
    wandb_tags: list[str] = field(default_factory=list)
    log_every_n_steps: int = field(default=20)
    log_level: str = field(default="INFO")
    # Optional path to S3 to which the experiment directory is synced.
    remote_sync: str = field(default=None)

    # --Training
    # total number of samples to train on. Mutually exclusive with `num_epochs`.
    total_train_samples: int = field(default=None)
    # Number of epochs over the input datasets. If set, it is converted to
    # `total_train_samples` using `epochs_to_samples`. Mutually exclusive with
    # `total_train_samples`.
    num_epochs: int = field(default=None)
    # Number of checkpoint windows the total budget is split into.
    num_checkpoints: int = field(default=5)
    max_checkpoint_limit: int = field(default=None)

    # --Validation
    total_val_samples: int = field(default=None)
    val_every_n_checkpoints: int = field(default=1)

    # --Params Subclasses
    data: DataParams = field(default_factory=DataParams)
    distributed: DistributedParams = field(default_factory=DistributedParams)
    hparams: HyperParams = field(default_factory=HyperParams)
    model: ModelParams = field(default_factory=ModelParams)

    def __post_init__(self):
        """
        Derive fields, initialize shared attributes, and validate consistency.
        """
        super().__post_init__()

        # Allow sub-params to read the full config and set shared/derived fields.
        self.init_shared_attributes(self)

        # Mutual exclusivity check for training budget specification.
        if self.num_epochs is not None and self.total_train_samples is not None:
            raise ValueError("Set either num_epochs or total_train_samples, not both.")

        # If epochs were specified, derive the total sample budget now.
        if self.num_epochs is not None:
            logging.info(
                f"Setting total_train_samples based on self.num_epochs={self.num_epochs} epochs. "
                "If you have already set total_train_samples, this will be ignored."
            )
            total_train_samples = epochs_to_samples(self.data.dataset_manifest, self.num_epochs)
            object.__setattr__(self, "total_train_samples", total_train_samples)

        self.check_asserts()

    def check_asserts(self):
        """
        Validate cross-field invariants for batch sizing and dataset config.
        """
        # Global batch must shard evenly across processes.
        assert self.hparams.global_batch_size % self.distributed.world_size == 0

        # Consistency between accumulation, per-GPU microbatch, and global batch.
        assert (
            self.hparams.accum_freq * self.distributed.world_size * self.hparams.per_gpu_batch_size
            == self.hparams.global_batch_size
        )

        # Dataset-related lists must align in length.
        assert len(self.data.dataset_manifest) == len(self.data.dataset_modality)
        assert len(self.data.dataset_manifest) == len(self.data.dataset_weighting)

        # Training budget must be resolved at this point.
        assert self.total_train_samples is not None

        # If epochs were requested, multiple passes must be allowed.
        if self.num_epochs is not None:
            assert self.data.allow_multiple_epochs

        # This check causes an error when loading from yaml due to load-time constraints.
        # Commenting out for now.
        # if self.distributed.fsdp and not self.distributed.use_distributed:
        #     raise ValueError(f"--fsdp can only be specified in distributed mode.")


def load_params_from_yaml(params_class: Type[BaseParams], path: str, localize_params: bool = False) -> BaseParams:
    """
    Load a draccus params object from a yaml file with support for s3 paths.

    Warning:
    If loading from s3, the file will be copied to a temporary file and deleted after loading.
    This does not allow !include statements in the yaml files because those need to be relative to the file.
    Hopefully s3 configs do not have !include statements (they shouldn't).

    Args:
        params_class: dataclass type to load.
        path: local filesystem path or S3 URI to the YAML file.
        localize_params: if True, first localize all paths in the config before loading.
    """
    if localize_params:
        config_dict = yaml_load(path)
        base_path, _ = os.path.split(path)
        yaml_dict = localize_paths(config_dict, base_path)

        # Create a temporary file
        fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", prefix="localized_config_")
        os.close(fd)  # Close the file descriptor
        with open(temp_file_path, "w") as f:
            yaml.dump(yaml_dict, f)
        path = temp_file_path

    # Use from_file method which handles unknown key stripping
    params = params_class.from_file(path)
    return params


def load_experiment_params_from_yaml(path: str, localize_params: bool = False) -> TrainExperimentParams:
    """
    Convenience wrapper to load `TrainExperimentParams` from YAML.
    If `localize_params` is True, first localize all paths in the config before loading.
    """
    return load_params_from_yaml(TrainExperimentParams, path, localize_params)

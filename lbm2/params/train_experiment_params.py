import logging
from dataclasses import dataclass, field
from typing import Type

import draccus

from lbm2.data.utils import epochs_to_samples
from lbm2.file_utils import copy_to_temp_file
from lbm2.params.base_data_params import DataParams
from lbm2.params.base_params import BaseParams

# Need to import these here to register the subclasses with draccus.
from lbm2.params.data_params import ImageCaptionDataParams, TextDataParams, TextUntokenizedDataParams  # noqa: F401
from lbm2.params.distributed_params import DistributedParams
from lbm2.params.hyper_params import HyperParams
from lbm2.params.model_params import ModelParams


@dataclass(frozen=True)
class TrainExperimentParams(BaseParams):
    """
    Top-level, immutable configuration for a training experiment.
    """

    # -- Logging and remote sync
    # Optional explicit experiment name.
    # If `None`, a name will be generated at runtime (see `lbm2.utils.get_experiment_name`).
    name: str = field(default=None)
    # Optional base directory where the experiment folder is created. If `None`, defaults to `experiments/`.
    save_path: str = field(default=None)
    wandb: bool = field(default=True)
    wandb_project_name: str = field(default="lbm2")
    log_every_n_steps: int = field(default=20)
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
        self.data.init_shared_attributes(self)
        self.distributed.init_shared_attributes(self)
        self.hparams.init_shared_attributes(self)
        self.model.init_shared_attributes(self)

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


def load_params_from_yaml(params_class: Type[BaseParams], path: str) -> BaseParams:
    """
    Load a draccus params object from a yaml file with support for s3 paths.

    Warning:
    If loading from s3, the file will be copied to a temporary file and deleted after loading.
    This does not allow !include statements in the yaml files because those need to be relative to the file.
    Hopefully s3 configs do not have !include statements (they shouldn't).

    Args:
        params_class: dataclass type to load.
        path: local filesystem path or S3 URI to the YAML file.
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
    """
    Convenience wrapper to load `TrainExperimentParams` from YAML.
    """
    return load_params_from_yaml(TrainExperimentParams, path)

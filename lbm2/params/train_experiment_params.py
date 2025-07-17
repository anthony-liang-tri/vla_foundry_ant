import os
import tempfile
import yaml
import draccus
from dataclasses import dataclass, field
from lbm2.params.data_params import DataParams
from lbm2.params.distributed_params import DistributedParams
from lbm2.params.hyper_params import HyperParams
from lbm2.params.model_params import ModelParams
from lbm2.params.base_params import BaseParams
from lbm2.file_utils import yaml_load

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
            logging.info(f"Setting total_train_samples based on self.num_epochs={self.num_epochs} epochs. If you have already set total_train_samples, this will be ignored.")
            total_train_samples = epochs_to_samples(self.data.dataset_manifest, self.num_epochs)
            object.__setattr__(self, 'total_train_samples', total_train_samples)
        self.check_asserts()

    def check_asserts(self):
        assert self.hparams.global_batch_size % self.distributed.world_size == 0
        assert self.hparams.accum_freq * self.distributed.world_size * self.hparams.per_gpu_batch_size == self.hparams.global_batch_size
        assert len(self.data.dataset_manifest) == len(self.data.dataset_modality)
        assert len(self.data.dataset_manifest) == len(self.data.dataset_weighting)
        assert self.total_train_samples is not None
        if self.num_epochs is not None:
            assert self.data.allow_multiple_epochs
        # This causes an error when loading from yaml. Commenting out for now.
        # if self.distributed.fsdp and not self.distributed.use_distributed:
        #     raise ValueError(f"--fsdp can only be specified in distributed mode.")

def load_params_from_yaml(path: str) -> TrainExperimentParams:
    curr_yaml = yaml_load(path)
    # Save curr_yaml to a temporary file and delete it after loading
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=True) as temp_file:
        yaml.dump(curr_yaml, temp_file)
        temp_file.flush()  # Ensure data is written
        return draccus.load(TrainExperimentParams, temp_file.name)

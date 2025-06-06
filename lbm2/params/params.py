import argparse
import logging
from dataclasses import dataclass, asdict, fields
from file_utils import json_load
from params.data_params import DataParams, add_data_params
from params.distributed_params import DistributedParams, add_distributed_params
from params.experiment_params import ExperimentParams, add_experiment_params
from params.model_params import ModelParams, add_model_params
from params.extra.vit_params import ViTParams, add_vit_params


@dataclass
class Params:
    name: str
    data: DataParams
    distributed: DistributedParams
    experiment: ExperimentParams
    model: ModelParams
    
    # extra
    vit: ViTParams

    def asdict(self):
        return asdict(self)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of experiment")
    add_data_params(parser)
    add_distributed_params(parser)
    add_experiment_params(parser)
    add_model_params(parser)
    add_vit_params(parser)
    args = parser.parse_args()
    return args

def get_params(args):
    cfg = Params(
        name=args.name,
        data=DataParams.from_args(args),
        distributed=DistributedParams.from_args(args),
        experiment=ExperimentParams.from_args(args),
        model=ModelParams.from_args(args),
        vit=ViTParams.from_args(args),
    )
    handle_listtype_params(cfg)
    return cfg

def load_params_from_json(params_path):
    raw = json_load(params_path)

    def filter_missing(dataclass, input_dict):
        """Filter a dict to only include keys that exist in the dataclass."""
        valid_fields = {f.name for f in fields(dataclass)}
        for field_name in valid_fields:
            if field_name not in input_dict:
                logging.warning(f"Missing key in config: '{field_name}'. Setting to None")
                input_dict[field_name] = None
        
        filtered = {}
        for k, v in input_dict.items():
            if k in valid_fields:
                filtered[k] = v
            else:
                logging.warning(f"Ignored unexpected key in config: '{k}'")
        
        return filtered
    
    return Params(
        name=raw["name"],
        data=DataParams(**filter_missing(DataParams, raw["data"])),
        distributed=DistributedParams(**filter_missing(DistributedParams, raw["distributed"])),
        experiment=ExperimentParams(**filter_missing(ExperimentParams, raw["experiment"])),
        model=ModelParams(**filter_missing(ModelParams, raw["model"])),
        vit=ViTParams(**filter_missing(ViTParams, raw["vit"])),
    )

def handle_listtype_params(cfg):
    object.__setattr__(cfg.data, 'dataset_manifest', cfg.data.dataset_manifest.split(','))
    object.__setattr__(cfg.data, 'dataset_modality', cfg.data.dataset_modality.split(','))
    if cfg.data.dataset_weighting is not None:
        weighting = [float(i) for i in cfg.data.dataset_weighting.split(',')]
        object.__setattr__(cfg.data, 'dataset_weighting', weighting)

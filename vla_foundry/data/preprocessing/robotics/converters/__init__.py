import ray

from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_params import PreprocessParams


def get_converter(source_type: str, cfg: PreprocessParams) -> BaseRoboticsConverter:
    if source_type == "spartan":
        from vla_foundry.data.preprocessing.robotics.converters.spartan import SpartanConverter

        return SpartanConverter(cfg)
    elif source_type == "lerobot":
        from vla_foundry.data.preprocessing.robotics.converters.lerobot import LeRobotConverter

        return LeRobotConverter(cfg)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

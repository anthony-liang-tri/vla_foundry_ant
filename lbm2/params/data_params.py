from dataclasses import dataclass, fields
from typing import List


def add_data_params(parser):
    parser.add_argument(
        "--dataset-manifest",
        type=str,
        default=None,
        help="S3 path to train dataset manifests. Strings separated by comma",
    )
    parser.add_argument(
        "--dataset-weighting",
        type=str,
        default=None,
        help="As many numbers as there are data sources. Strings separated by comma",
    )
    parser.add_argument(
        "--dataset-modality",
        type=str,
        default=None,
        help="As many strings as there are data sources. Strings separated by comma",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="webdataset",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--total-train-samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of dataloader workers per GPU."
    )

@dataclass(frozen=True)
class DataParams:
    dataset_manifest: List[str]
    dataset_weighting: List[float]
    dataset_modality: List[str]
    dataset_type: str
    tokenizer: str
    processor: str
    total_train_samples: int
    num_workers: int

    # These are defined in the add_params of other files but we use them here
    global_batch_size: int
    seq_len: int
    vocab_size: int
    seed: int

    @classmethod
    def from_args(cls, args):
        init_kwargs = {
            f.name: getattr(args, f.name)
            for f in fields(cls)
            if hasattr(args, f.name)
        }
        return cls(**init_kwargs)
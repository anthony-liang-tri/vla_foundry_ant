from dataclasses import dataclass, fields


def add_experiment_params(parser):
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="S3 path to save to",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="amp_bfloat16",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--per-gpu-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--accum-freq",
        type=int,
        default=None,
        help="Recommended to keep None. Auto-inferred. Use per-gpu-batch-size and global-batch-size instead."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
    )
    parser.add_argument(
        "--warmup",
        type=str,
        default='0.1',
        help='If >1, treat as number of steps. If <1, treat as percentage of run.',
    )
    parser.add_argument(
        "--lr-cooldown-end",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--force-min-lr",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0e-8
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "mse"]
    )
    parser.add_argument(
        "--z-loss-coefficient",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_false",
        dest="wandb",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="lbm2",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None
    )
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
    )

@dataclass(frozen=True)
class ExperimentParams:
    save_path: str
    precision: str
    global_batch_size: int
    per_gpu_batch_size: int
    accum_freq: int

    seed: int
    lr: float
    lr_scheduler: str
    warmup: str
    lr_cooldown_end: float
    force_min_lr: float
    optimizer: str
    wd: float
    beta1: float
    beta2: float
    eps: float
    loss_function: str
    z_loss_coefficient: float
    grad_clip_norm: float
    grad_checkpointing: bool
    torchcompile: bool
    wandb: bool
    wandb_project_name: str
    log_every_n_steps: int
    num_checkpoints: int
    remote_sync: str
    resume_from_checkpoint: str
    resume_weights_only: bool

    # These are defined in the add_params of other files but we use them here
    total_train_samples: int
    num_epochs: int

    @classmethod
    def from_args(cls, args):
        init_kwargs = {
            f.name: getattr(args, f.name)
            for f in fields(cls)
            if hasattr(args, f.name)
        }
        return cls(**init_kwargs)
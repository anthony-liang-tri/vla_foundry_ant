import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def const_lr_cooldown(
    optimizer,
    base_lr,
    warmup_length,
    steps,
    cooldown_steps,
    cooldown_power=1.0,
    cooldown_end_lr=0.0,
):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e / es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps, min_lr, force_min_lr):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr)
            lr = max(lr, force_min_lr)
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def create_scheduler(experiment_configs, optimizer):
    total_steps = experiment_configs.total_train_samples // experiment_configs.global_batch_size
    if float(experiment_configs.warmup) < 1:
        warmup = int(float(experiment_configs.warmup) * total_steps)
    else:
        warmup = int(experiment_configs.warmup)
    if experiment_configs.lr_scheduler == "cosine":
        scheduler = cosine_lr(
            optimizer,
            experiment_configs.lr,
            warmup,
            total_steps,
            experiment_configs.lr_cooldown_end,
            experiment_configs.force_min_lr,
        )
    elif args.lr_scheduler == "const":
        scheduler = const_lr(
            optimizer,
            experiment_configs.lr,
            warmup,
        )
    else:
        raise ValueError(f"Unknown scheduler, {experiment_configs.lr_scheduler}. Available options are: cosine, const.")
    return scheduler

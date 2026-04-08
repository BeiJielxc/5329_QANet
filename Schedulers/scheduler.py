import math

from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR


# ── Scheduler factories ──────────────────────────────────────────────────────

def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


class _WarmupLambda:
    """Picklable linear-warmup callable for LambdaLR."""
    def __init__(self, target_lr: float, warmup_steps: int):
        self.target_lr    = target_lr
        self.warmup_steps = warmup_steps

    def __call__(self, t: int) -> float:
        return self.target_lr * min((t + 1) / self.warmup_steps, 1.0)


class _WarmupCosineLambda:
    """Picklable linear-warmup + cosine-decay callable for LambdaLR.

    Phase 1 (t < warmup_steps):  lr linearly rises from 0 to target_lr.
    Phase 2 (t >= warmup_steps): lr follows cosine decay from target_lr
                                  down to eta_min over the remaining steps.
    """
    def __init__(self, target_lr: float, warmup_steps: int,
                 total_steps: int, eta_min: float = 1e-6):
        self.target_lr    = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.eta_min      = eta_min

    def __call__(self, t: int) -> float:
        if t < self.warmup_steps:
            return self.target_lr * (t + 1) / self.warmup_steps
        progress = (t - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.eta_min + (self.target_lr - self.eta_min) * cosine


def lambda_scheduler(optimizer, args):
    """Linear warmup for warmup_steps, then constant at learning_rate."""
    target_lr    = getattr(args, "learning_rate", 1e-3)
    warmup_steps = getattr(args, "warmup_steps",  1000)
    return LambdaLR(optimizer, lr_lambda=_WarmupLambda(target_lr, warmup_steps))


def warmup_cosine_scheduler(optimizer, args):
    """Linear warmup then cosine decay to eta_min.
    Pairs with Adam(lr=1.0): lambda output IS the effective lr.
    """
    target_lr    = getattr(args, "learning_rate", 1e-3)
    warmup_steps = getattr(args, "warmup_steps",  1000)
    total_steps  = getattr(args, "num_steps",     60000)
    eta_min      = getattr(args, "eta_min",       1e-6)
    return LambdaLR(
        optimizer,
        lr_lambda=_WarmupCosineLambda(target_lr, warmup_steps, total_steps, eta_min),
    )


# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":         cosine_scheduler,
    "step":           step_scheduler,
    "lambda":         lambda_scheduler,
    "warmup_cosine":  warmup_cosine_scheduler,
}

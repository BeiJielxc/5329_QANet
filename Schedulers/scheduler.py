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


class _WarmupConstantLR:
    """Picklable lr schedule: linear warmup then constant.

    For the first ``warmup_steps``, lr ramps linearly from 0 to ``lr``.
    After that, lr stays at ``lr``.  Works with Adam (base_lr=1.0) so
    that effective_lr = base_lr * factor = factor.
    """
    def __init__(self, lr, warmup_steps=1000):
        self.lr = lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        return self.lr


def lambda_scheduler(optimizer, args):
    """LambdaLR with warmup + constant factor — uses args.learning_rate
    so that Adam (base_lr=1.0) gets effective lr = learning_rate."""
    lr = getattr(args, "learning_rate", 1e-3)
    warmup = getattr(args, "warmup_steps", 1000)
    return LambdaLR(optimizer, lr_lambda=_WarmupConstantLR(lr, warmup))


# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
}

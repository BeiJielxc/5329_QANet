import os
import sys
import json
import math
import inspect
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401
import torch
import torch.nn as nn

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

PROJECT_ROOT = "/root/autodl-tmp/sandbox/assignment1_2026_isolated"
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "_exp", "exp3_runtime_fixed")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, "/root/autodl-tmp/sandbox")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Working directory:", os.getcwd(), flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

from TrainTools.train import train
import TrainTools.train as train_module
import TrainTools.train_utils as train_utils
from EvaluateTools.evaluate import evaluate
from Data import SQuADDataset, load_word_char_mats, make_loader  # noqa: F401
from Models import QANet
from Models.Normalizations import normalizations
from Models.Normalizations import normalization as _norm_module
from Optimizers import optimizers
from Optimizers.adam import Adam

TRAIN_PARAM_NAMES = set(inspect.signature(train).parameters)
EVAL_PARAM_NAMES = set(inspect.signature(evaluate).parameters)

BASELINE_CONFIG = dict(
    batch_size=32,
    num_steps=60000,
    checkpoint=200,
    early_stop=30,
    seed=42,
    optimizer_name="adam",
    scheduler_name="lambda",
    learning_rate=1e-3,
    beta1=0.8,
    beta2=0.999,
    eps=1e-7,
    weight_decay=3e-6,
    warmup_steps=1000,
    ema_decay=0.9999,
    d_model=128,
    dropout=0.1,
    dropout_char=0.05,
)

EVAL_CONFIG = dict(
    dev_npz="_data/dev.npz",
    word_emb_json="_data/word_emb.json",
    char_emb_json="_data/char_emb.json",
    dev_eval_json="_data/dev_eval.json",
    d_model=128,
)

SEEDS = [42, 13, 7]
print("Shared config ready.", flush=True)


def patched_adam(params, args):
    return Adam(
        params=params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=getattr(args, "eps", 1e-7),
        weight_decay=args.weight_decay,
    )


optimizers["adam"] = patched_adam
print("Patched optimizer registry: adam now uses args.learning_rate directly.", flush=True)


def fixed_train_single_epoch(
    model,
    optimizer,
    scheduler,
    data_iter,
    steps,
    grad_clip,
    loss_fn,
    device,
    global_step: int = 0,
) -> float:
    model.train()
    loss_list = []
    for _ in train_utils.tqdm(range(steps), total=steps):
        optimizer.zero_grad(set_to_none=True)
        Cwid, Ccid, Qwid, Qcid, y1, y2, _ = next(data_iter)
        Cwid, Ccid = Cwid.to(device), Ccid.to(device)
        Qwid, Qcid = Qwid.to(device), Qcid.to(device)
        y1, y2 = y1.to(device), y2.to(device)
        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        loss = loss_fn(p1, p2, y1, y2)
        loss_list.append(float(loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
    mean_loss = float(np.mean(loss_list))
    print(f"STEP {global_step + steps:8d}  loss {mean_loss:8f}\n", flush=True)
    return mean_loss


train_utils.train_single_epoch = fixed_train_single_epoch
train_module.train_single_epoch = fixed_train_single_epoch
print("Patched training loop: backward -> grad clip -> optimizer step.", flush=True)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        n = len(self.normalized_shape)
        dims = tuple(range(-n, 0))
        rms = torch.sqrt(torch.mean(x**2, dim=dims, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class IdentityNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


normalizations["rms_norm"] = RMSNorm
normalizations["identity"] = IdentityNorm
_orig_get_norm = _norm_module.get_norm


def _patched_get_norm(name, d_model, length, num_groups=8):
    if name == "rms_norm":
        return RMSNorm([d_model, 1])
    if name == "identity":
        return IdentityNorm()
    return _orig_get_norm(name, d_model, length, num_groups=num_groups)


_norm_module.get_norm = _patched_get_norm
import Models.encoder as _enc_module

_enc_module.get_norm = _patched_get_norm
print("Registered normalizations:", list(normalizations.keys()), flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DIAG_BATCHES = 50


def paired_t_test(a, b):
    if scipy_stats is not None:
        t_stat, p_value = scipy_stats.ttest_rel(a, b)
        return float(t_stat), float(p_value), "scipy"
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diffs)
    if n < 2:
        raise ValueError("need at least 2 paired samples")
    mean_diff = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return 0.0, 1.0, "fallback"
    t_stat = mean_diff / (sd / math.sqrt(n))
    return float(t_stat), float("nan"), "fallback"


exp3_groups = [
    ("A_layer_norm", "layer_norm"),
    ("B_rms_norm", "rms_norm"),
    ("C_identity", "identity"),
]

os.makedirs(OUTPUT_ROOT, exist_ok=True)
exp3_results = {}
for group_name, norm in exp3_groups:
    for seed in SEEDS:
        run_tag = f"exp3_{group_name}_seed{seed}"
        save_dir = os.path.join(OUTPUT_ROOT, run_tag)
        log_dir = os.path.join(OUTPUT_ROOT, run_tag, "log")
        print("\n" + "=" * 60, flush=True)
        print(f"Experiment 3 | {group_name} | seed={seed} | norm_name={norm}", flush=True)
        print("=" * 60 + "\n", flush=True)
        train_kwargs = {k: v for k, v in BASELINE_CONFIG.items() if k in TRAIN_PARAM_NAMES and k != "seed"}
        results = train(
            **train_kwargs,
            seed=seed,
            save_dir=save_dir,
            log_dir=log_dir,
            norm_name=norm,
        )
        exp3_results[run_tag] = {
            "group": group_name,
            "norm": norm,
            "seed": seed,
            "best_f1": results["best_f1"],
            "best_em": results["best_em"],
            "history": results["history"],
        }
        with open(os.path.join(OUTPUT_ROOT, "results.json"), "w") as f:
            json.dump(
                {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in exp3_results.items()},
                f,
                indent=2,
            )
print("\nExperiment 3 training complete.", flush=True)

exp3_eval = {}
for group_name, norm in exp3_groups:
    f1_list, em_list = [], []
    for seed in SEEDS:
        run_tag = f"exp3_{group_name}_seed{seed}"
        save_dir = os.path.join(OUTPUT_ROOT, run_tag)
        log_dir = os.path.join(OUTPUT_ROOT, run_tag, "log")
        eval_kwargs = {k: v for k, v in EVAL_CONFIG.items() if k in EVAL_PARAM_NAMES}
        metrics = evaluate(
            **eval_kwargs,
            save_dir=save_dir,
            log_dir=log_dir,
            norm_name=norm,
        )
        exp3_eval[run_tag] = metrics
        f1_list.append(float(metrics["f1"]))
        em_list.append(float(metrics["exact_match"]))

    mean_f1 = float(np.mean(f1_list))
    std_f1 = float(np.std(f1_list, ddof=1))
    mean_em = float(np.mean(em_list))
    std_em = float(np.std(em_list, ddof=1))
    print(
        f"[{group_name}] F1 = {mean_f1:.4f} +/- {std_f1:.4f}, "
        f"EM = {mean_em:.4f} +/- {std_em:.4f}",
        flush=True,
    )

baseline = [exp3_eval[f"exp3_A_layer_norm_seed{seed}"]["f1"] for seed in SEEDS]
for group_name, _ in exp3_groups[1:]:
    vals = [exp3_eval[f"exp3_{group_name}_seed{seed}"]["f1"] for seed in SEEDS]
    t_stat, p_value, engine = paired_t_test(vals, baseline)
    print(f"[{group_name} vs A_layer_norm] paired t-test ({engine}): t={t_stat:.4f}, p={p_value}", flush=True)

with open(os.path.join(OUTPUT_ROOT, "exp3_eval.json"), "w") as f:
    json.dump(exp3_eval, f, indent=2)

print("Experiment 3 evaluation complete.", flush=True)

import os
import sys
import json
import math
import argparse
import numpy as np
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
sys.path.insert(0, PROJECT_ROOT)

from Data import SQuADDataset, load_word_char_mats
from Models import QANet
from Models.Normalizations import normalizations
from Models.Normalizations import normalization as _norm_module
from EvaluateTools.evaluate import run_eval, load_dev_eval, DEVICE
from Losses import losses

BASELINE_CONFIG = dict(
    para_limit=400,
    ques_limit=50,
    char_limit=16,
    d_model=128,
    num_heads=8,
    glove_dim=300,
    char_dim=64,
    dropout=0.1,
    dropout_char=0.05,
    pretrained_char=False,
)

EVAL_CONFIG = dict(
    dev_npz="_data/dev.npz",
    word_emb_json="_data/word_emb.json",
    char_emb_json="_data/char_emb.json",
    dev_eval_json="_data/dev_eval.json",
    batch_size=8,
    test_num_batches=-1,
    loss_name="qa_nll",
)

SEEDS = [42, 13, 7]
EXP3_GROUPS = [
    ("A_layer_norm", "layer_norm"),
    ("B_rms_norm", "rms_norm"),
    ("C_identity", "identity"),
]


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


def evaluate_checkpoint(save_dir, log_dir, norm_name):
    os.makedirs(log_dir, exist_ok=True)
    args = argparse.Namespace(
        **BASELINE_CONFIG,
        **{k: v for k, v in EVAL_CONFIG.items() if k in {"dev_npz", "word_emb_json", "char_emb_json", "dev_eval_json"}},
        norm_name=norm_name,
    )
    word_mat, char_mat = load_word_char_mats(args)
    model = QANet(word_mat, char_mat, args).to(DEVICE)
    ckpt = torch.load(os.path.join(save_dir, "model.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    dev_eval = load_dev_eval(args)
    dev_dataset = SQuADDataset(EVAL_CONFIG["dev_npz"])
    metrics, ans = run_eval(
        model,
        dev_dataset,
        dev_eval,
        num_batches=EVAL_CONFIG["test_num_batches"],
        batch_size=EVAL_CONFIG["batch_size"],
        use_random_batches=False,
        device=DEVICE,
        loss_fn=losses[EVAL_CONFIG["loss_name"]],
    )
    with open(os.path.join(log_dir, "answers.json"), "w") as f:
        json.dump(ans, f)
    return {
        "f1": float(metrics["f1"]),
        "exact_match": float(metrics["exact_match"]),
        "loss": float(metrics["loss"]),
    }


def main():
    results_path = os.path.join(OUTPUT_ROOT, "results.json")
    with open(results_path) as f:
        train_results = json.load(f)

    eval_results = {}
    summary = {}
    for group_name, norm_name in EXP3_GROUPS:
        f1_list, em_list = [], []
        for seed in SEEDS:
            run_tag = f"exp3_{group_name}_seed{seed}"
            save_dir = os.path.join(OUTPUT_ROOT, run_tag)
            log_dir = os.path.join(save_dir, "log_eval_fixed")
            metrics = evaluate_checkpoint(save_dir, log_dir, norm_name)
            eval_results[run_tag] = {
                "group": group_name,
                "norm": norm_name,
                "seed": seed,
                **metrics,
                "best_f1": train_results[run_tag]["best_f1"],
                "best_em": train_results[run_tag]["best_em"],
            }
            f1_list.append(metrics["f1"])
            em_list.append(metrics["exact_match"])
            print(
                f'{run_tag}: F1={metrics["f1"]:.4f}, '
                f'EM={metrics["exact_match"]:.4f}, '
                f'loss={metrics["loss"]:.4f}',
                flush=True,
            )
        summary[group_name] = {
            "norm": norm_name,
            "mean_f1": float(np.mean(f1_list)),
            "std_f1": float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0,
            "mean_em": float(np.mean(em_list)),
            "std_em": float(np.std(em_list, ddof=1)) if len(em_list) > 1 else 0.0,
            "f1_values": f1_list,
            "em_values": em_list,
        }

    baseline = summary["A_layer_norm"]["f1_values"]
    comparisons = {}
    for group_name in ["B_rms_norm", "C_identity"]:
        t_stat, p_value, engine = paired_t_test(summary[group_name]["f1_values"], baseline)
        comparisons[group_name] = {
            "vs": "A_layer_norm",
            "metric": "f1",
            "t_stat": t_stat,
            "p_value": p_value,
            "engine": engine,
        }

    payload = {
        "eval_results": eval_results,
        "summary": summary,
        "comparisons": comparisons,
    }
    out_path = os.path.join(OUTPUT_ROOT, "evaluation_summary_fixed.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved summary to", out_path, flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

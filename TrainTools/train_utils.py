"""
train_utils.py — Low-level training utilities used by train().
"""

import copy
import os

import numpy as np
import torch
from tqdm import tqdm


class EMA:
    """Exponential Moving Average of model parameters.

    Uses TF-style dynamic decay so EMA is valid from step 1:
        effective_decay = min(decay, (1 + step) / (10 + step))

    At step 1:     effective_decay ≈ 0.18  (mostly current weights)
    At step 1000:  effective_decay ≈ 0.99
    At step 10000: effective_decay ≈ 0.9999 (fully converged to target)

    Usage:
        ema = EMA(model, decay=0.9999)
        ema.update(model)          # after each optimizer step
        with ema.apply(model):     # eval with EMA weights
            metrics = run_eval(model, ...)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.step  = 0
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)

    def apply(self, model: torch.nn.Module):
        """Context manager: temporarily replace model params with EMA shadow."""
        return _EMAContext(self, model)


class _EMAContext:
    def __init__(self, ema: EMA, model: torch.nn.Module):
        self.ema   = ema
        self.model = model
        self.backup = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.ema.shadow[name])

    def __exit__(self, *_):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


def train_single_epoch(model, optimizer, scheduler, data_iter,
                       steps, grad_clip, loss_fn, device,
                       global_step: int = 0,
                       ema: EMA = None) -> float:
    """
    Run one block of `steps` training iterations consuming from `data_iter`.
    Returns the mean loss over this block.
    """
    model.train()
    loss_list = []

    for _ in tqdm(range(steps), total=steps):
        optimizer.zero_grad(set_to_none=True)

        Cwid, Ccid, Qwid, Qcid, y1, y2, _ = next(data_iter)
        Cwid, Ccid = Cwid.to(device), Ccid.to(device)
        Qwid, Qcid = Qwid.to(device), Qcid.to(device)
        y1, y2     = y1.to(device),   y2.to(device)

        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        loss   = loss_fn(p1, p2, y1, y2)
        loss_list.append(float(loss.item()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

    mean_loss = float(np.mean(loss_list))
    print(f"STEP {global_step + steps:8d}  loss {mean_loss:8f}\n")
    return mean_loss


def save_checkpoint(save_dir, ckpt_name, model, optimizer, scheduler,
                    step, best_f1, best_em, config, ema=None):
    """Save model, optimizer, scheduler state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step":            step,
        "best_f1":         best_f1,
        "best_em":         best_em,
        "config":          config,
    }
    if ema is not None:
        payload["ema_state"] = ema.shadow
    torch.save(payload, os.path.join(save_dir, ckpt_name))

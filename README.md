# Assignment 1 — QANet

**COMP5329 Deep Learning — University of Sydney, Semester 1 2026**

This repository contains a PyTorch implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf) for extractive question answering on SQuAD v1.1. The distributed code contains **intentional bugs** — your task is to find and fix them all. See the assignment spec for details.

The entire pipeline (download, preprocess, train, evaluate) is driven from a single notebook: **`assignment1.ipynb`**.

---

## Getting Started on Google Colab

### 1 — Clone the repo into Google Drive

Open a **new notebook** at [colab.research.google.com](https://colab.research.google.com) and run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
REPO_URL     = "https://github.com/usyddeeplearning/Assignment1--2026.git"
PROJECT_ROOT = "/content/drive/MyDrive/Assignment1--2026"

if not os.path.exists(PROJECT_ROOT):
    !git clone {REPO_URL} {PROJECT_ROOT}
    print("Done.")
else:
    print("Already cloned — skipping.")
```

Close this temporary notebook when done.

### 2 — Open the assignment notebook

In Google Drive, navigate to `MyDrive/Assignment1--2026/` and double-click **`assignment1.ipynb`** to open it in Colab. Run cells in order from Section 0 downward — the notebook handles dependency installation, data download, preprocessing, training, and evaluation.

> Your files live on Google Drive, so they persist across Colab sessions. You only need to clone once.

### 3 — Pulling updates

If the repo is updated after you've cloned it, open a Colab cell and run:

```python
!cd /content/drive/MyDrive/Assignment1--2026 && git pull
```

Be careful — if you've edited files that were also updated upstream, you may get merge conflicts. Commit or back up your changes first.

---

## Improvements Beyond Bug Fixes

After fixing all bugs, the following enhancements were made to align the implementation with the original QANet paper (Yu et al., 2018).

### 1 — Stochastic Depth (Layer Dropout)

**File:** `Models/encoder.py` — `EncoderBlock`

The paper applies stochastic depth within each encoder block: sublayer *l* has survival probability *p_l = 1 − (l / L)(1 − p_L)* with *p_L = 0.9*. During training, each sublayer (conv / self-attention / FFN) is randomly skipped with this probability schedule — deeper sublayers are dropped more often. Inverted scaling (dividing by *p_l* when the layer survives) keeps the expected output magnitude consistent at test time.

Previously, the code used element-wise dropout with an incorrect formula. The new implementation operates at the **layer level**: either the entire sublayer output is used (scaled) or skipped (identity residual).

### 2 — Exponential Moving Average (EMA)

**Files:** `TrainTools/train_utils.py` (EMA class), `TrainTools/train.py` (integration), `EvaluateTools/evaluate.py` (checkpoint loading)

The paper states: *"Exponential moving average is applied on all trainable variables with a decay rate 0.9999."* The `EMA` class maintains shadow copies of all trainable parameters, updated after every optimizer step as: *shadow = 0.9999 × shadow + 0.0001 × param*. During evaluation, the shadow parameters are swapped in; during training, the original parameters are restored. The EMA state is also saved in checkpoints and used by the standalone evaluation script.

### 3 — Model Dimensions: d_model=128, char_dim=200

**Files:** `TrainTools/train.py`, `EvaluateTools/evaluate.py`, `Tools/preproc.py`

The paper specifies *"the number of filters is d = 128"* and *"each character is represented as a trainable vector of dimension p2 = 200"*. The previous defaults (`d_model=96`, `char_dim=64`) have been updated to match the paper. This increases the model's capacity: the hidden size in all encoder blocks is now 128, and character embeddings are 200-dimensional. Note: after changing `char_dim`, you must **re-run the preprocess step** to regenerate character embeddings with the new dimension.

### 4 — Learning Rate Warmup

**File:** `Schedulers/scheduler.py`

The paper states: *"inverse exponential increase from 0.0 to 0.001 in the first 1000 steps, and then maintain a constant learning rate."* The lambda scheduler now implements a linear warmup from 0 to `learning_rate` over the first 1000 steps, then holds constant. This stabilises training in the early phase when attention weights are still random. The `_WarmupConstantLR` callable class is picklable for checkpoint serialisation.

### 5 — Early Stopping Based on F1 + Best-Only Checkpoint Saving

**File:** `TrainTools/train.py`

Previously, early stopping required **both** F1 and EM to drop below their historical best, making it almost impossible to trigger. The checkpoint was also saved unconditionally at every evaluation — so the final checkpoint could be a degraded model rather than the best one.

Now, early stopping is based solely on dev F1: if F1 does not improve for `early_stop` consecutive evaluations (default 10 × 200 = 2000 steps), training stops. The checkpoint is only saved when F1 improves, so `model.pt` always contains the best-performing model.

### 6 — Dev Evaluation Coverage Increased

**File:** `assignment1.ipynb` (Cell 10)

`test_num_batches` increased from 150 to 500 (~38% of the dev set). This reduces variance in the dev F1 estimate, making early stopping decisions and best-model selection more reliable, with only a modest increase in evaluation time.

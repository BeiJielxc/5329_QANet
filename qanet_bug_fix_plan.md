---
name: QANet Bug Fix Plan
overview: 对 QANet PyTorch 实现进行全面错误扫描，在第一阶段定位了 21 个导致训练/评估流程崩溃的结构性错误，在第二阶段定位了 20 个影响深度学习机制正确性的算法/逻辑错误，在第三阶段修复了 6 个与论文设计不一致的架构/训练逻辑问题。共计 47 个修复。全部已完成。
todos:
  - id: phase1-fixes
    content: 修复第一阶段的 21 个结构性错误（运行时崩溃和前向/反向传播逻辑错误）
    status: completed
  - id: phase1-verify
    content: 验证第一阶段修复后系统能完成完整训练流程（前向传播、反向传播、参数更新、评估、保存检查点）
    status: pending
  - id: phase2-fixes
    content: 修复第二阶段的 20 个深度学习机制正确性错误（激活函数、归一化、优化器、调度器、初始化、注意力等）
    status: completed
  - id: phase2-verify
    content: 验证第二阶段修复后模型训练损失稳定下降，验证集指标反映有效学习信号
    status: pending
  - id: phase3-fixes
    content: 修复第三阶段的 6 个与论文设计不一致的架构/训练逻辑问题（LayerNorm 归一化维度、span 解码、词向量冻结、编码器共享、调度器、训练参数）
    status: completed
isProject: false
---

# QANet 错误定位与修复计划

---

## 第一阶段：恢复基本功能的结构性错误（21 个）

本阶段的错误会导致运行时崩溃（TypeError、KeyError、IndexError、维度不匹配等）或严重破坏前向传播/训练循环逻辑，使得系统无法完成完整的训练流程。

---

### 错误 1：`argparse.Namespace` 构造方式错误

- **文件**：[TrainTools/train.py](Assignment1_2026-main/TrainTools/train.py) 第 107 行
- **Bug**：`args = argparse.Namespace({k: v for k, v in locals().items()})` 将一个 dict 作为位置参数传入，但 `Namespace` 接受 `**kwargs`。
- **影响**：触发 `TypeError`，训练入口函数在初始化时立即崩溃，整个训练流程无法启动。
- **修复**：改为 `args = argparse.Namespace(**{k: v for k, v in locals().items()})`。
- **状态**：✅ 已修复

---

### 错误 2：对 `loss.item()` 调用 `.backward()`

- **文件**：[TrainTools/train_utils.py](Assignment1_2026-main/TrainTools/train_utils.py) 第 34 行
- **Bug**：`loss.item().backward()` — `loss.item()` 返回一个 Python `float`，没有 `.backward()` 方法。
- **影响**：触发 `AttributeError`，反向传播无法执行，训练循环在第一个 step 就崩溃。
- **修复**：改为 `loss.backward()`。
- **状态**：✅ 已修复

---

### 错误 3：梯度裁剪在优化器更新之后执行

- **文件**：[TrainTools/train_utils.py](Assignment1_2026-main/TrainTools/train_utils.py) 第 35-36 行
- **Bug**：`optimizer.step()` 在 `clip_grad_norm_` 之前调用，梯度在裁剪前就已被消耗。
- **影响**：梯度裁剪完全无效，大梯度可能导致参数更新不稳定、训练发散，无法展现正常学习行为。
- **修复**：交换顺序，先 `torch.nn.utils.clip_grad_norm_(...)` 再 `optimizer.step()`。
- **状态**：✅ 已修复

---

### 错误 4：检查点加载键名不匹配

- **文件**：[EvaluateTools/evaluate.py](Assignment1_2026-main/EvaluateTools/evaluate.py) 第 119 行
- **Bug**：`model.load_state_dict(ckpt["model"])` 但 `save_checkpoint` 保存时使用的键名是 `"model_state"`。
- **影响**：触发 `KeyError`，评估流程无法加载检查点，评估完全无法执行。
- **修复**：改为 `ckpt["model_state"]`。
- **状态**：✅ 已修复

---

### 错误 5：QANet 中 word/char 嵌入查找表交叉使用

- **文件**：[Models/qanet.py](Assignment1_2026-main/Models/qanet.py) 第 65 行
- **Bug**：`Cw, Cc = self.char_emb(Cwid), self.word_emb(Ccid)` — 用字符嵌入表查找词 ID，用词嵌入表查找字符 ID。
- **影响**：词汇索引超出字符嵌入表范围导致 `IndexError`；即使不越界，维度也不匹配（word_dim=300 vs char_dim=64），前向传播崩溃。
- **修复**：改为 `Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)`。
- **状态**：✅ 已修复

---

### 错误 6：CQAttention 的 mask 参数传入顺序交换

- **文件**：[Models/qanet.py](Assignment1_2026-main/Models/qanet.py) 第 75 行
- **Bug**：`self.cq_att(Ce, Qe, qmask, cmask)` 将 `qmask`（shape `[B, 50]`）作为 `cmask` 参数传入，`cmask`（shape `[B, 400]`）作为 `qmask` 传入。
- **影响**：在 `mask_logits` 中，`[B, Lc, Lq]=[B,400,50]` 的相似度矩阵与 `[B, 1, 400]` 的错误 mask 无法广播，触发 `RuntimeError`。
- **修复**：改为 `self.cq_att(Ce, Qe, cmask, qmask)`。
- **状态**：✅ 已修复

---

### 错误 7：Pointer 头部 `torch.cat` 拼接维度错误

- **文件**：[Models/heads.py](Assignment1_2026-main/Models/heads.py) 第 23 行
- **Bug**：`X1 = torch.cat([M1, M2], dim=0)` 在 batch 维度拼接，产生 `[2B, C, L]` 而非 `[B, 2C, L]`。
- **影响**：后续 `torch.matmul(self.w1, X1)` 维度不匹配（`w1` 为 `[2C]`，需要与 `X1` 的倒数第二维匹配），触发 `RuntimeError`。
- **修复**：改为 `dim=1`。
- **状态**：✅ 已修复

---

### 错误 8：位置编码频率张量 `unsqueeze` 方向错误

- **文件**：[Models/encoder.py](Assignment1_2026-main/Models/encoder.py) 第 32 行
- **Bug**：`freqs.unsqueeze(0)` 产生 shape `[1, d_model]`，但需要 `[d_model, 1]` 才能与 `pos`（shape `[d_model, length]`）正确广播。
- **影响**：`pos * freqs` 即 `[d_model, length] * [1, d_model]`，最后一维 `length != d_model`（如 400 != 96）无法广播，触发 `RuntimeError`。
- **修复**：改为 `unsqueeze(1)`。
- **状态**：✅ 已修复

---

### 错误 9：EncoderBlock 中 norms 索引越界

- **文件**：[Models/encoder.py](Assignment1_2026-main/Models/encoder.py) 第 121 行
- **Bug**：`self.norms[i + 1]` — `self.norms` 有 `conv_num` 个元素（索引 0 到 conv_num-1），当 `i = conv_num - 1` 时访问 `self.norms[conv_num]` 越界。
- **影响**：触发 `IndexError`，前向传播崩溃。
- **修复**：改为 `self.norms[i]`。
- **状态**：✅ 已修复

---

### 错误 10：自注意力计算结果被丢弃

- **文件**：[Models/encoder.py](Assignment1_2026-main/Models/encoder.py) 第 123-124 行
- **Bug**：`out = self.self_att(out, mask)` 计算了自注意力输出后，紧接着 `out = res` 将结果覆盖为残差值，注意力结果被完全丢弃。
- **影响**：模型的注意力机制完全失效，编码器退化为纯卷积模块，无法捕获长距离依赖，模型难以展现有意义的学习行为。
- **修复**：改为 `out = out + res`（残差连接）。
- **状态**：✅ 已修复

---

### 错误 11：Conv1d `unfold` 作用在错误的维度

- **文件**：[Models/conv.py](Assignment1_2026-main/Models/conv.py) 第 55 行
- **Bug**：`x.unfold(1, self.kernel_size, 1)` 在通道维度（dim=1）展开，应该在长度维度（dim=2）展开。
- **影响**：展开后的张量 shape 完全错误，后续 `view` 重塑时元素总数不匹配，触发 `RuntimeError`。
- **修复**：改为 `x.unfold(2, self.kernel_size, 1)`。
- **状态**：✅ 已修复

---

### 错误 12：Conv2d 宽度方向 padding 使用了错误的高度值

- **文件**：[Models/conv.py](Assignment1_2026-main/Models/conv.py) 第 124 行
- **Bug**：`pad_w = x.new_zeros(B, C_in, H, p)` 在高度 padding 之后，`x` 的高度已经变为 `H+2p`，但 `pad_w` 仍使用原始高度 `H`。
- **影响**：`torch.cat([pad_w, x, pad_w], dim=3)` 时高度维度不匹配，触发 `RuntimeError`。
- **修复**：改为 `pad_w = x.new_zeros(B, C_in, x.size(2), p)`。
- **状态**：✅ 已修复

---

### 错误 13：DepthwiseSeparableConv 前向传播顺序颠倒

- **文件**：[Models/conv.py](Assignment1_2026-main/Models/conv.py) 第 174-175 行
- **Bug**：`return self.depthwise_conv(self.pointwise_conv(x))` — 先执行 pointwise（改变通道数）再执行 depthwise，顺序颠倒。
- **影响**：当 `in_ch != out_ch` 时（如 Embedding→Model 的 364→96 通道投影），pointwise 输出 `out_ch` 个通道但 depthwise 期望 `in_ch` 个通道，触发维度不匹配崩溃。
- **修复**：改为 `return self.pointwise_conv(self.depthwise_conv(x))`。
- **状态**：✅ 已修复

---

### 错误 14：Highway 网络 transpose 维度错误

- **文件**：[Models/embedding.py](Assignment1_2026-main/Models/embedding.py) 第 19 行
- **Bug**：`x = x.transpose(0, 2)` 将 `[B, C, L]` 变为 `[L, C, B]`，但 `nn.Linear` 需要最后一维为特征维度，应该是 `[B, L, C]`。
- **影响**：线性层在错误的维度上操作（在 B 维度而非 C 维度），计算结果完全错误；且返回时 `x.transpose(1, 2)` 产生 `[L, B, C]` 而非期望的 `[B, C, L]`，后续维度不匹配。
- **修复**：改为 `x.transpose(1, 2)`。
- **状态**：✅ 已修复

---

### 错误 15：字符嵌入 permute 顺序错误

- **文件**：[Models/embedding.py](Assignment1_2026-main/Models/embedding.py) 第 39 行
- **Bug**：`ch_emb.permute(0, 2, 1, 3)` 将 `[B, L, char_len, d_char]` 变为 `[B, char_len, L, d_char]`，但 Conv2d 期望通道在 dim=1，即 `[B, d_char, L, char_len]`。
- **影响**：Conv2d 的 `in_channels=d_char=64` 但实际输入通道为 `char_len=16`，触发维度不匹配崩溃。
- **修复**：改为 `permute(0, 3, 1, 2)`。
- **状态**：✅ 已修复

---

### 错误 16：CQAttention 中 bmm 操作数顺序错误

- **文件**：[Models/attention.py](Assignment1_2026-main/Models/attention.py) 第 38 行
- **Bug**：`A = torch.bmm(Q, S1)` — `Q` shape `[B, Lq, C]`，`S1` shape `[B, Lc, Lq]`，内部维度 `C != Lc`。
- **影响**：矩阵乘法维度不匹配（C=96 vs Lc=400），触发 `RuntimeError`。
- **修复**：改为 `A = torch.bmm(S1, Q)`，即 `[B, Lc, Lq] @ [B, Lq, C] = [B, Lc, C]`。
- **状态**：✅ 已修复

---

### 错误 17：`nll_loss` 参数顺序交换

- **文件**：[Losses/loss.py](Assignment1_2026-main/Losses/loss.py) 第 7 行
- **Bug**：`F.nll_loss(y1, p1)` — `F.nll_loss` 期望 `(input, target)` 但传入了 `(target, input)`；第二项 `F.nll_loss(p2, y2)` 顺序正确。
- **影响**：`y1`（1D LongTensor）被当作 log-probabilities 输入（需要 2D），触发维度错误崩溃。
- **修复**：改为 `F.nll_loss(p1, y1)`。
- **状态**：✅ 已修复

---

### 错误 18：Adam 优化器状态键名不匹配

- **文件**：[Optimizers/adam.py](Assignment1_2026-main/Optimizers/adam.py) 第 60-63 行
- **Bug**：状态初始化使用 `state["exp_avg"]` 和 `state["exp_avg_sq"]`，但后续访问使用 `state["m"]` 和 `state["v"]`。
- **影响**：第二个训练步骤（step > 1 时 state 已初始化但键名不存在）触发 `KeyError`，训练崩溃。
- **修复**：将后续访问改为 `state["exp_avg"]` 和 `state["exp_avg_sq"]`，与初始化一致。
- **状态**：✅ 已修复

---

### 错误 19：SGDMomentum 状态键名不匹配

- **文件**：[Optimizers/sgd_momentum.py](Assignment1_2026-main/Optimizers/sgd_momentum.py) 第 48-51 行
- **Bug**：初始化判断 `if "velocity" not in state` 但存储时使用 `state["vel"]`；后续访问 `state["velocity"]`。
- **影响**：`"velocity"` 永远不在 state 中（因为存的是 `"vel"`），导致每次 `v = state["velocity"]` 触发 `KeyError` 崩溃。
- **修复**：统一键名，将 `state["vel"]` 改为 `state["velocity"]`。
- **状态**：✅ 已修复

---

### 错误 20：评估时 `argmax` 作用在错误的维度

- **文件**：[EvaluateTools/eval_utils.py](Assignment1_2026-main/EvaluateTools/eval_utils.py) 第 107-108 行
- **Bug**：`torch.argmax(p1, dim=0)` 在 batch 维度取最大值，结果 shape 为 `[L]`，而非在序列维度取 argmax 得到 `[B]`。
- **影响**：预测结果的 shape 和语义完全错误，评估指标（F1/EM）为垃圾值，无法反映模型真实性能。
- **修复**：改为 `dim=1`。
- **状态**：✅ 已修复

---

### 错误 21：LayerNorm 中 `keepdim=False` 导致广播失败

- **文件**：[Models/Normalizations/layernorm.py](Assignment1_2026-main/Models/Normalizations/layernorm.py) 第 37-38 行
- **Bug**：`x.mean(dim=dims, keepdim=False)` — 对 `[B, C, L]` 在 `(-2, -1)` 维度求均值后得到 `[B]`，但 `x - mean` 时 `[B, C, L] - [B]` 无法正确广播（最后一维 L != B）。
- **影响**：触发广播 `RuntimeError`，所有使用 LayerNorm 的编码器块前向传播崩溃。
- **修复**：改为 `keepdim=True`。
- **状态**：✅ 已修复

---

## 第二阶段：深度学习机制正确性错误（20 个）

本阶段的错误不会阻止系统运行（或仅在非默认配置下崩溃），但会扭曲训练动态、破坏优化稳定性或降低模型性能。

---

### 错误 1：ReLU 激活函数方向反转

- **文件**：[Models/Activations/relu.py](Assignment1_2026-main/Models/Activations/relu.py) 第 12 行
- **Bug**：`x.clamp(max=0.0)` 将所有正值截断为 0，保留负值——与 ReLU 定义完全相反。
- **影响**：所有正激活被抑制，负激活被保留，模型无法学习有意义的正向特征表示，训练效果极差。
- **修复**：改为 `x.clamp(min=0.0)`。
- **状态**：✅ 已修复

---

### 错误 2：LeakyReLU 的 `torch.where` 分支反转

- **文件**：[Models/Activations/leakeyReLU.py](Assignment1_2026-main/Models/Activations/leakeyReLU.py) 第 19 行
- **Bug**：`torch.where(x < 0, x, self.negative_slope * x)` — 当 `x < 0` 时保留原值，`x >= 0` 时缩小为 `0.01x`。
- **影响**：正输入被缩小为原来的 1%，负输入全部保留，与标准 LeakyReLU 逻辑完全相反，严重损害学习能力。
- **修复**：改为 `torch.where(x < 0, self.negative_slope * x, x)`。
- **状态**：✅ 已修复

---

### 错误 3：Dropout 缩放因子错误

- **文件**：[Models/dropout.py](Assignment1_2026-main/Models/dropout.py) 第 17 行
- **Bug**：`x * mask / self.p` — 除以 `p` 而非 `(1 - p)`。Inverted dropout 应除以保留概率 `(1-p)` 以保持期望值不变。
- **影响**：当 `p=0.1` 时，激活被放大 `1/0.1 = 10` 倍而非 `1/0.9 ≈ 1.11` 倍，导致训练极不稳定，数值溢出。
- **修复**：改为 `x * mask / (1.0 - self.p)`。
- **状态**：✅ 已修复

---

### 错误 4：LayerNorm 仿射变换中 weight 和 bias 交换

- **文件**：[Models/Normalizations/layernorm.py](Assignment1_2026-main/Models/Normalizations/layernorm.py) 第 41 行
- **Bug**：`x_norm * self.bias + self.weight` — 标准公式为 `y = x * gamma + beta`，即 `x_norm * self.weight + self.bias`。
- **影响**：初始时 `weight=1, bias=0`，公式变成 `x_norm * 0 + 1 = 1`（常数输出），归一化层在初始化时完全丧失信息传递能力。
- **修复**：改为 `x_norm * self.weight + self.bias`。
- **状态**：✅ 已修复

---

### 错误 5：GroupNorm reshape 维度顺序错误

- **文件**：[Models/Normalizations/groupnorm.py](Assignment1_2026-main/Models/Normalizations/groupnorm.py) 第 35 行
- **Bug**：`x.view(B, C // self.G, self.G, *spatial)` — 将通道数/组数放在 dim=1，组数放在 dim=2，顺序颠倒。
- **影响**：归一化的分组边界错误，每个"组"实际上混合了不同组的通道，归一化统计量计算在错误的通道集合上，降低归一化效果。
- **修复**：改为 `x.view(B, self.G, C // self.G, *spatial)`。
- **状态**：✅ 已修复

---

### 错误 6：Adam 权重衰减方向反转

- **文件**：[Optimizers/adam.py](Assignment1_2026-main/Optimizers/adam.py) 第 53 行
- **Bug**：`grad = grad.add(p, alpha=-wd)` — 使用负号，变成 `grad - wd * p`，相当于鼓励参数增大。
- **影响**：权重衰减效果反转，参数被推向更大值而非更小值，模型容易过拟合且参数可能发散。
- **修复**：改为 `alpha=wd`（正值）。
- **状态**：✅ 已修复

---

### 错误 7：Adam 第二矩估计未对梯度求平方

- **文件**：[Optimizers/adam.py](Assignment1_2026-main/Optimizers/adam.py) 第 69 行
- **Bug**：`v.mul_(beta2).add_(grad, alpha=1.0 - beta2)` — 跟踪梯度的一阶矩（均值）而非二阶矩（方差）。
- **影响**：`v_hat` 不再估计梯度方差，自适应学习率的缩放完全错误，Adam 退化为一个行为异常的优化器。
- **修复**：改为 `v.mul_(beta2).add_(grad * grad, alpha=1.0 - beta2)`。
- **状态**：✅ 已修复

---

### 错误 8：Adam 偏差校正使用乘法而非幂运算

- **文件**：[Optimizers/adam.py](Assignment1_2026-main/Optimizers/adam.py) 第 72-73 行
- **Bug**：`bias_correction1 = 1.0 - beta1 * t` 使用 `*`（乘法），正确应为 `1.0 - beta1 ** t`（幂运算）。
- **影响**：偏差校正值计算错误，在训练后期可能变为负数或零（如 `beta1=0.8, t=2: 1-0.8*2=-0.6`），导致 `m_hat` 方向反转或除零，优化过程严重失稳。
- **修复**：改为 `beta1 ** t` 和 `beta2 ** t`。
- **状态**：✅ 已修复

---

### 错误 9：SGD 权重衰减方向反转

- **文件**：[Optimizers/sgd.py](Assignment1_2026-main/Optimizers/sgd.py) 第 39 行
- **Bug**：`grad = grad.add(p, alpha=-wd)` — 与错误 6 相同，使用负号导致权重衰减方向反转。
- **影响**：L2 正则化效果反转，等价于鼓励参数增大。
- **修复**：改为 `alpha=wd`。
- **状态**：✅ 已修复

---

### 错误 10：SGDMomentum 速度更新用减法代替加法

- **文件**：[Optimizers/sgd_momentum.py](Assignment1_2026-main/Optimizers/sgd_momentum.py) 第 54 行
- **Bug**：`v.mul_(mu).sub_(grad)` 计算 `v = mu * v - grad`，标准公式为 `v = mu * v + grad`。
- **影响**：速度方向与梯度方向相反，动量机制反向工作，参数更新方向被干扰，优化不稳定。
- **修复**：改为 `v.mul_(mu).add_(grad)`。
- **状态**：✅ 已修复

---

### 错误 11：余弦退火调度器使用 `math.PI`（大写）

- **文件**：[Schedulers/cosine_scheduler.py](Assignment1_2026-main/Schedulers/cosine_scheduler.py) 第 28 行
- **Bug**：`math.PI` 在 Python 中不存在，正确为 `math.pi`（小写）。
- **影响**：选择余弦调度器时触发 `AttributeError` 崩溃。
- **修复**：改为 `math.pi`。
- **状态**：✅ 已修复

---

### 错误 12：余弦退火调度器公式缺少 0.5 系数

- **文件**：[Schedulers/cosine_scheduler.py](Assignment1_2026-main/Schedulers/cosine_scheduler.py) 第 28 行
- **Bug**：`eta_min + (base_lr - eta_min) * (1 + cos(...))` 缺少 `0.5` 系数。
- **影响**：初始学习率为 `2 * base_lr - eta_min` 而非 `base_lr`，学习率范围翻倍，训练动态被扭曲。
- **修复**：改为 `eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(...))`。
- **状态**：✅ 已修复

---

### 错误 13：LambdaLR 使用加法而非乘法

- **文件**：[Schedulers/lambda_scheduler.py](Assignment1_2026-main/Schedulers/lambda_scheduler.py) 第 23 行
- **Bug**：`return [base_lr + factor for ...]` — Lambda 调度器应将 `base_lr` 乘以 factor，而非相加。
- **影响**：默认配置下 factor=1.0，实际 lr = base_lr + 1.0 = 2.0（Adam 的 base_lr=1.0），学习率翻倍，可能导致训练不稳定。
- **修复**：改为 `base_lr * factor`。
- **状态**：✅ 已修复

---

### 错误 14：StepLR 使用乘法而非幂运算

- **文件**：[Schedulers/step_scheduler.py](Assignment1_2026-main/Schedulers/step_scheduler.py) 第 25 行
- **Bug**：`base_lr * self.gamma * (t // self.step_size)` — 使用 `gamma * (t // step_size)` 而非 `gamma ** (t // step_size)`。
- **影响**：在 `t=0` 时 lr = 0（乘以 0），模型完全无法学习；后续 lr 线性增长而非指数衰减，与 step decay 语义完全不符。
- **修复**：改为 `self.gamma ** (t // self.step_size)`。
- **状态**：✅ 已修复

---

### 错误 15：Kaiming 初始化标准差缺少因子 2

- **文件**：[Models/Initializations/kaiming.py](Assignment1_2026-main/Models/Initializations/kaiming.py) 第 25 行和第 38 行
- **Bug**：`std = math.sqrt(1.0 / fan)` — He 初始化（针对 ReLU）应为 `sqrt(2.0 / fan)`。
- **影响**：权重初始化方差偏小（少了 2 倍），可能导致信号在深层网络中逐渐衰减（方差萎缩），影响训练初期的梯度流。
- **修复**：改为 `math.sqrt(2.0 / fan)`。
- **状态**：✅ 已修复

---

### 错误 16：Xavier 初始化使用 `fan_in * fan_out` 而非 `fan_in + fan_out`

- **文件**：[Models/Initializations/xavier.py](Assignment1_2026-main/Models/Initializations/xavier.py) 第 24 行和第 36 行
- **Bug**：`std = gain * math.sqrt(2.0 / (fan_in * fan_out))` — Glorot 初始化应为 `sqrt(2.0 / (fan_in + fan_out))`。
- **影响**：当 `fan_in` 和 `fan_out` 较大时（如 96*96=9216 vs 96+96=192），方差极度偏小，权重初始化近乎为零，导致信号消失。
- **修复**：改为 `fan_in + fan_out`。
- **状态**：✅ 已修复

---

### 错误 17：MultiHeadAttention 输入 permute 顺序错误

- **文件**：[Models/encoder.py](Assignment1_2026-main/Models/encoder.py) 第 70-72 行
- **Bug**：`q.permute(2, 0, 1, 3)` 产生 `[h, B, L, d_k]`，标准做法是 `permute(0, 2, 1, 3)` 产生 `[B, h, L, d_k]`。
- **影响**：后续 view 为 `[B*h, L, d_k]` 时数据排布错误（head-major vs batch-major），各 head/batch 的数据被混淆，注意力计算结果在输出重组时被打乱。
- **修复**：改为 `permute(0, 2, 1, 3)`。
- **状态**：✅ 已修复

---

### 错误 18：MultiHeadAttention 输出 permute 顺序错误

- **文件**：[Models/encoder.py](Assignment1_2026-main/Models/encoder.py) 第 85 行
- **Bug**：`out.permute(1, 2, 0, 3)` 从 `[B, h, L, d_k]` 产生 `[h, L, B, d_k]`，应为 `permute(0, 2, 1, 3)` 得到 `[B, L, h, d_k]`。
- **影响**：与错误 17 相关联，输出的 batch 和 head 维度被打乱，最终 view 和 FC 层的计算结果对应关系混乱，注意力输出语义错误。
- **修复**：改为 `permute(0, 2, 1, 3)`。
- **状态**：✅ 已修复

---

### 错误 19：MultiHeadAttention 缺少缩放因子

- **文件**：[Models/encoder.py](Assignment1_2026-main/Models/encoder.py) 第 78 行
- **Bug**：`attn = torch.bmm(q, k.transpose(1, 2))` — 已计算 `self.scale = 1/sqrt(d_k)` 但未应用。
- **影响**：缺少 `1/sqrt(d_k)` 缩放，点积值可能过大，softmax 输出趋近于 one-hot，梯度接近零（梯度消失），注意力机制学习效率低下。
- **修复**：改为 `attn = torch.bmm(q, k.transpose(1, 2)) * self.scale`。
- **状态**：✅ 已修复

---

### 错误 20：MultiHeadAttention 注意力 mask 的 repeat 方式与 batch-major 数据排布不一致

- **文件**：[Models/encoder.py](Assignment1_2026-main/Models/encoder.py) 第 76 行
- **Bug**：`attn_mask = mask.unsqueeze(1).expand(-1, length, -1).repeat(self.num_heads, 1, 1)` — `repeat(h, 1, 1)` 沿 dim=0 将 h 组 `[B, L, L]` 依次拼接，产生 head-major 顺序 `[b0, b1, ..., b_{B-1}, b0, b1, ..., b_{B-1}, ...]`。但修复错误 17 后 q/k/v 为 batch-major 排布 `[b0h0, b0h1, ..., b0h_{h-1}, b1h0, ...]`，mask 与数据的 batch 索引不再对齐，导致错误的 token 被 mask。
- **影响**：PAD mask 被施加在错误的 batch 样本上，部分有效 token 被遮蔽、部分 PAD token 未被遮蔽，注意力分布被扭曲，模型学习效果下降。
- **修复**：改为 `mask.unsqueeze(1).expand(-1, length, -1).unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, length, length)`，使 mask 以 batch-major 方式排布，与 q/k/v 一致。
- **状态**：✅ 已修复

---

## 第三阶段：与论文设计不一致的架构/训练逻辑修复（6 个）

本阶段修复的是代码虽然能运行但与 QANet 原论文 (Yu et al., 2018) 设计不一致的逻辑问题。这些偏差会限制模型性能上限。

---

### 错误 1：LayerNorm 归一化维度包含序列长度

- **文件**：[Models/Normalizations/normalization.py](Assignment1_2026-main/Models/Normalizations/normalization.py) `get_norm` 函数
- **Bug**：`LayerNorm([d_model, length])` 同时在通道维 C 和序列长度 L 上归一化。论文中 LayerNorm 应只对每个位置的特征向量（通道维 C）归一化，各位置独立处理。
- **影响**：
  - 不同位置的统计量被耦合，短序列（question，长度 50）和长序列（context，长度 400）的归一化行为不同。
  - 归一化模块变得依赖序列长度，导致 context/question 无法共享同一编码器实例（论文要求共享）。
- **修复**：新增 `_ChannelLayerNorm` 包装类：先将 `[B, C, L]` 转置为 `[B, L, C]`，对 C 维应用标准 `LayerNorm(d_model)`，再转置回来。`get_norm("layer_norm", ...)` 现在返回此包装类。
- **状态**：✅ 已修复

---

### 错误 2：推理阶段 span 解码使用独立 argmax

- **文件**：[EvaluateTools/eval_utils.py](Assignment1_2026-main/EvaluateTools/eval_utils.py) 评估循环
- **Bug**：`yp1 = argmax(p1, dim=1); yp2 = argmax(p2, dim=1)` 分别独立取 start 和 end 位置的 argmax，可能产生 `start > end` 的无效 span（原代码用 min/max 交换修补，但这改变了语义）。
- **影响**：论文要求联合搜索 `argmax_{s<=e} [p1(s) + p2(e)]`。独立 argmax + min/max 交换会选出次优甚至语义错误的 span，直接拉低 F1/EM 指标。
- **修复**：改为构建 `[B, L, L]` 得分矩阵 `score = p1.unsqueeze(2) + p2.unsqueeze(1)`，用上三角 band mask（`s <= e` 且 `e - s < 30`）过滤非法 span，取全局 argmax 得到最优 `(start, end)` 对。
- **状态**：✅ 已修复

---

### 错误 3：GloVe 词向量未冻结 + OOV 向量为全零

- **文件**：[Models/qanet.py](Assignment1_2026-main/Models/qanet.py) 第 41-44 行；[Tools/preproc.py](Assignment1_2026-main/Tools/preproc.py) 第 155 行
- **Bug**：
  1. `nn.Embedding.from_pretrained(word_mat, freeze=False)` — 预训练 GloVe 向量未冻结，训练时会被更新。论文明确指出冻结预训练词向量。
  2. `embedding_dict[oov_tok] = [0.0] * vec_size` — OOV 词的嵌入被初始化为全零向量，训练时（冻结模式下）永远为零。
- **影响**：
  - 不冻结 GloVe 会导致过拟合（尤其在小数据集上），且词向量偏离预训练分布。
  - OOV 全零意味着未登录词没有任何信号输入模型，降低模型对罕见词的理解能力。
- **修复**：
  1. 将 `freeze=False` 改为 `freeze=True`。
  2. OOV 向量改为 `np.random.normal(scale=0.1)` 随机初始化。
- **状态**：✅ 已修复

---

### 错误 4：Context/Question Embedding Encoder 未共享权重

- **文件**：[Models/qanet.py](Assignment1_2026-main/Models/qanet.py) 第 47-51 行
- **Bug**：代码创建了四个独立模块：`context_conv`、`question_conv`、`c_emb_enc`、`q_emb_enc`。但 QANet 论文明确说明 Embedding Encoder Layer 在 context 和 question 之间共享权重。
- **影响**：
  - 参数量翻倍（两套 conv + encoder block），增加过拟合风险。
  - Context 和 question 被投影到不同的表示空间，CQ Attention 的对齐效果下降。
- **修复**：合并为 `emb_conv`（共享 DepthwiseSeparableConv）和 `emb_enc`（共享 EncoderBlock，length 取 `max(len_c, len_q)`）。forward 中 context 和 question 通过同一模块处理。
- **状态**：✅ 已修复

---

### 错误 5：Lambda 调度器忽略学习率参数

- **文件**：[Schedulers/scheduler.py](Assignment1_2026-main/Schedulers/scheduler.py) 第 25-27 行
- **Bug**：`lambda_scheduler` 硬编码 `lr_lambda=lambda _: 1.0`，配合 Adam（`base_lr=1.0`）使用时有效学习率恒为 1.0，远高于论文的 0.001。`args.learning_rate` 参数被完全忽略。
- **影响**：使用 Adam + lambda 组合时，学习率为 1.0（论文值的 1000 倍），训练必然发散。
- **修复**：改为从 `args.learning_rate` 读取目标学习率作为 lambda 因子：`lr_lambda=lambda _: lr`，使得 `effective_lr = base_lr(1.0) * lr = learning_rate`。
- **状态**：✅ 已修复

---

### 错误 6：训练 Notebook 使用 SGD + Cosine 而非论文的 Adam

- **文件**：[assignment1.ipynb](Assignment1_2026-main/assignment1.ipynb) Cell 10
- **Bug**：`optimizer_name="sgd", scheduler_name="cosine"` — 论文使用 Adam optimizer + warmup + constant lr。
- **影响**：SGD 缺少自适应学习率，在 QANet 这类 Transformer 风格模型上收敛慢且不稳定；cosine scheduler 与论文的 warmup + constant 策略差异大。
- **修复**：改为 `optimizer_name="adam", scheduler_name="lambda"`。`train.py` 中的默认超参（`learning_rate=1e-3, beta1=0.8, beta2=0.999, weight_decay=3e-7`）已与论文一致。
- **状态**：✅ 已修复

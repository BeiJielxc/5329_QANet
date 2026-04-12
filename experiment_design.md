# QANet 核心机制的必要性验证：小规模架构下经典深度学习设计选择的实证分析

## 研究主题

**标题**：Revisiting Standard Deep Learning Practices in Small-Scale QANet: Are They Mechanistically Necessary?

**描述**：经典深度学习理论中的若干"标准做法"——如注意力缩放、层级式正则化、均值中心化归一化——通常被视为模型训练的必要组件。然而，这些机制的理论假设往往建立在大规模模型配置之上（例如 d_k=64 的标准 Transformer）。本研究以 QANet（d_model=128, d_k=16）在 SQuAD v1.1 抽取式问答任务上的实现为实验平台，系统性地验证三种核心设计选择的机制性必要性。通过单变量控制实验与多维度诊断指标，我们旨在揭示：这些理论上"应该有效"的机制，在具体的小规模架构和特定任务条件下，其预期的因果效应是否真正被激活。

---

## 统一实验协议

以下协议适用于所有三个实验，确保结果的可比性和可复现性。

| 项目 | 设置 |
|------|------|
| 数据集 | SQuAD v1.1 (train-mini / dev-v1.1) |
| 评测集 | Full dev set (`test_num_batches=-1`) |
| 随机种子 | 每组实验运行 seed=42, 13, 7 三次，报告均值 ± 标准差 |
| 训练步数 | 60000 (early stopping patience=30) |
| 主要指标 | F1, EM (full dev) |
| 辅助指标 | 因实验而异（见各实验设计） |
| 基线配置 | d_model=128, num_heads=8, batch_size=32, dropout=0.1, optimizer=adam, scheduler=lambda, lr=1e-3, warmup_steps=1000 |
| 公平性原则 | 每个实验仅改变一个自变量，其余所有配置严格一致 |

---

## 实验一：Scaled Dot-Product Attention 中缩放因子的机制性作用

**类别**：架构组件（Architectural Component）

### 1.1 研究问题

在 Multi-Head Self-Attention 中，对 Q·Kᵀ 乘以 1/√d_k 的缩放因子是否对 QANet 的训练动态和最终性能产生实质性影响？若有影响，其因果机制是什么？

### 1.2 理论动机

Vaswani et al. (2017) 提出缩放因子的理由是：当 d_k 较大时，点积 Q·Kᵀ 的方差会随 d_k 线性增长（假设 q, k 的分量独立且均值为 0、方差为 1，则 q·k 的方差 = d_k）。过大的点积值会将 softmax 推入饱和区域，导致梯度接近于零，训练效率下降。

当前代码中，缩放因子被定义但从未使用：

- `encoder.py` 第 53 行：`self.scale = 1.0 / math.sqrt(self.d_k)` — 定义了缩放因子
- `encoder.py` 第 78 行：`attn = torch.bmm(q, k.transpose(1, 2))` — 未乘以 `self.scale`

当前 d_k = d_model / num_heads = 128 / 8 = 16，缩放因子为 1/√16 = 0.25。这里存在一个有趣的张力：d_k=16 相对较小，理论上饱和效应可能不如 d_k=64（如标准 Transformer）那么严重。这使得实验结论非平凡——不能简单假设"加了缩放就一定好"。

### 1.3 假设

- **H1（主假设）**：添加 1/√d_k 缩放因子将提升模型的 F1 和 EM 性能，因为缩放防止了 softmax 饱和，从而改善了梯度传播和注意力分布的多样性。
- **H1-null（零假设）**：在 d_k=16 的较小维度下，缩放因子对性能无统计显著影响，因为点积的量级增长不足以触发严重的 softmax 饱和。

### 1.4 实验设计

| 要素 | 说明 |
|------|------|
| **自变量** | 是否在 `attn = Q·Kᵀ` 后乘以 1/√d_k |
| **因变量** | (1) F1/EM 性能 (2) 注意力熵 (3) 注意力层梯度范数 (4) 损失收敛曲线 |
| **控制变量** | d_model, num_heads, dropout, optimizer, scheduler, seed, batch_size 及其余所有超参数 |

**实验组设置**：

| 组别 | 代码改动 | 说明 |
|------|----------|------|
| **Control** | `attn = torch.bmm(q, k.transpose(1,2))` | 原始无缩放 |
| **Treatment** | `attn = torch.bmm(q, k.transpose(1,2)) * self.scale` | 标准缩放 |

### 1.5 量化指标与评测协议

**主指标**：
- Full dev F1 / EM（`test_num_batches=-1`），3 次 seed 均值 ± 标准差

**辅助诊断指标**（需在训练过程中记录）：

1. **注意力熵** H(α)：对每个 head 的注意力分布计算 H = -Σ αᵢ log αᵢ，取所有 head、所有 encoder block 的均值。高熵表示注意力分散，低熵表示注意力过度集中（尖锐）。每隔一定 step 在 dev 子集上采样计算。
2. **注意力层梯度 L2 范数**：记录 `self_att` 模块中 `q_linear`、`k_linear` 参数梯度的 L2 范数随训练步数的变化。若无缩放导致 softmax 饱和，梯度范数会更小且更不稳定。
3. **收敛速度**：定义为达到特定 F1 阈值（例如 35.0）所需的训练步数。
4. **注意力点积统计量**：在 softmax 之前，记录 `attn = Q·Kᵀ` 的均值和方差。验证理论预测：无缩放时方差约为 d_k 倍于有缩放时。

### 1.6 因果分析框架

| 观察到的结果模式 | 因果解释 |
|------------------|----------|
| Treatment F1 > Control 且 Treatment 注意力熵更高 | 缩放成功防止了 softmax 饱和，注意力分布更合理，模型能关注更多相关位置 |
| Treatment F1 ≈ Control 且两组点积方差均较小 | d_k=16 维度太小，点积量级未增长到触发饱和的程度，缩放因子在该配置下冗余 |
| Treatment F1 < Control | 无缩放可能在 d_k 较小时隐式产生了更"confident"的注意力模式，对该任务反而有益；需分析具体注意力模式差异 |
| 两组 F1 相似但收敛速度不同 | 缩放因子主要影响训练动态而非最终表达能力 |

---

## 实验二：卷积层 Stochastic Depth Dropout 的正则化机制分析

**类别**：正则化技术（Regularization Technique）

### 2.1 研究问题

EncoderBlock 中卷积层的线性递增 dropout 策略（stochastic depth pattern）如何影响模型的泛化能力？这种层级递增的正则化方式是否优于均匀 dropout？

### 2.2 理论动机

当前代码中 EncoderBlock 实现了一种受 Stochastic Depth (Huang et al., 2016) 启发的 dropout 模式：

- `encoder.py` 第 95-96 行：dropout 率随卷积层深度线性递增 `dropout * (i + 1) / conv_num`
- `encoder.py` 第 118-119 行：dropout 仅在每 2 层卷积后施加 `if (i + 1) % 2 == 0`

以 embedding encoder (conv_num=4, dropout=0.1) 为例，4 层卷积的 dropout 率分别为 [0.025, 0.05, 0.075, 0.1]，但只有第 2 层（p=0.05）和第 4 层（p=0.1）实际被施加。

这背后的假设是：**浅层卷积提取的是基础特征，丢弃代价更高；深层卷积的特征更冗余，可以承受更强的正则化。** 但这个假设是否成立？均匀 dropout 是否就够了？完全去除 conv dropout 是否表明模型并未在卷积特征层面过拟合？

### 2.3 假设

- **H1（主假设）**：线性递增的 stochastic depth dropout 比均匀 dropout 更有效，因为它在保护浅层基础特征表示的同时对深层冗余特征施加更强正则化，从而在表达能力和泛化之间取得更好的平衡。
- **H2（辅助假设）**：完全移除卷积层 dropout 会导致 train/dev 性能差距增大（过拟合加剧），验证该组件的正则化必要性。

### 2.4 实验设计

| 要素 | 说明 |
|------|------|
| **自变量** | 卷积层的 dropout 策略（分布模式和施加频率） |
| **因变量** | (1) F1/EM 性能 (2) 泛化差距（train F1 − dev F1） (3) 各层特征表示稳定性 |
| **控制变量** | 注意力层和 FFN 的 dropout 保持不变（`self.drop = Dropout(dropout)`），模型其余配置完全一致 |

**实验组设置**：

| 组别 | Dropout 策略 | 具体 dropout 率 (conv_num=4, p=0.1) | 施加频率 |
|------|-------------|--------------------------------------|----------|
| **A (Control)** | 线性递增 + 隔层施加 | [0.025, **0.05**, 0.075, **0.1**] | 每 2 层 |
| **B** | 均匀 + 隔层施加 | [0.1, **0.1**, 0.1, **0.1**] | 每 2 层 |
| **C** | 线性递增 + 每层施加 | [**0.025**, **0.05**, **0.075**, **0.1**] | 每层 |
| **D** | 无卷积层 dropout | [0, 0, 0, 0] | 无 |

（加粗表示实际被施加的层）

### 2.5 量化指标与评测协议

**主指标**：
- Full dev F1 / EM，3 次 seed 均值 ± 标准差

**辅助诊断指标**：

1. **泛化差距 (Generalization Gap)**：ΔF1 = train F1 − dev F1。随训练步数追踪，差距越大表明过拟合越严重。这是验证正则化有效性的关键指标。
2. **收敛曲线特征**：
   - 达到最佳 dev F1 所需步数
   - Early stopping 触发时刻
   - Dev loss 开始上升的拐点（过拟合起始点）
3. **各卷积层输出的特征方差**：每隔一定步数，在 dev 子集上记录每层卷积输出的 channel-wise 方差。如果 dropout 作为噪声注入有效，它应该防止深层特征方差的坍缩或爆炸。

### 2.6 因果分析框架

| 比较对 | 验证的因果关系 |
|--------|-------------|
| A vs D | 卷积层 dropout 的 **正则化必要性**：若 D 的泛化差距显著大于 A，则证明卷积层确实需要正则化 |
| A vs B | **线性递增策略 vs 均匀策略** 的优劣：若 A 优于 B，则说明"浅层保护、深层正则化"的梯度式策略有机制性优势 |
| A vs C | **隔层施加 vs 每层施加** 的影响：若 C 优于 A，则说明当前实现遗漏了部分层的正则化，存在改进空间；若 C 不如 A，则隔层施加可能是一种隐式的"残差信号保护"机制 |
| B vs D | 均匀 dropout 的边际收益：隔离 dropout 策略分布的影响，只看"有 vs 无"的效果 |

**关键因果链验证**：若 A 在 dev F1 上显著优于 B，但两者 train F1 相近，则因果链为：**线性递增 → 浅层特征保护 → 更好的基础表示传递 → 更好的泛化**。若 A 和 B 相近，但均显著优于 D，则因果结论为：**dropout 本身必要，但分布模式无关紧要**。

---

## 实验三：LayerNorm vs RMSNorm——均值中心化在 QANet 中的必要性

**类别**：归一化策略（Normalization Strategy）

### 3.1 研究问题

LayerNorm 中的均值中心化（mean-centering）操作对 QANet 的性能贡献是多少？去除该操作（即使用 RMSNorm）是否会导致性能下降？

### 3.2 理论动机

LayerNorm (Ba et al., 2016) 执行两步操作：(1) 均值中心化 x ← x − μ，(2) 方差归一化 x ← x / √(σ² + ε)，随后施加可学习的仿射变换。RMSNorm (Zhang & Sennrich, 2019) 的核心论点是：**归一化的主要贡献来自方差的缩放不变性，均值中心化可能是冗余的。**  RMSNorm 已被 LLaMA (Touvron et al., 2023)、Gemma 等现代大规模架构广泛采用。

当前实现中的 LayerNorm 使用 `normalized_shape=[d_model, 1]`（`normalization.py` 第 41 行），即仅在 channel 维度上进行归一化（对每个位置独立归一化），与标准 Transformer 的 LayerNorm 行为一致。每个 EncoderBlock 包含多个归一化层（`normb`、`norms[0..conv_num-1]`、`norme`），因此归一化策略的影响在整个模型中被放大。

此实验的机制性核心在于：**在 QANet 的 pre-norm 残差结构（Norm → Sublayer → Add）中，均值偏移是否蕴含了有意义的信号？** 如果均值偏移编码了有用信息（例如表示一个 token 的"整体激活强度"），那么中心化会破坏这部分信息；反之如果均值主要是噪声，中心化可以简化优化景观。

### 3.3 假设

- **H1（主假设）**：RMSNorm 可以达到与 LayerNorm 统计上无显著差异的 F1/EM 性能，因为在 QANet 的 d_model=128 维度下，均值中心化的贡献是次要的。
- **H2（辅助假设）**：RMSNorm 由于减少了一次 reduce 操作（省去均值计算），会带来更快的每步训练速度（wall-clock time）。

### 3.4 实验设计

| 要素 | 说明 |
|------|------|
| **自变量** | 归一化方法（LayerNorm / RMSNorm / 无归一化） |
| **因变量** | (1) F1/EM (2) 训练收敛动态 (3) 归一化前的激活统计量 (4) 每步训练耗时 |
| **控制变量** | 归一化层以外的所有模块和超参数完全一致 |

**实验组设置**：

| 组别 | 归一化方法 | 数学形式 | 说明 |
|------|-----------|---------|------|
| **A (Control)** | LayerNorm | x̂ = (x−μ) / √(σ²+ε) · γ + β | 原始配置 |
| **B (Treatment)** | RMSNorm | x̂ = x / √(mean(x²)+ε) · γ | 去除均值中心化，去除 bias β |
| **C (Ablation)** | Identity（无归一化） | x̂ = x | 验证归一化本身的必要性 |

**RMSNorm 实现方式**：在 `Normalizations/` 目录下新增 `RMSNorm` 类并注册到 `normalizations` 字典中，通过 `norm_name="rms_norm"` 参数切换。

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        n = len(self.weight.shape)
        dims = tuple(range(-n, 0))
        rms = torch.sqrt(torch.mean(x ** 2, dim=dims, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

### 3.5 量化指标与评测协议

**主指标**：
- Full dev F1 / EM（3 次 seed 均值 ± 标准差）
- 对 3 次运行结果进行 **paired t-test**（或 Wilcoxon signed-rank test），报告 p-value，以判断性能差异的统计显著性

**辅助诊断指标**：

1. **归一化前激活的均值统计**：在每个归一化层的输入上，计算 channel-wise 均值 μ 的绝对值的平均（|μ|\_avg）。如果该值持续较大，说明均值偏移是显著的，中心化有意义；如果该值接近 0，说明均值本身就很小，中心化冗余。
2. **训练收敛曲线**：对比 A/B/C 三组的 dev loss 和 dev F1 随 step 的变化，关注：
   - 达到最佳 F1 的步数
   - 是否出现训练不稳定（loss spike）
3. **每步训练时间**：测量平均每个训练 step 的 wall-clock time（取 1000 步平均），量化 RMSNorm 的计算效率优势。
4. **组 C 的表现**：若无归一化时模型完全不收敛或严重退化，则确认归一化的基础必要性。

### 3.6 因果分析框架

| 观察到的结果模式 | 因果解释 |
|------------------|----------|
| A ≈ B >> C | 归一化本身是关键机制（方差缩放不变性），但均值中心化冗余。支持 Zhang & Sennrich (2019) 的论点。 |
| A > B >> C | 均值中心化在 QANet 中贡献了额外的优化稳定性。需进一步分析 \|μ\| 统计量确认均值偏移确实显著。 |
| A ≈ B ≈ C | 在当前配置下归一化整体贡献有限（可能因 d_model 较小或残差连接已足够稳定训练），需分析梯度范数验证。 |
| B > A | RMSNorm 反而更好——可能的原因是均值中心化在 pre-norm 残差结构中破坏了有用的均值信号。需分析残差路径上均值的信息内容。 |

**关键因果链验证**：若 A ≈ B，则检查归一化前 |μ|\_avg 的值：
- 若 |μ|\_avg ≈ 0 → 均值本身接近零，中心化操作本质上是恒等变换，不影响结果
- 若 |μ|\_avg >> 0 但 A ≈ B → 均值虽大，但被学习到的仿射参数（LayerNorm 的 β 或后续层）补偿了，中心化的信息丢失被弥补

这两种因果机制有不同的理论含义，必须通过统计量来区分。

---

## 三个实验的整体关系

| 实验 | 机制类别 | 研究层面 | 代码改动位置 |
|------|---------|---------|-------------|
| 实验一 | 架构组件 | 注意力机制内部 | `Models/encoder.py` — `MultiHeadAttention.forward` |
| 实验二 | 正则化技术 | 编码器块的训练正则化 | `Models/encoder.py` — `EncoderBlock.__init__` + `forward` |
| 实验三 | 归一化策略 | 跨所有编码器块的归一化层 | `Models/Normalizations/` 模块 |

三个实验遵循 **单变量控制** 原则，实验间互不干扰。在确定各实验最优配置后，可选择进行 **联合实验**（同时应用最佳改动），观察是否存在协同效应或相互抵消——但这不是必须的。

每个实验的核心不在于"是否提升了性能"，而在于 **通过量化诊断指标建立因果解释**：为什么某个机制有效或无效，其底层的数学和优化原理是什么。

---

## 参考文献

- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Huang, G. et al. (2016). Deep Networks with Stochastic Depth. *ECCV*.
- Ba, J. L. et al. (2016). Layer Normalization. *arXiv:1607.06450*.
- Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS*.
- Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.
- Yu, A. W. et al. (2018). QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension. *ICLR*.

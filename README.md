# LLM-Rec：输出头微调与选择性参数合成

快手探索者 LLM-Rec 挑战赛要求一个模型同时理解物料、分析用户兴趣、生成推荐结果，并保留通用知识能力。本方案基于 **OneReason-0.8B**，把最终调整集中在语义 ID 的输出层：先冻结主干、训练输出头，再把训练得到的差分以 **−0.5** 的系数写入目标模型的指定行。

最终模型为 **EXP-182**，最高正式评测分数为 **1.0716429387157187**，对应 EXP-201 评测记录。合成在离线阶段完成，推理时使用单个模型。

| 项目 | 最终方案 |
|---|---|
| 基座 | `OneReason-0.8B-pretrain-competition` |
| 调整方式 | 冻结 Transformer 主干，训练 `lm_head` 10 步 |
| 合成范围 | SID 的 a、b、c 三组输出头行，共 24,576 行 |
| 合成公式 | `Target − 0.5 × (Expert − Ancestor)` |
| 部署形式 | 合成后的单个因果语言模型 |
| 最高总分 | **1.0716429387157187** |

**阅读导航：** [任务与数据表示](#任务与数据表示) · [方案设计](#方案设计) · [输出头训练](#输出头训练) · [选择性参数合成](#选择性参数合成) · [推理与评测](#推理与评测) · [代码导航](#代码导航)

## 任务与数据表示

### 一个模型承担四类任务

比赛将推荐相关能力放进同一个语言模型中评估。不同任务使用不同输入和评价指标，模型既要生成物料标识，也要生成自然语言或选择答案。

| 任务组 | 输入与目标 | 评价方式 |
|---|---|---|
| R0：物料理解 | 根据物料的内容描述，生成对应的 itemic pattern | Pass@64 |
| R1：用户理解 | 根据历史行为判断兴趣关联，并生成兴趣演化主题 | 行为选择 F1、主题生成综合 F1 |
| R2：多域推荐 | 根据用户历史，预测下一个交互物料；覆盖视频、电商、广告、直播 | 各域 Pass@64 |
| R3：世界知识 | 回答通用知识单项选择题 | 单次回答准确率 |

Pass@64 关注候选集合中能否命中目标。模型需要将概率分配到正确的语义 ID 序列上，同时保持用户理解和知识问答的输出能力。四组任务共同决定最终成绩，因此方案采用局部的输出层调整。

### 物料如何表示成 token

OneReason 使用语义 ID（Semantic ID，SID）表示物料。一个 itemic pattern 由一个域前缀和三个语义 token 组成：

```text
<|domain_begin|> <s_a_i> <s_b_j> <s_c_k>
       域          第 1 层    第 2 层    第 3 层
```

每层语义码本有 8,192 个取值。三层 token 按顺序生成，共同表示一个物料的语义编码。域前缀告诉模型该编码属于哪一种推荐场景。

| 场景 | 域前缀 | 数据中的域值 |
|---|---|---|
| 短视频 | `<\|video_begin\|>` | `video/video` |
| 电商商品 | `<\|prod_begin\|>` | `goods` |
| 广告 | `<\|ad_begin\|>` | `video/ad` |
| 直播 | `<\|living_begin\|>` | `live` |

例如，`<|video_begin|><s_a_123><s_b_456><s_c_789>` 展示了一个短视频物料的编码格式。这里的数字仅用于说明格式，没有对应到真实用户或物料。

物料数据通过 `(domain, pid)` 关联。`Pid2Sid` 提供三层编码，`Pid2Caption` 提供内容描述，`UserProfile` 提供用户的多域历史行为。它们分别支撑“由描述生成 SID”和“由历史预测 SID”两类输入。自然语言与 SID 使用同一个模型词表，最终都由输出头产生 token 分数。

## 方案设计

### 调整输出头的含义

设某一步的隐藏状态为 $h\in\mathbb{R}^{d}$，输出头为 $W\in\mathbb{R}^{V\times d}$，其中 $V$ 是词表大小。每个 token 的分数来自对应权重行与隐藏状态的内积：

$$
z = Wh,\qquad z_j = W_j h.
$$

Transformer 主干负责把输入上下文编码成隐藏状态，输出头负责把这个状态映射到词表。调整 SID 所在的权重行，就能改变这些 token 在当前上下文中的分数。最终方案把参数变更限制在这一位置。

这里需要区分两个范围：**训练时整个 `lm_head` 可训练，合成时只提取其中的 a、b、c 行**。训练得到的其他输出头变化不会写入最终模型。

### 三个模型各自提供什么

| 名称 | 在方案中的作用 | 对应模型 |
|---|---|---|
| Ancestor | 输出头训练的起点，用来确定差分的零点 | aligned-replay v8t freeze8 |
| Expert | 从 Ancestor 出发，训练输出头 10 步后的模型 | headfix + Liger 修复后的训练结果 |
| Target | 提供最终模型的主干和原始输出头 | EXP-174 |

Ancestor 与 Expert 形成一对，二者相减得到本次训练产生的更新方向。Target 是这个方向的应用对象，它不需要等于 Ancestor。

最终使用负系数 $-0.5$，相当于沿更新方向反向移动半步。训练阶段仍然正常最小化交叉熵损失；反向和缩放发生在离线合成阶段。这个系数是最终选用的模型配置，不代表反向更新对其他任务或模型也一定有效。

## 输出头训练

### 训练输入与冻结范围

训练使用 `takeover_r0_cbalanced_replay_v1` 数据和 `qwen3_nothink` 模板，从 aligned-replay 模型开始进行监督微调。数据承担生成输出头更新方向的作用，最终部署的模型主体仍由 Target 提供。

配置将可训练 Transformer 层数设为 0，并将 `lm_head` 列为额外可训练模块。主干权重冻结后，优化器只更新输出头。多个训练样本通过 packing 组成长序列，配置同时启用 neat packing。

| 参数 | 设置 |
|---|---|
| 微调方式 | `freeze`，`freeze_trainable_layers: 0` |
| 额外可训练模块 | `lm_head` |
| 优化步数 | 10 |
| 学习率 | `1e-5` |
| 调度 | cosine，warmup ratio 为 `0.03` |
| 单设备 batch size | 1 |
| 梯度累积 | 16 |
| 最大序列长度 | 32,768 |
| 权重衰减 / 梯度裁剪 | `0.0` / `1.0` |
| 训练精度 / Attention | FP16 / xFormers |
| 随机种子 | 20260720 |

完整设置见[输出头训练配置](07_接管/experiments/configs/exp182_headfix_reproduction_step10_fp16_xformers_ligerfix.yaml)。

### 冻结主干后的梯度路径

对一个有效目标 token $y$，训练损失为 $-\log\operatorname{softmax}(Wh)_y$。即使 $h$ 来自冻结主干，损失对 $W$ 仍然存在梯度，输出头应当继续学习。

所用 Liger 0.6.4 fused cross-entropy 实现会根据输入是否请求梯度决定权重梯度的计算路径。冻结主干后，隐藏状态通常不带梯度；本方案在输出头仍可训练时，将其转换为新的局部叶子张量：

```python
_input = _input.detach().requires_grad_(True)
```

这使融合算子能够计算输出头梯度。`detach()` 切断了与前面计算图的联系，主干参数仍处于冻结状态。

[训练入口](07_接管/experiments/run_frozen_head_training.py)通过 `sitecustomize` 把[梯度补丁](07_接管/experiments/liger_frozen_head_patch.py)加载到训练进程，并检查对应 Liger 源文件版本。这个处理服务于当前冻结方式与算子的组合，不改变交叉熵目标。

## 选择性参数合成

### 按行定义最终权重

令 $S$ 为选中的 SID 行集合，输出头更新方向为：

$$
\Delta W = W_{\mathrm{expert}} - W_{\mathrm{ancestor}}.
$$

最终输出头按行计算：

$$
W_{\mathrm{final},j} =
\begin{cases}
W_{\mathrm{target},j} - 0.5\Delta W_j, & j\in S,\\
W_{\mathrm{target},j}, & j\notin S.
\end{cases}
$$

所有非 `lm_head` 参数直接取自 Target，包括 Transformer 主干和输入 embedding。代码使用的词表区间为左闭右开区间：

| SID 分组 | token ID 区间 | 行数 |
|---|---|---:|
| a | `[151669, 159861)` | 8,192 |
| b | `[159861, 168053)` | 8,192 |
| c | `[168053, 176245)` | 8,192 |
| 合计 | `[151669, 176245)` | **24,576** |

假设某个被选中位置的 Target 参数为 0.20，Expert 相对 Ancestor 增加了 0.04，最终值就是 `0.20 − 0.5 × 0.04 = 0.18`。这个演算只说明差分方向与系数，没有使用实际模型参数。

### 参数局部变化怎样影响生成

对于同一个输入前缀，最终模型和 Target 使用相同的主干参数。因此，在同一个隐藏状态 $h$ 上，选中 token 的 logit 变化为：

$$
\Delta z_j = -0.5\Delta W_j h,\qquad j\in S.
$$

这个式子说明，效果取决于权重差分与当前上下文表示的内积。系数为负并不意味着所有 SID 的分数都会降低：内积本身可能为正，也可能为负。

未选中行在同一个隐藏状态下的原始 logit 不变，但 softmax 的归一化分母会受到其他 token 分数变化的影响。生成过程又是自回归的，一旦选择了不同 token，后续上下文也会改变。因此，“只改输出头的一部分行”描述的是参数范围，不能理解成模型其他输出概率完全不变。

### 数值计算与导出

[合成器](07_接管/experiments/compose_selective_lm_head_task_vector.py)先检查三个输入模型的配置、词表大小、张量清单和形状，再进行计算。差分与缩放在 FP32 中完成，写入结果前转换回 Target 的参数类型。

导出后，程序重新计算预期权重，并逐张量检查保存结果。选中行应符合合成公式，其他行和其他参数应与 Target 一致。模型配置与 tokenizer 文件也从 Target 复制，使导出的目录仍对应原来的模型结构和词表。

## 推理与评测

### 单模型生成

部署时加载导出的最终模型。R0 输入物料描述，生成候选 SID；R2 输入用户历史与推荐指令，生成下一物料；R1、R3 则按各自任务输出主题、判断或选择题答案。

初赛的 R0 使用 non-thinking beam64；R2 合并 thinking 与 non-thinking 各 32 个候选，按 64 个候选是否命中计分。这些是评测侧的生成设置，离线合成本身不增加推理分支，也不需要在每次请求中加载 Ancestor 或 Expert。

### 最终成绩

| 指标 | 分数 |
|---|---:|
| **总分** | **1.0716429387157187** |
| R0：物料理解 | 0.2452961672473868 |
| R1：用户理解，两项合计 | 0.14655643689584122 |
| R2：多域推荐，四域合计 | 0.5244 |
| R3：世界知识 | 0.1553903345724907 |

总分为八个子任务的加权贡献之和，上表将其汇总到 R0–R3 四组。R0 的权重为 17.6；R1 两项各为 0.3；R2 视频、电商、广告、直播依次为 9.6、3.4、1.4、0.9；R3 为 0.3。分项已经乘过权重，不能当作原始准确率，也不能根据柱高判断某个模块带来的增益。

成绩对应 EXP-182 模型的 EXP-201 评测，数据见 [results.json](docs/results.json)。

## 代码导航

源码按“启动训练 → 修复梯度路径 → 合成输出头 → 核对导出结果”组织。阅读最终方法时，可以先从合成器中的 `compose_head` 开始，它直接实现了上面的分段公式。

| 文件 | 负责的步骤 |
|---|---|
| [训练配置](07_接管/experiments/configs/exp182_headfix_reproduction_step10_fp16_xformers_ligerfix.yaml) | 冻结范围、数据模板、10 步训练参数 |
| [run_frozen_head_training.py](07_接管/experiments/run_frozen_head_training.py) | 加载补丁并启动训练进程 |
| [liger_frozen_head_patch.py](07_接管/experiments/liger_frozen_head_patch.py) | 冻结主干情况下的局部梯度处理 |
| [compose_selective_lm_head_task_vector.py](07_接管/experiments/compose_selective_lm_head_task_vector.py) | a/b/c 行上的负半幅差分合成 |
| [merge_model_weights.py](07_接管/experiments/merge_model_weights.py) | 模型读取、目录约定和兼容性检查 |
| [finalize_exp182_ligerfix.py](07_接管/experiments/finalize_exp182_ligerfix.py) | 训练步数与最终模型检查 |

合成器的调用形式如下。三个目录应包含相互兼容的 `model.safetensors`、配置与 tokenizer 文件，路径需满足辅助模块的工作区约定。

```bash
python 07_接管/experiments/compose_selective_lm_head_task_vector.py \
  --ancestor /path/to/ancestor \
  --target /path/to/exp174 \
  --expert /path/to/head-expert \
  --groups a,b,c \
  --alpha=-0.5 \
  --output-name exp182-negative-half-abc
```

输出模型保存在 `07_接管/experiments/selective_task_vector_models/`，对应记录保存在 `manifests/`。输出名称需要尚未使用；完整权重与训练数据需另行准备。

## 相关资料

- [OneReason 竞赛基座](https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition)
- [竞赛数据与字段说明](https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition)
- [OneReason Technical Report](https://arxiv.org/abs/2606.06260)

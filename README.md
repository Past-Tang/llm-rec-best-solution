# LLM-Rec：输出头微调与选择性权重合成

本方案在推荐模型上微调输出头，再将训练得到的权重变化反向应用到指定的 SID token 上。最终模型为 EXP-182，正式评测最高分为 **1.0716429387157187**，对应 EXP-201 评测记录。

## 方案思路

推荐对象通过 SID（语义标识符）表示，模型需要生成对应的 token。输出头 `lm_head` 将隐藏状态映射为词表分数，其中 a、b、c 三组 token 对应本方案调整的 SID 区域。

训练和合成分开进行。先从 aligned-replay 模型出发，冻结 Transformer 主干，只训练 `lm_head`，得到一个输出头更新方向。随后以 EXP-174 为目标模型，把这一方向乘以 −0.5，应用到 a、b、c 三组 SID 对应的行。目标模型的其余参数保持原值。

这样可以单独调整 SID 的输出分数，而不必把此次训练产生的全部参数变化写入目标模型。

## 输出头训练

训练使用 `takeover_r0_cbalanced_replay_v1` 数据，模板为 `qwen3_nothink`。主干冻结后，仅输出头参与参数更新，训练共 10 步。

| 参数 | 设置 |
|---|---|
| 可训练模块 | `lm_head` |
| 学习率 | `1e-5` |
| 学习率调度 | cosine，warmup ratio 为 0.03 |
| 单卡 batch size | 1 |
| 梯度累积 | 16 |
| 最大序列长度 | 32768 |
| 精度 | FP16 |
| Attention | xFormers |
| 数据打包 | packing、neat packing |
| 随机种子 | 20260720 |

完整参数见 [训练配置](07_接管/experiments/configs/exp182_headfix_reproduction_step10_fp16_xformers_ligerfix.yaml)。

### Liger 梯度修复

冻结主干后，输入输出头的隐藏状态不需要梯度，但 `lm_head` 仍然需要更新。这里使用的 Liger 0.6.4 fused cross-entropy 实现依据输入的 `requires_grad` 决定是否分配权重梯度，因而需要额外处理这种情况。

[补丁](07_接管/experiments/liger_frozen_head_patch.py)在输出头可训练、输入不需要梯度时执行：

```python
_input = _input.detach().requires_grad_(True)
```

新的叶子张量支持局部反向传播，同时切断通往冻结主干的梯度路径。[训练入口](07_接管/experiments/run_frozen_head_training.py)通过 `sitecustomize` 将补丁传入训练子进程，并校验对应 Liger 源文件的哈希。

## 选择性权重合成

合成使用三个模型：`ancestor` 是输出头训练的初始模型，`expert` 是训练 10 步后的模型，`target` 是 EXP-174。对选定的输出头行，计算：

```text
W_final = W_target - 0.5 × (W_expert - W_ancestor)
```

三个 SID 分组使用左闭右开的词表区间：

| 分组 | token 区间 |
|---|---|
| a | `[151669, 159861)` |
| b | `[159861, 168053)` |
| c | `[168053, 176245)` |

[合成代码](07_接管/experiments/compose_selective_lm_head_task_vector.py)先以 FP32 计算差分，再转回目标参数的数据类型。未选中的输出头行，以及所有非 `lm_head` 参数，直接取自 `target`。保存后会重新计算并逐张量比较，检查输出是否符合合成公式。

```bash
python 07_接管/experiments/compose_selective_lm_head_task_vector.py \
  --ancestor /path/to/ancestor \
  --target /path/to/exp174 \
  --expert /path/to/exp182-head-expert \
  --groups a,b,c \
  --alpha=-0.5 \
  --output-name exp182-negative-half-abc
```

三个输入目录需包含兼容的模型权重与配置，路径需符合辅助模块的工作区约定。

## 评测结果

| 总分 | R0 | R1 | R2 | R3 |
|---:|---:|---:|---:|---:|
| **1.0716429387157187** | 0.2452961672473868 | 0.14655643689584122 | 0.5244 | 0.1553903345724907 |

分项与总分取自 [EXP-201 正式评测记录](docs/results.json)。

## 代码结构

| 文件 | 功能 |
|---|---|
| `07_接管/experiments/configs/` | 输出头训练参数 |
| `run_frozen_head_training.py` | 加载 Liger 补丁并启动训练 |
| `liger_frozen_head_patch.py` | 冻结主干场景下的梯度修复 |
| `compose_selective_lm_head_task_vector.py` | SID 分组上的权重合成 |
| `merge_model_weights.py` | 模型读取、兼容性检查与文件操作 |
| `finalize_exp182_ligerfix.py` | 检查训练步数、合成参数和输出张量 |

表中的 Python 文件均位于 `07_接管/experiments/`。依赖包括 PyTorch、Safetensors、PyYAML、LLaMA-Factory、Liger Kernel 和 xFormers。数据与模型权重需另行准备。

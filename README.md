# LLM-Rec：最高分方案代码整理

本仓库整理 LLM-Rec 比赛的 **EXP-182 / EXP-201 最高分方案**：冻结主干的输出头训练、Liger 梯度修复，以及选择性 `lm_head` task-vector 合成。

**历史最高正式成绩：1.0716429387157187。** EXP-201 是 EXP-182 模型的复评，不是另一次模型训练。这里保留已找回的方案代码与必要说明，不提供比赛数据、模型权重或平台凭证，也不承诺从空环境直接重现该分数。

## 成绩

| 记录 | 分数 | 说明 |
|---|---:|---|
| EXP-182 | 1.070878603725826 | 最高分模型的首次正式评测，来自历史交接记录 |
| EXP-199 | 1.0704834485560581 | 同一模型复评，来自终局审计记录 |
| **EXP-201** | **1.0716429387157187** | 同一模型复评，终局最佳 |

EXP-201 分项：R0 = 0.2452961672473868，R1 = 0.14655643689584122，R2 = 0.5244，R3 = 0.1553903345724907。

公开的精简成绩记录见 [`docs/results.json`](docs/results.json)。分数引用归档证据，没有在整理过程中重新提交评测。重复评测出现的差异不应被解释成算法改进。

## 方法

### 1. 冻结主干，只训练输出头

使用历史 aligned-replay 模型作为 ancestor，冻结 Transformer 主干，训练 `lm_head`。EXP-182 的最终配置采用：

| 参数 | 值 |
|---|---|
| 数据 | `takeover_r0_cbalanced_replay_v1` |
| 训练部分 | `lm_head` |
| 最大步数 | 10 |
| 学习率 | 1e-5，cosine |
| 单卡 batch / 梯度累积 | 1 / 16 |
| 序列长度 | 32768 |
| 精度 / Attention | FP16 / xFormers |
| 模板 | `qwen3_nothink` |
| Seed | 20260720 |

详见 [`exp182 配置`](07_接管/experiments/configs/exp182_headfix_reproduction_step10_fp16_xformers_ligerfix.yaml)。此配置是 Liger-fix 分支，不是同期尝试过的 chunked-CE 分支。

### 2. 修复冻结主干时的 Liger 梯度路径

历史 Liger 0.6.4 fused linear cross-entropy 对冻结 hidden states 的处理会影响可训练输出头的梯度。保留的补丁对输入建立 detached leaf，支持局部 fused backward，同时保持主干冻结；补丁使用上游源码哈希约束，不能无条件套到其他 Liger 版本。

- [`liger_frozen_head_patch.py`](07_接管/experiments/liger_frozen_head_patch.py)
- [`run_frozen_head_training.py`](07_接管/experiments/run_frozen_head_training.py)
- [`sitecustomize.py`](07_接管/experiments/frozen_head_liger_site/sitecustomize.py)

### 3. 对 SID 输出行施加 negative-half 更新

最终输出以历史 EXP-174 模型为 target，以前述 ancestor 和训练得到的 expert 定义更新方向：

```text
output[selected_rows] = target[selected_rows]
                      - 0.5 * (expert[selected_rows] - ancestor[selected_rows])
```

选定 a、b、c 三组 SID 对应的 `lm_head` 行，区间分别为 `[151669,159861)`、`[159861,168053)`、`[168053,176245)`。其他输出行和其余模型参数保持 target 的值。

核心实现：[`compose_selective_lm_head_task_vector.py`](07_接管/experiments/compose_selective_lm_head_task_vector.py)。它包含形状、有限值和保存后张量一致性检查。

调用接口示意（需要自行提供三个完整的兼容模型目录）：

```bash
python 07_接管/experiments/compose_selective_lm_head_task_vector.py \
  --ancestor /path/to/ancestor \
  --target /path/to/exp174 \
  --expert /path/to/exp182-head-expert \
  --groups a,b,c --alpha=-0.5 \
  --output-name exp182-negative-half-abc
```

路径还受原辅助模块的工作区约束；此示例用于说明历史接口，不是已验证的安装教程。

## 仓库内容

```text
07_接管/experiments/
  configs/                        EXP-182 Liger-fix 配置
  compose_selective_lm_head_task_vector.py
  liger_frozen_head_patch.py
  frozen_head_liger_site/          子进程补丁入口
  run_frozen_head_training.py      训练启动适配
  finalize_exp182_ligerfix.py      原历史审计逻辑
  merge_model_weights.py          合成辅助模块
  run_training.py                  原训练启动辅助模块
  audit_recovery_run.py            原审计辅助模块
docs/
  results.json                    精简成绩证据
  recovered-source-manifest.json   恢复代码的补丁时间与文件哈希
  supporting-source-manifest.json  辅助源码来源与哈希
```

## 来源和范围

2026-10-19 从保存的历史源码补丁中恢复了 EXP-182 的六个核心文件，恢复时应用了找到的相关后续补丁。辅助模块来自保存的初赛工作区；它们未被重新验证为最终训练环境的完整快照。服务器绝对路径已替换为 `/workspace/LLM-Rec` 等通用占位路径。

该模型经历过原工作卷丢失：部分中间权重、数据和运行态仍缺失。**找回核心源代码不等于找回完整模型或恢复全部实验资产。** 本仓库不以较低分的 EXP-158 代码冒充 EXP-182；也没有重新发明一个算法并标记为原实现。

本次只做源代码整理、脱敏及语法/文件检查，不做训练、推理、环境重建或成绩复现。原审计工具依赖的 manifest、LLaMA-Factory 工作树、数据和权重均需另外提供。Python 依赖主要包括 PyTorch、PyYAML、Safetensors、LLaMA-Factory、Liger Kernel 与 xFormers，具体历史环境限制以代码中的检查为准。

旧 Git 历史、平台 API 客户端、自动提交脚本、对话记录、凭证、训练数据和模型文件均不进入本仓库。代码来源与许可说明见 [NOTICE.md](NOTICE.md)。

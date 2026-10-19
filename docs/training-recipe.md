# 数据构造与模型训练

最终模型包含两条来源不同的参数链。EXP-174 提供模型主体；aligned-replay 模型及其输出头微调结果提供最后一次更新方向。两条链在选择性输出头合成处汇合。

```text
OneReason → EXP-136 → EXP-158 → EXP-173 adapter × (−0.5) → EXP-174
                                                               │
v8t → aligned-replay → 输出头训练 10 步 → 输出头差分 × (−0.5) ────┤
                                                               ↓
                                                    只更新 SID a/b/c 行
                                                               ↓
                                                            EXP-182
```

## 数据如何进入训练

实验代码位于 `07_接管/experiments/`。下表中的文件名均相对该目录。

| 阶段 | 代码 | 处理内容 |
|---|---|---|
| 公共混合语料 | `build_exp132_public091.py` | 将固定版本的公共数据转换成训练字段，保留推荐、物料、用户和知识任务 |
| Answer-only replay | `build_exp142_public091_answer_only.py` | 从答案中去除推理段，保留最终回答 |
| Caption / SID 对齐 | `build_r0_aligned_video.py`、`build_r0_aligned_replay.py` | 连接物料描述与 SID，保留 replay，并限制同一 SID 的视角数量 |
| c-balanced replay | `build_r0_cbalanced_replay.py` | 对末级 c 编码限制采样数量，同一 SID 保留一个视角 |
| EXP-158 训练集 | `build_exp157_answer_only_r0_compact.py`、`build_exp157_schema_v2.py` | 组合 144,162 条 R0 与 32,705 条 answer-only replay，并统一字段 |
| 无歧义 R0 | `build_exp161_external_unambiguous_r0.py` | 筛选最终 SID 答案，排除已有基线与评测集合中的内容 |
| EXP-173 训练集 | `build_exp173_dataset.py` | 组合 32,705 条 replay 与 5,174 条无歧义 R0，统一 `system` 字段 |

`build_r0_multiview.py`、`build_r0_scale_candidate.py` 是对齐构造的共享实现；多域对齐代码用于构造筛选时参考的基线集合。v8t 的早期数据与训练配方保存在 `04_作战/rft_v2/`。

## 主体模型的训练与合成

| 阶段 | 参数与选择 | 配置 / 实现 |
|---|---|---|
| EXP-136 | rank 128、alpha 16、rsLoRA、LoRA+ 比率 16、学习率 `1e-5`、652 步 | `configs/exp136_public091_v66_rslora_loraplus_r128_a16_xformers_16k_ga8_fp16.yaml` |
| EXP-157 → EXP-158 | 从 EXP-136 开始，rank 64、alpha 64、学习率 `5e-6`；训练配方为 2 轮，选用第 1 轮、step 543 | `configs/exp157_answer_only_r0_continuation_r64_a64_xformers_16k_ga8_fp16.yaml`、`finalize_exp158.py` |
| EXP-173 | 从 EXP-158 开始，rank 64、alpha 64、学习率 `5e-6`，训练 1 轮 | `configs/exp173_exp158_unambiguous_r0_r64_a64_xformers_16k_ga8_fp16.yaml` |
| EXP-174 | 将 EXP-173 adapter 的 LoRA B 矩阵乘以 `−0.5`，再合并到 EXP-158 | `prepare_exp174.py` |
| EXP-182 | 在 EXP-174 上应用独立输出头差分，仅修改 SID a/b/c 的 24,576 行 | `compose_selective_lm_head_task_vector.py` |

对于 LoRA 更新 `ΔW = sBA`，只缩放 B 就得到 `−0.5ΔW`。EXP-174 使用这次反向更新作为模型主体；EXP-182 随后的操作针对另一条训练链产生的输出头差分。二者不能合并解释为同一次训练。

## 代码中的路径与校验

配置中的 `/workspace/LLM-Rec` 是统一示例根目录。数据、模型和历史运行产物需由使用者提供；仓库保存算法与配方，没有附带比赛数据或模型权重。LLaMA-Factory、PEFT、Liger、xFormers 由外部环境提供。

构造器保留了原有的行数、内容哈希和排除集校验。`finalize_*`、`materialize_*`、`audit_*` 以及 `preregistrations/` 记录的是当时的实验验证流程，会读取历史 manifest 或评测记录。它们用于解释如何选择和核对产物，不能在缺少这些输入时当作一键运行入口。

本次补齐没有重新训练模型，也没有重新运行官方评测。源码出处与导出哈希见 [supplement-source-manifest.json](supplement-source-manifest.json)。

# V5 训练集验证记录

这些结果是代码修订的训练集证据。两份报告均为 `formal_eligible=false`，没有正式冻结、validation 或 blind 消耗。

| 检查 | 配置 | 结果 |
|---|---|---|
| 被动辨识安全门 | 4 seeds ×64 场景，H125 | 全部通过，Q2 动作及状态逐比特一致 |
| K35/物理上界 | 另 4 seeds ×128 场景，H126 | 全部训练 bank 通过 |
| 回归测试 | CPU，完整 pytest suite | 343 passed，4 skipped，129 subtests passed |
| 普通训练 smoke | CPU Torch，H16/B8，2 updates | loss 0.216318 /0.226824；梯度有限；无 skip |
| Pipeline 命令解析 | all stages，CPU，dry-run | 成功；未启动正式阶段 |

## 被动辨识

| Seed | 场景数 | Q2 parity 最大误差 | 支持覆盖 | 安全门 |
|---|---:|---:|---:|---|
| 3707 | 64 | 0.0 | 100% | 通过 |
| 4707 | 64 | 0.0 | 100% | 通过 |
| 5707 | 64 | 0.0 | 100% | 通过 |
| 6707 | 64 | 0.0 | 100% | 通过 |

覆盖只指共享 rise/fall tau 的离线支持条件，不能授权所有六个能力轴。

## 独立训练 bank 上的物理可行性

| Seed | 最大单轴 normalized-z RMS（call100/125） | 连续 observer 最差 p95 motor RMS | 最差 max motor RMS |
|---|---:|---:|---:|
| 13707 | 2.1e-05 | 2.477e-08 | 3.98e-08 |
| 14707 | 1.388e-05 | 2.343e-08 | 6.503e-08 |
| 15707 | 1.425e-05 | 2.776e-08 | 5.66e-08 |
| 16707 | 1.288e-05 | 2.582e-08 | 5.022e-08 |

连续 observer 上界以 privileged motor truth 在较早前缀拟合 tau，再保留 25 个转移预测评分。这里的约 1e-8 误差不是部署侧 identifier 的精度。实际学生仍须独立通过均值、连续 observer、校准、闭环和迁移门。

K15 在 seed16707 的 call100 相对覆盖检查失败；作为参考组，该失败保留。K35 最近离散候选的绝对误差失败也保留，未被改写为训练精度通过。部分 call25 的秩/条件数可能不足，对应未定义诊断量保存为 null；正式发布检查在 call100/125，仍要求逐场景满秩及既定误差界。

## 复现与边界

- Python 3.12；PyTorch 2.14.0+cpu；CUDA unavailable。
- 最终测试命令：`pytest -q --tb=short`。
- 训练 bank 复现命令见 [设计说明](passive_identification_v5.md)。
- 现有 4 项 skipped 测试没有被算作通过；CUDA 验证未执行。
- formal-freeze、identifier 正式训练、A1/A2/B/C、正式 MS、blind 均未运行。
- 原 Q2 checkpoint 未修改；未生成或推广新的 structured 权重。

原始报告：

- [probe_v5_train_only.json](../reports/probe_v5_train_only.json)，SHA-256 `97ea1f42384c9c5c4601733869a90b0f8f2cd31648b72bdee708dd2b3fef139d`。
- [passive_identification_v5_train_ceilings.json](../reports/passive_identification_v5_train_ceilings.json)，SHA-256 `cc324ab3ef375017c9e09995c9551644615e8cf0204f5e8211d696d89e918d65`。

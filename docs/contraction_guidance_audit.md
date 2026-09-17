# DiffPhys 物理条件 Metric：审计、实现与实验说明

审计基线：`bbdbd0c3449d5a1c388b1cba5e6e3ac8fd50c3ee`（2026-09-16 master）。
FiLM 参考：`ROBERTJEFFLV/A-Trial-on-RL` 的
`634a44ce009f9521764fed4282f24f24059fb6f5`，
`isaac-training/training/scripts/ppo.py:FiLMGatedFusion`。

## 结论和证据边界

建议先比较 **旧 context + concat → 完整 context + concat → 完整 context + FiLM**，
再逐项加入瞬态约束、尾部损失和辅助梯度限幅。
FiLM 改变条件信息怎样影响特征；它不负责机型均衡更新，也不会自行保证闭环稳定。

用户提供的单 seed A/B/C 数据显示 C 后期改善生存率，但不是全面改善悬停精度。
这些长跑检查点和原始日志不在本次环境中，不能据此定位每一次尖峰的原因。
不同更新数上的最佳模型不能视为严格的等预算比较。
本次修复不能被解释为已证明优于 C；最终证据仍须来自用户 GPU 上的对照。

## 查到的问题

| 项目 | 源码证据 | 影响与处理 |
| --- | --- | --- |
| 全长度行选择忽略索引 | `_select_rows` 只检查索引数量等于行数即返回原张量 | 全排列和重复采样返回错误行；两个回归用例在旧代码失败。修复通用选择，同时保留 rollout 的有序唯一索引零复制路径。默认 8/512 抽样不一定触发，不能称为原长跑失败根因。 |
| 物理指导已存在，但不完整 | `StateGeometry.context` 原本包含 20D 质量、惯量、电机时间常数、推重比、力矩惯量比、外力和外力矩 | 原假设“MLP 没有物理输入”不成立；独立改变 `rotor_torque_constant` 会改变真实偏航力矩，却不改变旧 context。补充完整条件输入。 |
| 瞬态放大不参与优化 | 原 `contraction_loss` 只对端点 ratio 求损失；prefix 最大增益仅报告 | 中间先放大、最后缩小可躲过端点目标；增加独立于 Metric 的真实前缀增益惩罚，可单独关闭。 |
| 抽样均值不能约束最坏方向 | 默认 8 场景 × 2 方向，损失取平均 | 添加抽样尾部聚合和质量分位覆盖；这仍不是全方向检查，已有 `worst_direction` 仅是单状态端点审计。 |
| 辅助梯度影响不透明 | 原先记录辅助梯度范数，没有同时记录 task 范数、两者夹角和实际占比 | 新增合并前诊断及可选 norm cap；Metric 梯度可独立缩放。 |

最新 master 已有的数值截断检查、真实 JVP、GRU 等价后端、终止前缀和回滚保护均保留。
终止/dead 分项由离散事件产生，对终止时刻没有梯度，这是现有目标的限制而非新增故障。
它们会影响总分和 CVaR 选样，但不会直接给 Actor 提供“如何避免终止”的平滑梯度。

## 已实现的条件输入与结构

67D context 的固定顺序如下；旧 20D 完整保留在前部。

| 输入 | 维数 | 处理 |
| --- | ---: | --- |
| 旧物理参数 | 20 | 沿用原归一化 |
| 机臂长度 | 1 | SI 数值的对数 / 5 |
| 四个旋翼位置 | 12 | 除以机臂长度 |
| 四个旋翼反扭矩系数 | 4 | 除以机臂长度 |
| 四个电机的二次推力多项式 | 12 | 改写到归一化电机坐标，再除以 `mass*g` |
| 观测噪声标准差 | 18 | 按对应状态坐标尺度归一化 |

噪声输入是传感器配置，不是额外真值传感器。当前 RAPTOR 的 18D 噪声配置全部为零，
因此这些通道在该协议内没有机型区分信息，也不代表已学会随机噪声鲁棒性。
配对扰动仍共享同一噪声序列；没有把质量、惯量或电机真值加入部署 Actor。

将状态编码为 `h`，物理条件产生 `gamma, beta`；门控同时读取 `h` 和物理条件：

```text
gate = sigmoid(Linear(concat(h, physical_context)))
h_mod = h + gate * (0.5*tanh(gamma)*h + 0.5*tanh(beta))
M = bounded_SPD_head(h_mod)
```

这是适配 Metric 的残差 FiLM，借鉴参考仓库的有界缩放/平移和门控思想。
没有搬入其 PPO、静态/动态障碍编码器、`[0,1]` 配置截断或 LayerNorm：
物理 context 含有负的对数值，直接照搬截断会损坏参数信息。
Metric 继续满足 `0.5 I <= M <= 2 I`；它从辅助损失经真实短窗 JVP 指导 Actor，
不会直接替换 H500 task BPTT 的导数。

默认 memory=64、hidden=64、rank=4 时：旧 Metric 49,072 参数，
完整 context 的 concat 52,080 参数，完整 context 的 FiLM 68,912 参数。
后两者容量不同，所以 FiLM 实验不能单独证明收益来自门控结构；必要时增加容量匹配对照。
这些参数数目不能换算为 GPU 训练耗时，本次没有 GPU 性能结论。

## 损失与梯度开关

设 `r` 为原始端点耗散比，`G` 为同一扰动方向在所有实际前缀中的最大归一化欧氏范数平方增益：

```text
L_pair = relu(log(r))^2 + prefix_weight * relu(log(G / prefix_max_gain_squared))^2
L_aux  = mean(worst ceil(tail_fraction * N) sampled pair losses)
```

前缀项检查归一化后的物理、电机、历史和 GRU 状态，不是米单位的位置误差。
只计实际执行的转移，包括首个失败转移；冻结填充不参与。
默认平方增益预算 4 相当于范数允许放大到 2 倍，是实验参数，未证明对所有机型可达。
惯性与电机滞后可能导致必要的短暂放大；不应强制每一时刻所有方向都缩小。
由于 Metric 上下界比为 4，即便端点 ratio 达标也不代表欧氏范数端点完全不放大。

| CLI 开关 | 默认 | 用途 |
| --- | --- | --- |
| `--contraction-context` | `legacy` | `physics` 启用完整 67D |
| `--contraction-fusion` | `concat` | `film_gated` 启用条件调制 |
| `--contraction-prefix-weight` | `0` | 打开真实前缀放大惩罚 |
| `--contraction-prefix-max-gain-squared` | `4` | 前缀允许的平方增益 |
| `--contraction-tail-fraction` | `1` | 如 `0.25` 关注最差的抽样四分之一 |
| `--contraction-sampling` | `uniform` | `mass_quantiles` 等数量覆盖当前 TRAIN 批次质量分位 |
| `--contraction-actor-max-ratio` | `0` | 如 `0.5` 限制辅助梯度范数不超过 task 的一半 |
| `--contraction-metric-gradient-scale` | `0` | `0` 沿用原乘数；正数独立设置 Metric 梯度乘数 |

训练采样的质量覆盖不修改 TRAIN 场景分布或任务 CVaR，不等于不同尺寸贡献相等。
诊断 EVAL 继续使用固定均匀子集，避免改变训练覆盖开关时偷偷改变评估分布。
配置、网络、两个 Adam 和抽样进度都写入检查点，精确 resume 仍严格核对源码和配置。
v1 的旧 Metric 可以按旧配置重评分；跨代码/架构实验使用 `--init-checkpoint`，不能伪造 source hash 强行 resume。

旧合并式为 `0.1*(g_task + 0.1*g_aux)`：辅助分支的绝对系数是 0.01，
相对 task 的系数是 0.1，不是 0.01。实际影响取决于两条梯度的范数和夹角。
Adam 对整体梯度缩放近似不敏感，epsilon、裁剪和时变缩放会打破这个近似；
单纯把 0.01 改成 1 不能保证更快学习。
norm cap 也不是 Adam 更新量的上界，没有保证 task 单调改善。

## 最小对照顺序

| 组 | context / fusion | 其它辅助开关 | 回答的问题 |
| --- | --- | --- | --- |
| C0 | legacy / concat | 原默认 | 修改后旧 C 基线 |
| P | physics / concat | 同 C0 | 补全物理信息是否有帮助 |
| F | physics / film_gated | 同 C0 | 改变条件融合方式是否有帮助 |
| 后续单项实验 | 选中的结构 | 每次只改 prefix、tail、sampling 或 cap 之一 | 收益来自哪个训练机制 |

单 GPU、单 seed=7，顺序运行。所有组从相同 Actor 权重开始，使用相同 TRAIN/EVAL，
统一采用新的 Adam 和新 Metric；不要把续训旧 C 与新初始化 F 直接比较。
新配置 `configs/response_raptor_film_metric.args` 只开启 context 与 FiLM，
其余新增指导默认关闭，避免一次叠加所有改动。

先在实际 GPU 上跑已有 CUDA 回归，确认高阶导数支持，再启动明确预算的试跑：

```bash
python -m pytest -q tests/test_contraction.py tests/test_contraction_guidance.py
python tools/train_response_control.py $(cat configs/response_raptor_film_metric.args)
```

该配置最多 50 updates / 1800 秒，主要检查执行与耗时，不能证明长期效果。
P/C0 可以覆盖 `--contraction-fusion concat` 及 `--contraction-context legacy`，
每组必须给新的 `--work-dir`。若使用成熟 Actor，三组均传入同一个 `--init-checkpoint`，
并核对其机型协议、网络、dtype 与现有配置兼容。
性能比较同时报告等 update 与等墙钟时间；C 在原实验中很晚才超过 A，不能用短试跑断言 FiLM 优劣。

验收优先看固定 EVAL 的提前终止率、位置/速度/角速度 RMS、饱和风险，再看辅助统计。
不能只因为 Metric loss 降低就宣布稳定性改善。
两条梯度的新增日志：`task_gradient_norm`（已乘全局 scale）、`actor_gradient_norm`
（未加权辅助梯度）、`auxiliary_task_norm_ratio`、`task_auxiliary_cosine`、`actor_gradient_capped`。

## 验证记录

环境：CPU，PyTorch 2.14.0+cpu；没有 CUDA 和长跑检查点。
旧收缩相关测试在修改前为 41 passed / 1 CUDA skipped。
全长度行选择的排列和重复索引测试在修改前均失败，修复后通过。
默认配置相对基线源码的独立比较：L2F loss=0.1768148935967079，
RAPTOR loss=0.03442104200910484；两者损失逐位相同，Actor/Metric 最大梯度差均为 0。

新验证覆盖：物理 context 可辨识性和推力单位转换、两种融合的 SPD 界与批量前缀，
L2F/RAPTOR FiLM+前缀+尾部目标的 Actor 混合导数有限差分，
前缀冻结屏蔽、尾部聚合、独立 Metric 梯度、质量覆盖和 RNG、精确断点续训，
v1 重评分/禁止不兼容续训、部署 Actor 隔离。
最终 `python -m pytest -q`：167 passed，4 CUDA skipped，耗时 59.33 秒。
`git diff --check` 通过；单元测试不代表学得更好或可部署飞行。

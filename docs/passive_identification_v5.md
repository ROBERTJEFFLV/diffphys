# V5：从必须主动激励，改为先检查已有信息

本修订基于 `8de9385a524455e3627d5dd479bc35df902ba285`。Q2 权重、模拟器、
原 v4 波形和 Q2 +5% 安全门保留。V4 的失败结论仍成立；v5 是新的辨识协议，
不能用 v5 报告把旧实验改判为通过。当前代码不是已训练并获准部署的新控制器。

## 1. 为什么可以先不加探针

电机模型是四个转子共用两个时间常数的分段一阶系统：

```
m[t+1] = m[t] + min(dt / tau_branch, 1) * (u[t] - m[t])
branch = rise if u[t] >= m[t], else fall
```

辨识需要的是足够且对齐的输入变化与响应，并不要求这些输入变化一定由附加探针产生。
Q2 在恢复阶段本来就会改变四个电机指令。V5 先使用这段自然激励，默认附加指令为零；
如果已完成的历史不足，则停止相应能力的使用，而不是为了满足固定的探针形状继续扰动。

正式配置收集前 100 个完整物理转移，在 call100 首次发布能力，此后每 25 步发布。
call100 使用的是转移 0…99 的响应；收集 call125 的输出需要 126 行观测。
这是一份明确的新启动合同，并不声称 100 步是数学上的最短辨识时间。

`probe_contract_v5.py` 注册训练 seeds 3707/4707/5707/6707、64 场景的 4×4
能力分层、H125、单独的 validation seed7707 及契约摘要。默认诊断只运行训练集。
正式冻结必须额外指定 `--formal-freeze`，且先通过全部训练检查。

默认路径满足逐比特动作和状态一致性，因而没有额外探针对 Q2 造成的安全退化。
这没有放宽 1.05 倍门。报告仍检查全局及每个能力单元的均值、池化 p99、
完整轨迹和 H75/H125 窗口，保留所有失败场景。

可选 collective 探针仅供实验：四个电机增加同一个标量，在所有电机的
box/rate/trust 约束交集中投影。逐电机裁剪可能把共同升力变化变成差分力矩，
因此不采用它。实验有持续 abort 状态和恢复观察窗口，不能生成正式冻结记录。
即便共同指令也不保证零力矩，因为当前转速、rise/fall 分支和推力截断可能不同。
本次开发中主动候选仍有安全或覆盖失败，没有将它们推广为正式方案。

## 2. 共享时间常数不等于六维能力可辨识

离线共享 tau 检查使用实际转移，并检查两个分支的样本数、时间覆盖和多电机贡献。
继承的 `weighted_fisher_information` 字段只是无噪声模型的加权激励代理，
没有指定观测噪声模型和 nuisance 参数消元，不能称为统计 Fisher 信息证书。
V5 的新探针报告将它明确标为 `weighted_excitation_proxy`。

`identification_information.py` 增加部署侧必要条件：使用已执行指令的创新和
测得的响应，积累模态回归量的中心化 Gram 矩阵及 rise/fall 能量。
条件投影避免把共线的运动误认为额外激励。共同升力信息不能授权角向能力和惯量；
惯量还需要提供独立信息的陀螺耦合项。

没有支持的物理能力轴采用先验值及保守 UCB；适应性增益和二阶 residual 还需
校准有效、信息支持及现有宽度/失败门共同允许。原始均值输出保留给 A1 监督，
避免“先通过 A2 校准才能训练 A1”的循环依赖。这些检查只是必要条件；
有激励不能自动证明 identifier 拟合正确或闭环稳定。

信息记录在启动后冻结，匹配当前每个 episode 参数固定的模拟器假设。
如果真实系统在飞行中更换载荷或发生电机故障，必须另行设计变化检测、
重新辨识和回退流程。本修订没有把启动信息冒充时变参数的长期证书。

## 3. 修复模型与状态的一致性

- **Q2 观测**：加载器保存 checkpoint 的观测坐标、积分限幅、泄漏等设置。
  DAgger 教师使用自己的原始积分，学生的 anti-windup 状态不会改写教师。
- **实际动作**：DAgger 从同一个输入 state，用实际执行动作调用生产状态转移，
  包含 observer、motor bank、积分、激励历史、probe 状态和启动指令历史。
  第二次计算不会使用已经推进后的 state，因此每个物理转移仍只消耗一个 call。
- **连续电机状态**：call100 用预测的连续 rise/fall tau 重放已保存的启动指令，
  消除固定初始 tau 留下的状态偏差，然后继续每步一阶 observer。
  这里不读取真实电机状态；初值来自 previous_action。
- **真实推力方程**：模拟器的单电机归一化推力为
  `max(0, 1 + (TW - 1) * m)`，高 TW 时不能全程当作线性函数。
  物理拟合改为分段仿射求解，力矩用实际截断后的转子推力与 full-range authority
  归一化；仍使用 midpoint omega 处理惯量耦合。
- **慢速外力估计**：保存一个 cadence 的方向、速度增量和 motor estimate，
  在新 TW 到达后按相同截断方程重新计算。旧的两个线性统计量无法正确处理
  TW 更新时跨越零推力折点的问题。

K35 是 identifier 的离散输入特征集，不是部署电机状态只能选取的 35 个值。
因此新版合同分别检查 K35 相对 legacy 的表示覆盖，以及**连续 observer**的
物理上界。旧的最近候选绝对误差仍完整保留在 `nearest_mode_failed_checks`。
连续上界只用较早前缀拟合 tau，末尾 25 个转移留作预测评分，不用未来真值重置状态。
这项拟合允许使用 privileged motor truth，是可行性上界，绝不是训练精度。
正式 A1 另行要求真实生产 observer 在 call100/125 的 motor RMS 误差满足
p95 ≤0.001、max ≤0.005；缺少该证据会失败。原有六维均值、平衡点和迁移门继续执行。

## 4. MS 和稳定性中容易遗漏的自由度

完整 codec v3 增加启动指令、信息状态、probe 记录和外力估计窗口；旧 codec
只能通过显式兼容路径读取，不能继承新的能力授权。完整状态单样本 Jacobian
也按所有字段切分，避免新增历史字段保留整个 batch。

SO(3) 残差在 identity 的导数恢复为正确的切空间映射，并处理接近 pi 的轴恢复。
pi 仍是对数映射的切割，不声称那里光滑。`log(predicted.T @ actual)` 对 actual
切向量的导数为 +I，对 predicted 为 -I。

MS 边界不能更改信息账本、已执行启动指令、计时及离散授权等事实。
这些列固定，真正动态的 observer、外力窗口和其他 recurrent state 保留连续性约束。
求解后的连续轨迹恢复、held-out 接受门及校准失效流程继续保留。

只有实测 JVP 满足中性对称方向的 yaw basis 才可被投影掉。当前带绝对姿态输入的
MLP 不天然保证 yaw 等变，不能直接声明一个“yaw gauge”来隐藏不稳定特征值。
后校准检查已迁移到完成辨识后的 call125。

## 5. 可复现检查与正式顺序

在已安装项目依赖的 Python 环境内，从仓库根目录运行：

```bash
python tools/diagnose_probe_v5.py --output tmp/probe_train_new.json --n-jobs 4
python tools/audit_passive_identification_train.py --output tmp/ceilings_train_new.json --n-jobs 4
pytest -q
python tools/run_structured_pipeline.py --stage all --device cpu --dry-run
```

训练用 K35 审计 seeds 为 13707/14707/15707/16707，每个 128 场景、H126。
与探针开发 seed 分离。该工具不会运行 validation、blind 或 optimizer，输出始终
`formal_eligible=false`，不能用作正式发布凭据。

完成代码审阅并决定正式消耗 seed7707 后，冻结当前协议：

```bash
python tools/diagnose_probe_v5.py --formal-freeze --output reports/probe_v5_formal.json --n-jobs 4
python tools/run_structured_pipeline.py --stage all --device cuda
```

冻结通过独占创建 claim 防止同一工作区重复消耗 validation；消费者核对契约、
Q2、代码摘要、报告摘要、实际数值及正式布尔字段。它不是跨机器的防重复评估服务：
复制仓库或删除 claim 不能制造新的独立 validation。源码、观测或状态规则变化后，
旧校准/部署摘要失效，不能只修改报告中的 passed 字段。

本次没有运行上述 formal-freeze 或正式流水线。审计结果见同目录验证说明及
`reports/probe_v5_train_only.json`、`reports/passive_identification_v5_train_ceilings.json`。
当前只有 CPU；没有 CUDA、训练后 identifier、新控制器 H500/H1000 性能或 blind 泛化结论。

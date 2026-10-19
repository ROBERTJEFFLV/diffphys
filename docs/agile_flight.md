# 三点机动飞行：数学、训练和悬停权重迁移

本说明对应 `Agile-Flight-training` 分支的 `--task agile`。任务是：从悬停出发，按照给定位置、姿态和非零速度经过中间点，再减速到终点并保持悬停。训练采用连续时间参考和连续损失；不设置“通过中点才切阶段”“悬停达标才结束”或“成功率达到门槛才升课程”等判断。

训练链路仍然是 **Actor → 四路电机命令 → 可微物理 → 任务损失 → Actor 参数更新**。没有增加 critic、教师网络、规划控制器或辅助模型。原有悬停任务保留，选择 `agile` 才启用这里的目标与课程。

## 1. 三个点怎样放进原有边界

对第 $i$ 架飞机，原有硬边界仍是世界坐标中的立方体：

$$
|p_x|,|p_y|,|p_z|\le B_i,\qquad B_i=30l_i,
$$

其中 $l_i$ 是该飞机的 `arm_length`，实际计算直接读取原来的 `position_limit`。目标变化不会平移、放大或重定义这个边界。

取中心 $\mathbf c$、水平前进单位向量 $\mathbf e$、与其垂直的水平侧向单位向量 $\mathbf n$，定义

$$
\mathbf d=h\mathbf n+H\mathbf e_z,
$$

$$
\boxed{\mathbf p_0=\mathbf c-L\mathbf e,\quad
\mathbf p_m=\mathbf c+\mathbf d,\quad
\mathbf p_f=\mathbf c+L\mathbf e.}
$$

`half_span` 对应 $L$；$h$ 控制侧向弯曲，$H$ 控制中点相对起终点的高度。名义路径为

$$
\boxed{\mathbf p(q)=\mathbf c+L(2q-1)\mathbf e+4q(1-q)\mathbf d,
\qquad q\in[0,1].}
$$

它是二次 Bézier 曲线，三个**控制点**为

$$
\mathbf C_0=\mathbf c-L\mathbf e,\quad
\mathbf C_1=\mathbf c+2\mathbf d,\quad
\mathbf C_2=\mathbf c+L\mathbf e.
$$

中间控制点是 $\mathbf c+2\mathbf d$，并非实际途经点 $\mathbf c+\mathbf d$。代码检查所有控制点满足

$$
\|\mathbf C_j\|_\infty\le(1-\mu)B_i,
$$

默认 `boundary_margin_fraction` $\mu=0.2$。Bézier 权重非负且和为 1，因此**整条名义参考曲线**都在这三个控制点的凸包内，包含连续时间中间位置。

这是参考曲线的几何性质。真实 Actor 可以偏离参考；实际越界仍由原来的 `RaptorSimulator.position_terminated` 判断。余量也不是对高斯扰动、碰撞或实飞安全的保证。

## 2. 时间、通过速度和中点姿态的关系

运动段时长为 $T$，取 $u=t/T$，用九次平滑时间函数：

$$
q=S_9(u)=126u^5-420u^6+540u^7-315u^8+70u^9.
$$

在 $t=0,T$ 时，位置的一至四阶导数均为零，包括速度、加速度、jerk 和 snap。因此无外力名义参考在起终点为水平悬停，角速度和角加速度也为零。$t\ge T$ 后参考位置保持在终点。

在 $t=T/2$ 时：

$$
\boxed{\mathbf v_m=\frac{315L}{64T}\mathbf e,\qquad
\mathbf a_m=-\frac{2\|\mathbf v_m\|^2}{L^2}\mathbf d.}
$$

令 $K=2\|\mathbf v_m\|^2/L^2=99225/(2048T^2)$。无外力时需要的单位质量推力为

$$
\mathbf f_m=\mathbf a_m+g\mathbf e_z
=-Kh\mathbf n+(g-KH)\mathbf e_z.
$$

机体推力轴沿 $\mathbf f_m$；在其竖直分量为正时，中点倾角满足

$$
\boxed{\tan\theta_m=\frac{K|h|}{g-KH}.}
$$

因此速度、曲率、高度和姿态不能独立乱抽。中点通常还有非零角速度：飞机正在改变倾斜方向，不能仅因为它位于途经点就把角速度参考设为零。

完整姿态由推力轴和固定偏航角共同确定。令 $\mathbf f=\ddot{\mathbf p}+g\mathbf e_z$，代码使用

$$
\mathbf b_3=\frac{\mathbf f}{\|\mathbf f\|},\qquad
\mathbf b_1=\frac{f_z\mathbf e-(\mathbf f\cdot\mathbf e)\mathbf e_z}
{\|f_z\mathbf e-(\mathbf f\cdot\mathbf e)\mathbf e_z\|},\qquad
\mathbf b_2=\mathbf b_3\times\mathbf b_1,\quad
R=[\mathbf b_1\;\mathbf b_2\;\mathbf b_3].
$$

在 $f_z>0$ 的适用范围内，机体前轴的水平投影与 $\mathbf e$ 同向。中点姿态不是把机体前轴直接对齐速度再任意添加倾角。角速度由 $[\boldsymbol\omega]_\times=R^T\dot R$ 求得，角加速度继续由解析时间导数求得；筛选器排除近自由落体、推力方向退化或进入倒飞范围的名义参考。

### 默认的 60° 和推力比 1.2

`tilt_degrees=60` 与 `thrust_ratio=1.2` 定义的是**请求中点姿态及该点总推力与重量的比值**。后者不是把飞机本身的推重比改为 1.2。默认构造

$$
\mathbf D=\eta\sin\theta\,\mathbf n+(1-\eta\cos\theta)\mathbf e_z,
$$

$$
L=0.45B_i,\qquad
\mathbf d=0.2B_i\frac{\mathbf D}{\|\mathbf D\|},\qquad
T_*^2=\frac{99225}{2048}\frac{0.2B_i}{g\|\mathbf D\|}.
$$

理想连续时间下，这给出

$$
\mathbf a_m=g[-\eta\sin\theta\,\mathbf n+(\eta\cos\theta-1)\mathbf e_z].
$$

60°、1.2 对应中点向下加速度 $0.4g$，所以是带高度变化的倾侧机动；不能把它当成恒定高度的 60° 转弯。恒高时仅中点就需要 $1/\cos60^\circ=2$ 倍重量的总推力。

实现会把运动时长量化到偶数个 100 Hz 控制步，让中点恰好落在网格上，并给末段悬停保留时间。默认 H500 共 5 秒，`hold_seconds=1`，故运动段最多 4 秒；小飞机的运动段通常更短。量化、时间上限和执行器限制后的实际中点速度、倾角会重新计算并写入报告。

本版本覆盖对称倾侧转弯和斜向跃升。`yaw_degrees` 可以固定航向；`center_fraction` 可以改变相对于原边界的中心位置。它不支持任意三个位置与任意三个姿态完全独立指定，也不包含翻滚一周或倒飞等动作。

## 3. 同一架飞机怎样适配难度

训练课程给出请求幅度 $\lambda_{\rm req}$。对同一架已经抽好的飞机，保持 $T$ 不变并缩放

$$
L_\lambda=\lambda L,\qquad \mathbf d_\lambda=\lambda\mathbf d.
$$

速度和加速度都随幅度线性变化；姿态从新的加速度重新求解，不能把角度直接当成 $\lambda\times60^\circ$。

任务生成器按固定的缩幅表检查这架飞机，默认乘数为 `1, 0.8, 0.6, 0.4, 0.25, 0.125, 0.0625, 0.03125, 0`。采用第一个通过**名义物理检查**的幅度。这里检查的是参考命令是否超出执行器能力，不是判断训练任务是否“学会”。不会为得到漂亮结果而重抽飞机、换噪声或删除困难物理组。

检查在每个 100 Hz 网格时刻计算总推力、角速度和角加速度，再求出所需转矩及四个电机各自的推力；这里没有新加角速度硬终止条件。使用原来的混控矩阵求解

$$
\begin{bmatrix}
F\\ \tau_x\\ \tau_y\\ \tau_z
\end{bmatrix}
=
\begin{bmatrix}
1&1&1&1\\
y_1&y_2&y_3&y_4\\
-x_1&-x_2&-x_3&-x_4\\
-k_{m1}&k_{m2}&-k_{m3}&k_{m4}
\end{bmatrix}
\begin{bmatrix}f_1\\f_2\\f_3\\f_4\end{bmatrix},
\quad
\boldsymbol\tau=I\dot{\boldsymbol\omega}
+\boldsymbol\omega\times(I\boldsymbol\omega).
$$

每个电机采用源代码中的推力曲线 $f=c_0+c_1m+c_2m^2$。特别地，`motor=0` 的推力是 $c_0$，不能把它近似成零。

相邻网格需要的电机状态为 $m_k,m_{k+1}$。按状态升降选择原来的上升/下降时间常数 $\tau$，记 $x=\Delta t/\tau$，则原版 RK4 对该一阶电机方程的系数为

$$
r=1-x+x^2/2-x^3/6+x^4/24,
\qquad
m_{{\rm cmd},k}=\frac{m_{k+1}-r m_k}{1-r}.
$$

代码还检查这些设定值位于电机有效范围内，并留出默认 1% 的执行器余量。这个检查覆盖原控制网格上的电机响应，不证明完整非线性飞机会精确跟踪，也不证明连续时间中所有执行器峰值都被捕获。

即使全部非零幅度都不可用，飞机仍保留，使用可表示的静止回退并记录原因。不可用的常力悬停也会记录；不会通过裁剪外力或替换飞机让该记录消失。`feasible` 表示这种名义构造检查的结果，不是学习成绩。

训练配置的 `hover_fraction=0.1` 额外保留约 10% 的主动悬停练习，用于让模型持续练习静止控制。主动悬停与无法生成非零幅度的静止回退分开记录；两类记录都保留在原始飞机集合中。

## 4. 连续损失与连续时间进度

运行时按照时间读取 `mission.nominal_reference(time)`，得到参考位置、速度、姿态和角速度；输入既支持每架飞机各自的 `[N]` 时间，也支持时间网格。时间进度连续增加，到运动段末尾后参考保持在终点。

Actor 看到测量状态、公开目标和连续时间进度。输入中的 `phase` 是 $\operatorname{clip}(t/T,0,1)$，位置参考由它经过 $S_9$ 后生成；其阶段信息不由仿真真值、“已经通过”或“悬停达标”触发。全程损失跟踪参考位置、速度，中点时间窗额外强调位置、速度和参考姿态，末段损失强调终点位置、零速度、平衡姿态与零角速度。没有在机动中惩罚角速度偏离零，也没有对全程角速度参考单独增加跟踪项。

最后至少 `hold_seconds` 的参考保持为终点位置和零速度、零角速度；训练逐时计算这一段的损失，不使用连续达标步数或通过/失败门槛。具体权重如下，所有时间窗在 rollout 前根据时钟预先计算。

| 损失部分 | 时间权重 | 默认主要权重 |
|---|---|---|
| 全程参考位置、速度 | 每步 $1/N_{\rm steps}$ | 位置 1，速度 0.25 |
| 中点附近位置、速度、姿态 | 以 $t/T=0.5$ 为中心、宽度 $0.06T$ 的归一化高斯窗 | 位置 4，速度 1，姿态 2 |
| 最后 1 秒的终点位置、速度、姿态、角速度 | 五次 smoothstep 逐渐增加，再归一化 | 位置 1，速度 0.5，姿态 0.25，角速度 0.05 |
| 最后一个状态 | 终点残差再计算一次 | 上述终点权重乘 2 |
| 电机命令变化、边界余量 | 每步 $1/N_{\rm steps}$ | 命令 0.02，边界 5 |

其中 $N_{\rm steps}$ 为 episode 的总控制步数，默认 500。

中点高斯窗跟踪的是**该时刻的移动参考**，仅在中心时刻它等于途经点。若把整个时间窗的位置目标都固定为途经点，会给减速停留留下错误激励。每架飞机的运动段可以提前结束，此时全程位置、速度损失继续跟踪终点；额外的静止姿态和角速度损失在整个 episode 的最后 `hold_seconds` 内平滑增加。

原边界导致的终止仍停止该行后续 Actor 和物理步进，但保留它的末状态，继续计算剩余时间的参考残差和终点残差。提前越界不能删除剩余任务损失，原有飞机行也不会从平均 objective 中移除。

损失中的位置与速度等量使用明确尺度无量纲化。姿态误差采用稳定的矩阵形式

$$
\ell_R=\frac14\|R-R_{\rm ref}\|_F^2
=1-\cos\theta_{\rm error}.
$$

向量残差使用 $\rho_\epsilon(\mathbf x)=\sqrt{\|\mathbf x\|^2+\epsilon^2}-\epsilon$，默认 $\epsilon=0.01$。默认全程位置尺度为 $B_i$，中点位置尺度为 $0.15B_i$，静止位置尺度为 $0.04B_i$；跟踪速度尺度为 $\max(\|\mathbf v_m\|,0.5\,\mathrm{m/s})$。静止速度、姿态和角速度尺度分别为 0.3 m/s、0.2 rad、0.5 rad/s。这些量用于归一化损失，均不是“通过阈值”。

### 只按更新数推进的课程

设已提交的优化器更新数为 $k$，预热更新数为 $k_0$，爬升更新数为 $K$。默认配置使用

$$
s=\operatorname{clip}\left(\frac{k-k_0}{K},0,1\right),\qquad
\lambda_{\rm req}(k)=\lambda_0+(\lambda_1-\lambda_0)(3s^2-2s^3).
$$

`start_difficulty=0.25`、`end_difficulty=1`、`warmup_updates=0`、`ramp_updates=1000`。`fixed_difficulty` 非空时可以固定请求幅度；`interpolation` 也可显式改为 `linear`。该计划不读取飞行指标，没有按成绩晋级或回退的状态。

验证使用固定物理样本、噪声及请求难度 1 的连续 objective，主动悬停练习比例固定为 0；每行都先请求机动，实际幅度仍经过同一飞机的执行器适配，不可用时的静止回退仍被保留并报告。按平均 objective 保存最佳 checkpoint。不要直接用不同课程阶段的训练 loss 判断能力提升，因为它们的任务幅度不同。

原来悬停任务中对每一步姿态变化的 `lambda_R * (1-cos(delta_R))` 惩罚不进入 agile 损失，因为机动本来就需要改变姿态。电机命令平滑、边界附近的连续惩罚和原有物理越界终止仍有各自作用；边界处理没有变成新的“任务达标”开关。

### 初始平衡和名义参考的区别

初始位置设为起点，速度、角速度及速度历史清零。电机状态和初始姿态根据已采样的恒定外力求静力平衡，末段损失使用同一个 `hover_rotation`。`previous_action` 仍是零占位，不暴露真实电机状态。

公开参考采用无外力几何。`hover_rotation` 只用于物理初始化及真值损失，不放进公开目标包，避免目标字段透露隐藏外力方向。脉冲时刻、作用力、作用点和未来噪声也不进入目标输入。

有恒定横向外力时，真实悬停初始姿态可能略有倾斜，而公开无外力参考在 $t=0$ 是水平姿态。这是软参考与受扰动系统之间的偏差，不能解释为两者在该时刻严格一致。当前损失没有全程姿态项，名义姿态主要在中点时间窗受到强调；持续扰动下，它仍可能与精确位置、速度跟踪发生权衡。任务损失不会构成精确跟踪或安全证明。

## 5. 反向传播与从悬停模型初始化

默认部署 Actor 的隐藏层仍为 64 维，读出仍为 `Linear(64,4) → tanh`。agile 将 GRU 输入从 16 维扩展为 36 维：原有 16 维反馈特征，加上 20 维目标特征。实际测量输入保持 22 维，目标包为额外的 23 维数据；包内信息在 Actor 中变换到机体系后形成 20 维增量特征。

原有位置特征改为相对于当前参考位置的误差，其余原始反馈定义保持，包括测得的重力方向。目标姿态不会替换重力方向。目标包包含当前参考位置、中点与终点位置、中点姿态和速度、归一化运动时间及剩余秒数。四路动作仍是原来的绝对归一化电机命令，没有增加额外 PID 或位置控制器。

`--init-checkpoint` 导入兼容的结构化悬停 `.pt` 时：

1. 保留原 GRU 输入权重的前 16 列。
2. 新增 20 维输入对应的整块 GRU 权重初始化为零。
3. 拷贝循环权重、GRU 偏置和原来的四电机读出层。
4. 新建 Adam、采样进度、课程进度和最佳验证记录；所有参数都继续参与训练。

导入器检查架构、隐藏维度、100 Hz 时间步、电机动作约定、张量形状、数值有限性和保存的模型摘要。没有这些元信息的裸 `state_dict`、不同动作语义或旧的其它 Actor 架构会被拒绝。

`--resume` 则用于同一 agile 任务的严格续训，需要保存的源码、任务配置、数值后端与其它绑定保持一致，并恢复 Adam、随机状态、采样和课程进度。它与 `--init-checkpoint` 互斥。不要通过修改旧 checkpoint 的摘要来绕过匹配检查。

训练保留全时域反向传播。主命令显式使用 `--time-decay 0`，因此不会在物理状态、GRU 记忆、速度历史或中间统计区间处主动截断远期梯度。

如另开实验采用 `--time-decay 0.2`，前向飞行与损失不变，但每个计划时间步的状态反向梯度会乘以 $e^{-0.2\Delta t}$。agile 在越界后的冻结状态上也继续相同的恒等状态扩展与反向衰减，使剩余损失的梯度不依赖同批其它飞机是否还在飞行；不会再次执行冻结行的 Actor 或物理步进。这是替代梯度，会减弱远期终点损失对较早动作的影响，不应称为原始完整梯度。已有的物理组梯度缩小裁剪也仍然会改变最终优化器收到的梯度。

## 6. 使用方法

需要 Python 3.11+、PyTorch 2.10 和 NumPy；测试还需要 pytest。在仓库根目录执行命令。下面的 `/absolute/path/hover.pt` 是占位符，运行前换成已有的兼容悬停权重文件；`runs/agile/...` 使用新的运行目录。

### 沿用现有 GPU 配置启动微调

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args \
    --task agile \
    --agile-config configs/agile_flight.json \
    --time-decay 0 \
    --init-checkpoint /absolute/path/hover.pt \
    --work-dir runs/agile/from_hover_seed7
```

`@configs/response_raptor_multi_airframe.args` 原来明确设置了 CUDA、3072 条训练场景、H500、Time Decay 1、50 次更新和 1800 秒预算。后面的 `--time-decay 0` 会覆盖旧值，其余计算资源和预算参数可在命令末尾显式覆盖。旧悬停损失的参数即使在该文件中出现，也不会成为 agile 的姿态变化惩罚。

运行预算和课程时间表是独立配置。默认课程需要 1000 次已提交更新才到完整请求幅度；沿用上面的 50 次预算时，只会从 0.25 增至约 0.255，不会完成课程。正式训练应根据资源明确设置 `--updates`、`--max-seconds` 和 JSON 中的 `ramp_updates`，并记录这些选择。短预算示例不能代替收敛验证。

若希望从随机 Actor 开始，只需去掉 `--init-checkpoint`。这与有悬停能力的模型初始化是两个不同实验，不能混为同一种训练起点。

### 小规模 CPU 命令示例

```bash
python tools/train_response_control.py @configs/response_raptor_multi_airframe.args \
    --task agile \
    --agile-config configs/agile_flight.json \
    --device cpu --threads 4 \
    --train-sampling random --scenarios 32 \
    --group-max-groups 4 --group-min-scenarios 32 \
    --eval-scenarios 16 \
    --time-decay 0 \
    --updates 10 --max-seconds 600 \
    --development-every 5 --checkpoint-every 5 \
    --init-checkpoint /absolute/path/hover.pt \
    --work-dir runs/agile/cpu_from_hover_seed7
```

`--scenarios` 是每个训练 bank 的场景数，因此这里共 $4\times32=128$ 条训练场景，验证共 $2\times16=32$ 条。这个例子改为随机物理采样，并覆盖原配置的分组参数，避免小批量与 `coverage128` 的固定配额要求冲突。它用于说明如何明确控制计算预算，不表示已经验证过十次更新的学习效果。

### 评估和续训

```bash
python tools/train_response_control.py --mode evaluate --task agile \
    --checkpoint /absolute/path/agile_checkpoint.pt \
    --eval-split validation --device cpu \
    --work-dir runs/agile/evaluation
```

评估读取 checkpoint 中保存的任务配置。`validation` 用于固定难度的模型比较；独立 `test` 分区用于最终检查，不参与训练或最佳模型选择。实际可选项以 `python tools/train_response_control.py --help` 为准。

严格续训时使用原运行目录、相同任务和数值配置，将初始化参数改为 `--resume /absolute/path/agile_checkpoint.pt`；需要延长预算时显式修改允许延长的更新数和总时间。

## 7. 配置、日志和验证边界

主要配置在 [`configs/agile_flight.json`](../configs/agile_flight.json)：

| 配置部分 | 作用 |
|---|---|
| `mission` | 请求倾角与推力比、几何、边界余量、末段保持时间、物理幅度适配 |
| `loss` | 连续位置、速度、姿态、角速度及命令/边界损失的尺度与权重 |
| `curriculum` | 按更新数执行的平滑难度时间表，不读取任务成功率 |
| `fixed_eval_difficulty` | 固定完整请求难度，保持验证目标可比 |

检查日志时同时看连续 objective 与任务生成报告中的 `actual_mid_tilt_degrees`、`actual_mid_speed_m_s`、`actual_difficulty`、`difficulty_capped_count`、`fallback_hover_count`、`infeasible_count` 和 `initial_rejection_counts`。它们区分请求难度、实际生成难度与执行器约束，不能只看配置中的“60°”。

源码与测试的入口包括：

- [`agile_mission.py`](../diffphys/agile_mission.py)：S9 参考、飞机保留、执行器检查和悬停初始化。
- [`agile_objective.py`](../diffphys/agile_objective.py)：连续任务目标、目标输入和训练指标。
- [`agile_curriculum.py`](../diffphys/agile_curriculum.py)：配置读取与按更新数推进的课程。
- [`agile_checkpoint.py`](../diffphys/agile_checkpoint.py)：悬停权重迁移与元信息校验。
- [`response_policy.py`](../diffphys/response_policy.py)：16→36 输入扩展，保持单 GRU 和四电机读出。

单元测试和小规模数值检查能验证公式、接口、梯度与保存恢复逻辑；它们不等于已经得到会特技飞行的权重，也不等于 CUDA 吞吐或实飞效果。训练结果应由实际运行日志中的连续损失和飞行轨迹单独说明。

### 本次实现的验证记录

2026-10-19，Python 3.12、PyTorch 2.10.0+cpu：全量测试收集 630 项，其中
603 项首次通过，22 项因无 CUDA 跳过；另 5 项旧断言需要兼容新增的可选轨迹字段
与 `goal_conditioning` 配置字段。更新这些断言后，相关模块及失败项均重新通过，
最终覆盖 **608 项通过、22 项 CUDA 跳过**。没有为通过测试更改原物理内核。

具体证据包括：连续损失与经真实 RK4 传播的有限差分一致；50 步统计分块与一次完整展开
的损失和全部 Actor 参数梯度一致；FP32/FP64 的原生分组 probe 与显式 VJP 一致；
可选 Time Decay 下，同一冻结样本的梯度不依赖其他飞机是否存活；16→36 权重迁移
保持多步悬停响应；两个短更新的连续训练与保存恢复在模型、Adam 和 RNG 上位级一致；
完整更新回放与固定验证重建可复现。这些测试中的 checkpoint 是结构兼容的测试权重，
没有导入用户尚未提供的已训练悬停模型，也没有进行长时间机动训练。

验证所用生产源码绑定 SHA256：
`4d7b6635a586649e3d29fc9ae826df290403dbc09e5520242b1c01d895fd87f8`。

```bash
python -m pytest -q tests --tb=short
python -m pytest -q tests/test_gpu_hotpath.py tests/test_linear_loss_architecture.py tests/test_core_contract.py
python -m pytest -q tests --lf --lfnf=none --tb=short
```

# 三点机动飞行：数学、训练和悬停权重迁移

本说明对应 `Agile-Flight-training` 分支的 `--task agile` 和 `agile-flight-training-v4` 配置。任务是：从点一带初速出发，按参考位姿和非零速度经过点二，再减速到点三。默认请求中间倾角为 **105°**，采用原边界内的较长路径，训练长度恢复为 **500 步、5 秒**。全部 500 步都可用于参考运动，不再预留末段保持时间；参考运动结束后持续发送终点目标。训练关闭恒定外力和外力脉冲，保留测量噪声与速度延迟。

每轮训练使用同一套三点生成规则，没有按更新次数或学习成绩改变目标幅度的课程。训练采用连续时间参考和连续损失，不设置到点切换、悬停达标或成功率门槛。名义参考仍从静止开始；真实初速作为初始跟踪误差，由同一套连续损失驱动恢复，详见第 4 节。

训练链路仍然是 **Actor → 四路电机命令 → 可微物理 → 任务损失 → Actor 参数更新**。没有增加 critic、教师网络、规划控制器或辅助模型。原有悬停任务保留，选择 `agile` 才启用这里的三点目标。初始化时导入原悬停 `.pt` 的权重，之后全部 Actor 参数继续训练；输出 `.pt` 保存完整的新模型，部署加载时不需要另一个基座 `.pt`。

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

机体推力轴沿 $\mathbf f_m$。采用可以区分锐角和钝角的双参数反正切，中点倾角满足

$$
\boxed{\theta_m=\operatorname{atan2}(K|h|,g-KH)\in[0,\pi].}
$$

因此速度、曲率、高度和姿态不能独立乱抽。中点通常还有非零角速度：飞机正在改变倾斜方向，不能仅因为它位于途经点就把角速度参考设为零。

完整姿态由推力轴和投影参考航向共同确定。令 $\mathbf f=\ddot{\mathbf p}+g\mathbf e_z$，代码使用

$$
\mathbf b_3=\frac{\mathbf f}{\|\mathbf f\|},\qquad
\mathbf x=\mathbf e-(\mathbf e\cdot\mathbf b_3)\mathbf b_3,\qquad
\mathbf b_1=\frac{\mathbf x}{\|\mathbf x\|},\qquad
\mathbf b_2=\mathbf b_3\times\mathbf b_1,\quad
R=[\mathbf b_1\;\mathbf b_2\;\mathbf b_3].
$$

这把参考航向 $\mathbf e$ 投影到推力轴的法平面，给出完整的旋转矩阵。起点、中点和终点的机体前轴与 $\mathbf e$ 同向，途中不再强制固定 Euler yaw。中点恰好 90° 时，$\mathbf f\cdot\mathbf e=0$，所以 $\mathbf x=\mathbf e$，没有旧公式的退化。

设 $c=\mathbf e\cdot\mathbf b_3$，解析导数为

$$
\dot{\mathbf x}=-\dot c\mathbf b_3-c\dot{\mathbf b}_3,\qquad
\ddot{\mathbf x}=-\ddot c\mathbf b_3-2\dot c\dot{\mathbf b}_3-c\ddot{\mathbf b}_3.
$$

单位化后得到 $R,\dot R,\ddot R$，再由 $[\boldsymbol\omega]_\times=R^T\dot R$ 求角速度及其解析导数。对当前曲线，$\mathbf f=A\mathbf e+C(h\mathbf n+H\mathbf e_z)+g\mathbf e_z$。与 $\mathbf e$ 垂直的推力分量满足

$$
\|\mathbf f_\perp\|^2=(Ch)^2+(CH+g)^2
\ge\frac{g^2h^2}{h^2+H^2}>0\quad(h\ne0).
$$

因此非零曲线且请求倾角在 $(0,180^\circ)$ 时，没有精确的航向平行奇点。筛选器仍排除推力模长过小、航向投影退化以及执行器超限；世界竖直推力为零或负值本身不再作为拒绝条件。旋翼总推力始终为 $m\|\mathbf f\|\ge0$，水平或向下的世界推力由机体朝向实现。

### 默认的 105° 和推力比 1.2

`tilt_degrees=105` 与 `thrust_ratio=1.2` 定义的是**请求中点姿态及该点总推力与重量的比值**。后者不是把飞机本身的推重比改为 1.2。默认构造

$$
\mathbf D=\eta\sin\theta\,\mathbf n+(1-\eta\cos\theta)\mathbf e_z,
$$

$$
L=0.70B_i,\qquad
\mathbf d=0.38B_i\frac{\mathbf D}{\|\mathbf D\|},\qquad
T_*^2=\frac{99225}{2048}\frac{0.38B_i}{g\|\mathbf D\|}.
$$

理想连续时间下，这给出

$$
\mathbf a_m=g[-\eta\sin\theta\,\mathbf n+(\eta\cos\theta-1)\mathbf e_z].
$$

105°、1.2 对应中点向下加速度约 $1.3106g$；恰好 90° 时竖直加速度为 $-g$。因此这里是带高度变化、可短暂超过水平倾角的机动，中点并不要求保持高度。默认中心为零时，端点控制点长度为 $0.70B_i$，中间控制点长度为 $0.76B_i$，均小于保留余量后的 $0.80B_i$；原边界本身没有扩大。

实现会把运动时长量化到偶数个 100 Hz 控制步，让中点恰好落在网格上。`configs/agile_flight.args` 默认 H500，共 5 秒，允许 $T=5$ 秒；`hold_seconds` 字段及相应的时间预留已经移除。通过物理检查的参考可以提前结束，之后持续给出终点位置、零速度和水平姿态。量化、时间上限和执行器限制后的实际中点速度、倾角会重新计算并写入报告。默认较长的空间几何保持为 $L=0.70B_i,\|\mathbf d\|=0.38B_i$，不会因恢复 H500 而主动缩短。

`tilt_degrees` 现在接受 $[0,180^\circ)$，包括 90°、105° 和更大倾角。精确 180° 不属于当前姿态构造的有效参数，接近 180° 的请求也可能因推力、转矩和电机响应被显著放慢，从而减小实际倾角。允许输入一个角度不等于每架机体都能完成该角度。`yaw_degrees` 固定路径的参考航向；`center_fraction` 改变相对于原边界的中心位置。该几何族允许短暂倒倾，但不描述完整翻滚一周，也不支持任意三个位置与任意三个姿态独立指定。

### Wolfram 核验与有界形状优化

[`agile_trajectory_optimization.wl`](agile_trajectory_optimization.wl) 已通过本次 Wolfram 工具实际执行。它独立求出了中点导数、端点的一至四阶导数、精确弧长，以及固定几何族中的弯曲量最优值。记 $b=\|\mathbf d\|$，路径长度为

$$
\mathcal S(L,b)=\sqrt{L^2+4b^2}
+\frac{L^2}{2b}\operatorname{asinh}\!\left(\frac{2b}{L}\right),
$$

直线极限为 $2L$。在不发生空间回退时，当前曲线长度约为 $1.639753B$，沿用 v3 的较长几何。相对于 v2 的 $L=0.45B,b=0.20B$、弧长约 $1.007376B$，几何本身延长约 62.8%。本次恢复 H500 保留这一跨度和弯曲量，v3 与 v4 的默认弧长相同。

Wolfram 对 S9 路径的平方 snap 积分给出

$$
J_{\rm snap}=\int_0^T\|\mathbf p^{(4)}(t)\|^2\,dt
=\frac{2419200(120285164b^2+8402199L^2)}{30808063T^7}.
$$

固定请求倾角及推力比时，前面的关系要求 $T_*^2\propto b$。因此固定较长半跨度 $L=0.70B$ 后，形状成本与 $(120285164b^2+8402199L^2)/b^{7/2}$ 成正比，其对 $b$ 的导数在 $L,b>0$ 时严格为负。Wolfram 的精确 `Minimize` 在选定的 $0.20B\le b\le0.38B$ 范围内得到 $b=0.38B$。这个上限使中间控制点保留 $0.04B$ 的额外余量，仍位于 $0.80B$ 的参考控制点界限之内。

这里优化的是明确给定几何族及余量内的平滑形状。下一节仍使用原机型、电机推力曲线和 RK4 电机响应逐候选检查，选择尽量短的可行运动时间。Wolfram 只用于离线推导与核验，训练、重放和模型部署无需调用 Wolfram。

## 3. 固定三点规则与同一架飞机的物理可行性

三点的位置、通过速度和中点姿态始终按前面的同一套公式生成，不根据训练进度增加跨度、倾角或速度。不同训练轮次的随机种子可以改变航向和左右转向，但不改变生成规则，也没有“难度值”或课程阶段。

原有采样包括不同尺寸、质量和电机能力的飞机。任务生成器对同一架已经抽好的飞机检查参考是否符合原有执行器能力，默认优先保留完整的 $L$ 和 $\mathbf d$，只调整运动时间。以理想时长量化后的 $T_0$ 为起点，`duration_search_factor=1.05` 生成递增候选；每个候选向上量化到 0.02 秒，去除重复时长，并包含允许的最大运动时长。采用第一个通过名义物理检查的完整几何候选。

这个时间搜索解决了旧几何缩放的一个问题：相同时间下同时缩短跨度和弯曲，会同时降低路径长度、速度和倾角。设加速度需要缩减为原来的 $s$ 倍，在相同归一化时间位置比较，有

| 调整方式 | 路径长度比例 | 加速度比例 | jerk 比例 | snap 比例 |
|---|---:|---:|---:|---:|
| $L\mapsto sL,\ \mathbf d\mapsto s\mathbf d$，时间不变 | $s$ | $s$ | $s$ | $s$ |
| $T\mapsto T/\sqrt{s}$，几何不变 | $1$ | $s$ | $s^{3/2}$ | $s^2$ |

因此延长时间能保留完整路径，并更明显地降低姿态变化和电机变化需求。实际姿态必须由新的加速度重新计算。令 $s=(T_*/T)^2$，完整几何的中点满足

$$
\theta_m(s)=\operatorname{atan2}\!\left(s\eta\sin\theta,1-s(1-\eta\cos\theta)\right),
$$

$$
\frac{d\theta_m}{ds}
=\frac{\eta\sin\theta}{(s\eta\sin\theta)^2+[1-s(1-\eta\cos\theta)]^2}>0.
$$

该式适用于 $0<\theta<180^\circ,\eta>0,s>0$，跨过 90° 仍连续且严格单调。因此在这个几何族内，较短的可行时长保留较大的实际中间倾角。Wolfram 使用的双参数写法是 `ArcTan[vertical, horizontal]`，参数顺序与 Python `atan2(horizontal, vertical)` 相反。可复现推导见 [`agile_trajectory_optimization.wl`](agile_trajectory_optimization.wl)。实现采用有限时间候选和 100 Hz 执行器审核；这不等于求得连续时间、全局最优的轨迹，也不证明全部可行时长构成单一区间。

若完整几何在全部候选上都不通过，才在最大运动时长下使用几何比例 $a$ 缩放

$$
L_a=aL,\qquad \mathbf d_a=a\mathbf d.
$$

速度和加速度都随这个比例线性变化；姿态从新的加速度重新求解，不能把角度直接当成 $a\times105^\circ$。报告中的 `reference_scale` 仍然只表示这次构造用了多大的几何比例，`duration_stretch_factor` 则单独记录时间变化；它们都不衡量学习难度。

空间回退仍按固定比例表检查，默认值为 `1, 0.8, 0.6, 0.4, 0.25, 0.125, 0.0625, 0.03125, 0`，采用第一个通过名义物理检查的比例。所有检查均在 rollout 前完成，不读取 Actor 的飞行表现，也不决定何时换点。所有已经抽好的飞机、测量噪声和物理组都保留。外力是否启用是单独保存的任务设置，不由参考可行性结果切换。

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

默认配置 `hover_fraction=0`，每架飞机都请求三点机动，不额外抽取单独的悬停练习。不能生成非零幅度时的静止回退仍单独报告，不能把这种退化记录当作完成了一条三点机动。

## 4. 无人机怎样知道点一、点二、点三的顺序

点一是起始位置：reset 直接把飞机放在点一，机体角速度为零；机动行保留已有的非零线速度，并为原零速度行补采初速。接下来，每个 0.01 秒控制步按照时钟读取 `mission.nominal_reference(time)`，生成当前参考。前面给出的曲线满足

$$
\mathbf p_{\rm ref}(0)=\mathbf p_0,\qquad
\mathbf p_{\rm ref}(T/2)=\mathbf p_m,\qquad
\mathbf p_{\rm ref}(T)=\mathbf p_f.
$$

由于 $S_9'(u)=630u^4(1-u)^4\ge0$，参考按点一、点二、点三的顺序推进；$t\ge T$ 后参考位置保持在点三，参考速度为零。点一已经作为起点满足，并没有另加一段从其它初始位置飞向点一的任务。

每步给 Actor 的公开目标包是 23 维：

| 目标信号 | 维度 | 用途 |
|---|---:|---|
| 当前参考位置 $\mathbf p_{\rm ref}(t)$ | 3 | 指明此刻应飞向的位置 |
| 点二位置、点三位置 | 3 + 3 | 提供未来途经点和终点的不同语义 |
| 点二姿态、点二通过速度 | 9 + 3 | 提供中途机动要求 |
| $\operatorname{clip}(t/T,0,1)$、本条轨迹剩余秒数 | 1 + 1 | 提供连续时间进度和剩余时间 |

当 $t\ge T$ 时，保持 23 维接口，三个位置槽全部填入 $\mathbf p_f$，原中点速度槽填零，原中点姿态槽填入公开的水平终点姿态；进度为 1，剩余时间非负截断。因此结束运动后只发送终点要求，不再继续发送已过时的中点倾角。这个变化由预定时钟触发，与飞机是否真的到达无关。预计算目标时钟先在 FP64 中乘以控制周期再转换到任务 dtype，与运动时长的量化一致，避免 FP32 舍入令终点命令晚一拍。

Actor 将这些目标与测量位置、姿态等一起转换为反馈特征，经 GRU 输出四路电机命令。顺序同时体现在目标字段的语义和随时间移动的参考中。它没有一个需要达标才能前进的离散任务编号。

**当前实现采用预定时间顺序。** 如果飞机在中点时刻没有飞到点二，参考仍然继续向点三推进；它不会等待补过点二。连续损失惩罚这个偏差，并通过可微物理把梯度传回较早的电机动作和网络参数。因此，这个实现表达了按顺序飞行的训练要求，但不能把“按时间推进参考”表述为已经保证实际逐点穿过。

给三点后，仍需用某种方式定义先后和运动时间；本实现使用从三点生成的平滑参考。当前生成器支持第 2 节说明的几何族，尚不是接受任意三个独立位置和姿态的通用命令接口。

### 连续损失怎样提供梯度

Actor 看到测量状态、公开目标和连续时间进度。输入中的 `phase` 是 $\operatorname{clip}(t/T,0,1)$，位置参考由它经过 $S_9$ 后生成；其阶段信息不由仿真真值、“已经通过”或“悬停达标”触发。全程损失跟踪参考位置、速度，中点时间窗额外强调位置、速度和参考姿态，最后 H 状态的单次终点残差强调终点位置、零速度、平衡姿态与零角速度。没有在机动中惩罚角速度偏离零，也没有对全程角速度参考单独增加跟踪项。

固定保持时间及对应的 `hold_weight`、逐步 `rest_*` 损失和指标均已移除。刹车仍有全程速度参考和最后状态的速度残差提供梯度；终点姿态及角速度仍由最后状态的连续残差引导，不设置任何悬停成功判断。具体权重如下，中点时间权重在 rollout 前根据时钟预先计算。

| 损失部分 | 时间权重 | 默认主要权重 |
|---|---|---|
| 全程参考位置、速度 | 每步 $1/N_{\rm steps}$ | 位置 1，速度 0.25 |
| 中点附近位置、速度、姿态 | 以 $t/T=0.5$ 为中心、宽度 $0.06T$ 的归一化高斯窗 | 位置 4，速度 1，姿态 2 |
| 最后 H 状态的终点位置、速度、姿态、角速度 | 只计算一次终点残差 | 位置 1，速度 0.5，姿态 0.25，角速度 0.05，再统一乘 2 |
| 电机命令变化、边界余量 | 每步 $1/N_{\rm steps}$ | 命令 0.02，边界 5 |

其中 $N_{\rm steps}$ 为 episode 的总控制步数，agile 参数文件默认 500。

中点高斯窗跟踪的是**该时刻的移动参考**，仅在中心时刻它等于途经点。若把整个时间窗的位置目标都固定为途经点，会给减速停留留下错误激励。每架飞机的运动段可以提前结束，此时普通位置、速度损失继续跟踪终点。终点后的参考可持续提供，但训练只对实际展开的 H500 计算梯度；持续发送目标本身不等于已经验证 5 秒以后的稳定悬停。

原边界导致的终止仍停止该行后续 Actor 和物理步进，但保留它的末状态，继续计算剩余时间的参考残差和终点残差。提前越界不能删除剩余任务损失，原有飞机行也不会从平均 objective 中移除。

损失中的位置与速度等量使用明确尺度无量纲化。姿态误差采用稳定的矩阵形式

$$
\ell_R=\frac14\|R-R_{\rm ref}\|_F^2
=1-\cos\theta_{\rm error}.
$$

向量残差使用 $\rho_\epsilon(\mathbf x)=\sqrt{\|\mathbf x\|^2+\epsilon^2}-\epsilon$，默认 $\epsilon=0.01$。默认全程位置尺度为 $B_i$，中点位置尺度为 $0.15B_i$，终点位置尺度为 $0.04B_i$；跟踪速度尺度为 $\max(\|\mathbf v_m\|,0.5\,\mathrm{m/s})$。终点速度、姿态和角速度尺度分别为 0.3 m/s、0.2 rad、0.5 rad/s，对应新的 `terminal_*_scale*` 字段。这些量用于归一化损失，均不是“通过阈值”。姿态损失在 90° 和 105° 的相对误差处有非零角度梯度；精确 180° 的相对误差是该矩阵损失的驻点，本次没有声称解决完整翻滚的全部姿态优化问题。

### 固定目标分布与验证

已提交的更新数只参与训练目标的随机种子构造和重放记录，不控制目标幅度。任务配置没有 `curriculum`、`start_difficulty`、`end_difficulty` 或 `fixed_eval_difficulty` 字段；旧课程配置会被版本和字段检查拒绝，避免继续使用已经移除的行为。

验证使用固定物理样本、噪声、目标种子及同一套三点构造规则，每行都先请求机动。每架飞机仍经过相同的执行器检查，不可用时的静止回退保留并报告。按固定验证集的平均连续 objective 保存最佳 checkpoint；训练目标的随机变化不构成课程晋级。

原来悬停任务中对每一步姿态变化的 `lambda_R * (1-cos(delta_R))` 惩罚不进入 agile 损失，因为机动本来就需要改变姿态。电机命令平滑、边界附近的连续惩罚和原有物理越界终止仍有各自作用；边界处理没有变成新的“任务达标”开关。

### 无外力、非零初速与名义参考的区别

agile 默认关闭外加恒定力和所有外力脉冲，包括由脉冲作用点产生的转矩。重力、旋翼自身推力和转矩、电机响应、原始机型分布、测量高斯噪声及 10–30 ms 速度延迟继续保留。`--disable-external-forces` 可显式选择这一设置；`--enable-external-forces` 可在另一个新运行中启用原外力协议。旧 hover 任务默认仍启用外力。`--disable-disturbances` 是同时关闭传感器噪声、延迟和外力的总开关，不能拿它替代这里的仅关闭外力。

对机动行，初始位置设为点一，保留原样本已有的非零三维世界线速度，其各轴独立服从 $U[-2.5,2.5]$ m/s。原 guidance 抽样等零速度行则由独立、固定种子的随机流补采同范围初速；不替换飞机，不重抽物理参数或测量噪声。报告 `initial_zero_velocity_replaced_count` 记录补采行数。默认每行都请求机动，因此不会额外保留原 guidance 的静止练习。

上述范围按轴定义，不是速度模长范围，也不强制速度朝向点二。三个历史速度槽全部设为这一个真实初速，避免延迟观测在起步时混入虚构的零速度历史。角速度清零，姿态和电机设为静力平衡；无外力时就是水平姿态和抵消重力的电机状态。`previous_action` 仍为零占位，不暴露真实电机值。

名义 S9 参考的 $\mathbf v_{\rm ref}(0)=0$ 保持不变，因此带初速的真实状态与参考在起点存在有意保留的速度误差；连续位置、速度损失负责引导恢复并继续跟踪三个点。没有增加第零个目标、初速消除阶段或“恢复成功”判定。初始速度不会在后续每步反复注入。只有显式请求的单独悬停行从零速开始；无法生成机动时回退的是静止参考，该机动行的真实初速仍保留，并单独报告。

这也意味着初速可能朝向边界外侧，较长路径的起点有时会更接近边界；参考曲线的凸包证明不保证这样的真实初始状态都能被及时刹住。代码不通过过滤或重采样隐藏这些飞机，仍由原越界处理和连续损失反映偏差。选择显式外力实验时，初始和终点平衡姿态可以补偿恒定力，但公开参考始终不含隐藏力；`hover_rotation` 只用于物理初始化与真值损失，未来脉冲、作用点和噪声也不进入目标输入。

## 5. 反向传播与从悬停模型初始化

默认部署 Actor 的隐藏层仍为 64 维，读出仍为 `Linear(64,4) → tanh`。agile 将 GRU 输入从 16 维扩展为 36 维：原有 16 维反馈特征，加上 20 维目标特征。实际测量输入保持 22 维，目标包为额外的 23 维数据；包内信息在 Actor 中变换到机体系后形成 20 维增量特征。

原有位置特征改为相对于当前参考位置的误差，其余原始反馈定义保持，包括测得的重力方向。目标姿态不会替换重力方向。目标包包含当前参考位置、中点与终点位置、中点姿态和速度、归一化运动时间及剩余秒数。四路动作仍是原来的绝对归一化电机命令，没有增加额外 PID 或位置控制器。

`--init-checkpoint` 导入兼容的结构化悬停 `.pt` 时：

1. 保留原 GRU 输入权重的前 16 列。
2. 新增 20 维输入对应的整块 GRU 权重初始化为零。
3. 拷贝循环权重、GRU 偏置和原来的四电机读出层。
4. 新建 Adam、采样进度和最佳验证记录；所有参数都继续参与训练，包括导入的原始权重。

训练后保存的是这一个 Actor 的**全部参数**和架构配置，不是只保存新增输入列的差量，也没有第二个冻结的悬停网络参与推理。原 `.pt` 仅在初始化时读取一次；新的 `.pt` 加载时不会再次访问原文件。来源摘要用于记录初始化来源，不是运行时依赖。

在当前仓库代码下，仅提供训练输出文件即可加载完整 Actor：

```python
import torch
from diffphys.response_training import load_policy_checkpoint

policy, saved = load_policy_checkpoint(
    "/absolute/path/agile_checkpoint.pt", device="cpu", dtype=torch.float32
)
policy.eval()
```

运行推理仍需每步的测量状态、GRU 记忆以及上节定义的目标包。目标包是飞行命令数据，不是另一个模型文件。loss 在训练时更新权重；部署时 Actor 根据观测和参考执行前向计算，输出电机命令。

导入器检查架构、隐藏维度、100 Hz 时间步、电机动作约定、张量形状、数值有限性和保存的模型摘要。没有这些元信息的裸 `state_dict`、不同动作语义或旧的其它 Actor 架构会被拒绝。

`--resume` 则用于同一 agile 任务的严格续训，需要保存的源码、任务配置、数值后端与其它绑定保持一致，并恢复 Adam、随机状态、采样和已提交更新数。它与 `--init-checkpoint` 互斥。不要通过修改旧 checkpoint 的摘要来绕过匹配检查。

训练保留全时域反向传播。agile 参数文件使用 `--time-decay 0`，因此不会在物理状态、GRU 记忆、速度历史或中间统计区间处主动截断远期梯度。这里沿用 `CPU_compile_accelerated` 的执行链：CUDA 单步 `RaptorSimulator.step` 通过 `torch.compile(fullgraph=True, dynamic=True)` 和 strict Inductor 编译，AOTAutograd 生成该物理步的前向与反向；CPU 保留原生物理参考。GRU、完整展开、任务损失、分组梯度探针和 Adam 仍由原训练链执行。H500 的完整 BPTT 由各步 autograd 图相连，不等于把整条 500 步训练编译成一个图。

如另开实验采用 `--time-decay 0.2`，前向飞行与损失不变，但每个计划时间步的状态反向梯度会乘以 $e^{-0.2\Delta t}$。agile 在越界后的冻结状态上也继续相同的恒等状态扩展与反向衰减，使剩余损失的梯度不依赖同批其它飞机是否还在飞行；不会再次执行冻结行的 Actor 或物理步进。这是替代梯度，会减弱远期终点损失对较早动作的影响，不应称为原始完整梯度。已有的物理组梯度缩小裁剪也仍然会改变最终优化器收到的梯度。

## 6. 使用方法

需要 Python 3.11+、PyTorch 2.10 和 NumPy；测试还需要 pytest。在仓库根目录执行命令。下面的 `/absolute/path/hover.pt` 是占位符，运行前换成已有的兼容悬停权重文件；`runs/agile/...` 使用新的运行目录。

### 使用加速链路的 agile 配置启动微调

```bash
python tools/train_response_control.py @configs/agile_flight.args \
    --init-checkpoint /absolute/path/hover.pt \
    --work-dir runs/agile/from_hover_seed7
```

`@configs/agile_flight.args` 复用原加速配置的 CUDA、CPU 预取和物理分组设置，选择 agile、H500、Time Decay 0 和无外力训练。训练规模仍是 3072 条场景，示例预算仍为 50 次更新或 1800 秒；这些资源和预算参数可在命令末尾显式覆盖。原 `configs/response_raptor_multi_airframe.args` 继续作为 hover 配置，保留 H500 和 Time Decay 1。旧悬停损失的参数即使在继承配置中出现，也不会成为 agile 的姿态变化惩罚。

外力设置与任务配置一起保存在 checkpoint 绑定中；评估、严格续训和更新重放沿用保存的设置。修改源码、任务配置或外力协议需要新运行，并通过 `--init-checkpoint` 显式迁移兼容权重；不能用新的 CLI 标志偷偷改变旧 checkpoint 的评估环境。

`--updates` 和 `--max-seconds` 只控制运行预算。第 1 次与后续更新都使用同一三点生成规则，没有需要等待完成的课程时间表。正式训练应按资源明确设置预算并记录选择；短预算示例不能代替收敛验证。

若希望从随机 Actor 开始，只需去掉 `--init-checkpoint`。这与有悬停能力的模型初始化是两个不同实验，不能混为同一种训练起点。

### 小规模 CPU 命令示例

```bash
python tools/train_response_control.py @configs/agile_flight.args \
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

评估读取 checkpoint 中保存的任务配置，只需这个输出 `.pt`。`validation` 用固定样本和目标做模型比较；独立 `test` 分区用于最终检查，不参与训练或最佳模型选择。实际可选项以 `python tools/train_response_control.py --help` 为准。

严格续训时使用原运行目录、相同任务和数值配置，将初始化参数改为 `--resume /absolute/path/agile_checkpoint.pt`；需要延长预算时显式修改允许延长的更新数和总时间。

## 7. 配置、日志和验证边界

主要配置在 [`configs/agile_flight.json`](../configs/agile_flight.json)：

| 配置部分 | 作用 |
|---|---|
| `version` | 当前为 `agile-flight-training-v4`，与旧版本的严格续训区分 |
| `mission` | 请求倾角与推力比、固定几何规则、边界余量、`duration_search_factor` 和物理可行性检查；无保持时间字段 |
| `loss` | 连续位置、速度、姿态、角速度及命令/边界损失的尺度与权重 |

检查日志时同时看连续 objective 与任务生成报告。`actual_mid_tilt_degrees`、`actual_mid_speed_m_s` 和 `actual_path_length_m` 记录真正生成的机动要求；`initial_speed_m_s` 记录真实初速，`initial_zero_velocity_replaced_count` 记录原零速度机动行的补采；`time_stretched_count` 与 `duration_stretch_factor` 记录时间扩展，`reference_scale` 与 `reference_capped_count` 记录剩余的空间回退。`fallback_hover_count`、`infeasible_count`、`initial_rejection_counts` 和 `actuator_screen_evaluations` 用于解释退化参考及规划工作量。不能只看配置中的“105°”，也不能把这些参考指标当成 Actor 已实现的飞行结果。

源码与测试的入口包括：

- [`agile_mission.py`](../diffphys/agile_mission.py)：S9 参考、保留路径的时间搜索、执行器检查和带初速初始化。
- [`agile_objective.py`](../diffphys/agile_objective.py)：连续任务目标、目标输入和训练指标。
- [`agile_config.py`](../diffphys/agile_config.py)：固定三点任务配置的严格读取与序列化。
- [`agile_checkpoint.py`](../diffphys/agile_checkpoint.py)：悬停权重迁移与元信息校验。
- [`response_policy.py`](../diffphys/response_policy.py)：16→36 输入扩展，保持单 GRU 和四电机读出。

单元测试和小规模数值检查能验证公式、接口、梯度与保存恢复逻辑；它们不等于已经得到会特技飞行的权重，也不等于 CUDA 吞吐或实飞效果。训练结果应由实际运行日志中的连续损失和飞行轨迹单独说明。

### 本次 v4 版本的验证

2026-10-19 集成验证使用 PyTorch 2.10.0+cpu。针对本次改动的回归共 **166 项通过**，覆盖全部 agile 测试、物理约定和生产管线。生产管线第一次在受限环境中因 PyTorch 进程间 socket 权限失败；取得所需进程通信权限后，原测试 11 项全部通过，没有改动采样器实现。

```bash
python -m pytest -q tests/test_agile*.py tests/test_core_contract.py \
    tests/test_production_pipeline.py
```

验证范围包括精确 90°、105° 和 135° 下的 SO(3) 姿态、解析角速度与角加速度、
参考占满 500 步、末点在任意后续时刻持续有效、FP32 时钟边界的一致性、
固定保持窗口完全移除、原边界、无外力但保留测量噪声/延迟、
初速和历史速度一致性、连续损失梯度、物理展开与统计分块、16→36 权重迁移、
固定任务采样、保存恢复和完整更新重放。独立检查点测试在 FP32/FP64 下验证
原始参数与新增目标列都能更新，并在移除原悬停文件后，由新 Python 进程仅加载
输出 `.pt` 复现保存的权重、电机输出和 GRU 状态。固定采样回归检查同一随机种子
下更新数不会改变生成规则；目标参数不会随训练表现推进。

另外，对同一批原始 32 架飞机（物理种子 314、任务种子 91、FP64、CPU 单线程）比较了父版本 `69f00b947ebb94889efa85dca9a5978afeeb36e1` 的生成器与当前生成器。两者使用相同原始机型、位置边界和 clean 名义参考夹具；父版本使用默认 H800/75°，v4 使用默认 H500/105°。clean 只用于这一参考对照，实际 agile 训练仍保留传感器噪声。这里没有运行 Actor 飞行或正式训练。完整数值、随机种子、逐机倾角和调用方式见 [`agile_trajectory_audit.json`](agile_trajectory_audit.json)。

| 经过执行器审核后的参考指标 | 父版本 v3 | 当前 v4 |
|---|---:|---:|
| 平均中点倾角 | 55.0396° | 62.8254° |
| 中点倾角范围 | 30.1714–74.8766° | 27.0904–104.7342° |
| 实际中点倾角超过 90° | 0 / 32 | 5 / 32 |
| 平均路径长度 | 5.7058 m | 5.7058 m |
| 平均中点速度 | 4.3716 m/s | 4.4305 m/s |
| 平均运动时长 | 2.685 s | 2.67125 s |
| 路径发生空间缩放的样本 | 0 / 32 | 0 / 32 |
| 通过名义执行器审核的样本 | 32 / 32 | 32 / 32 |

这表明配置上限和真实参考都已越过 90°，同时仍有机体因原执行器能力而降低实际倾角。单独用 `DisturbanceConfig(force_enabled=False)` 的真实 agile 噪声设置核对 reset：全部机动初速非零，非零原始速度保持逐位一致，三个历史速度槽与真实初速一致，外力和脉冲为零，传感器噪声存在且其采样带与速度延迟保持逐位一致。这些有限样本结果不能替代整个机型分布上的效果验证。

本次没有改动 `env_raptor.py`、`compiled_physics.py`、`strict_inductor.py`、`response_execution.py`、`scene_prefetch.py`、`response_grad_probe.py` 或原 hover 参数文件；它们仍与加速基线 `bf5b08d95e4cd99602f15e5b747ffd6012eed944` 一致。规划仍在物理 bank 预取交付后于 CPU 上执行，日志记录 `agile_mission_and_upload_seconds`。本次环境没有 CUDA，未重新执行 CUDA 前向/反向编译或吞吐探针。

测试使用结构兼容的合成检查点；用户实际已训练的悬停 `.pt` 尚未导入。本次没有进行
长时间机动训练，CPU 验证也不代表 CUDA 性能或已学会飞行。

# AFT：三点机动训练

本分支用于三点位姿机动：从点1出发，穿越点2区域，最后在点3停稳。
推荐入口只有 `tools/train_response_control.py @configs/agile_flight.args`。
原生四旋翼RK4、测量噪声/延迟、完整H500反传、物理分组裁剪、CUDA编译、CPU预取与审计保留。

本地最近一轮使用原平方点位 loss，见[点位 loss 回退](docs/point_loss_rollback.md)，已在第6426次尝试发生反传溢出后停止。
反传范围缩放与候选更新验收已按用户要求撤除；撤除记录见 `reports/root_cause_20261028/withdrawal.json`。
学习率为0.001、时间损失权重为3，仍保留原 loss、硬任务、Actor、物理和分组裁剪。当前继续深入调查根因，尚未恢复长训练。
撤回后的[根因调查](reports/root_cause_20261028/report.md)保存硬区域/软罚分空档、首次事件计分跳变、初态边界证书及反馈回路对照；这些诊断未写成新的生产修复。
随后按用户要求将穿点球半径统一为 `max(2L, 0.1 m)`，见[半径下限说明](docs/pass_region_floor.md)。这项任务几何修改没有重新引入上述防护补丁，尚未恢复训练。
固定EVAL每50次更新一次；调整此间隔时使用精确续训，保留当前Actor、Adam和更新计数。GUI独立刷新日志。
围栏±50L，起终点在±40L立方体表面，中点在±30L内部；目标机体平面正反等价，
允许绕机体z轴自由转动；在首次穿点交点，速度沿中点到终点方向且≥1.5 m/s，并垂直实际机体z轴。
同一个Actor、原训练规模和STOP条件保留。

此前从零训练使用[随机初态与中点配置](docs/random_via_training.md)：
`@configs/random_via_flight.args`。起终点不变，中点位于各轴边界内缩7L的立方体，
目标倾角61°–119°为原1°容差留余量，穿点速率下限1.5 m/s。
仍使用原完整姿态loss和同一个Actor。持续训练包装器按用户要求运行至手动停止。

## 当前任务

- 点1：指定位置、水平姿态，实际出生速度和角速度为零。
- 点2：默认105°姿态，区域内速率下限2 m/s。速度高于下限不受速度项惩罚，不要求恒定速度向量。
- 区域半径为 `max(2L, 0.1 m)`；靠近中点中心的容差另为0.1 m。
- 测量位置首次入球后仍以点2为目标，首次出球后锁定点3；切换不等待姿态或速度达标，不重置GRU。
- 点3：固定终点、水平姿态、零速度与停稳要求。末端义务始终计分。
- Actor使用69维任务/观测特征，经过原生 `GRUCell(69,64) -> Linear(64,4) -> tanh` 输出四路电机命令。

这是软损失训练，任务判定和源配置见 [三点任务说明](docs/pass_stop.md) 与 `configs/pass_stop.json`。
没有轨迹规划器、教师网络、Critic或第二套生产积分器。

## 运行

原冻结环境为Python3.11、PyTorch2.10.0+cu126、NumPy；可视化另安装 `requirements-gui.txt`。
精确续训使用与检查点绑定一致的源代码、PyTorch版本和配置。

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True python tools/train_response_control.py @configs/agile_flight.args --work-dir runs/pass_stop/new_run
```

默认3072个TRAIN场景、256个固定EVAL场景、100 Hz、H500；`--scenarios 768`指四个银行各768场景。
`coverage128`每个物理单元24场景。默认运行到50次更新或1800秒，需更长训练时显式指定预算。
初始化权重使用 `--init-checkpoint PATH`；同源精确续训使用 `--resume PATH`。二者不能混用。

以下冻结检查点的精确续训命令需在冻结标签 `frozen-aft-20261026-u1456` 的源码中执行；
当前新增随机任务支持已改变源码绑定，当前任务从随机权重开始。
在冻结版本中，从已保留的1456次检查点继续到累计2000次：

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True python tools/train_response_control.py @configs/agile_flight.args --resume frozen/20261026-u1456/latest.pt --work-dir runs/pass_stop/from_frozen --updates 2000 --max-seconds 3600
```

持续训练使用保留的运行包装器；它仅取消预算，仍调用同一训练函数：

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True python runs/pass_stop/local_trial_20261026_095338/continuous_train_response_control.py @configs/agile_flight.args --resume frozen/20261026-u1456/latest.pt --work-dir runs/pass_stop/from_frozen --development-every 10 --checkpoint-every 10
```

Ctrl+C或SIGTERM在当前更新完成后保存Actor、Adam、RNG并退出。数值错误仍保存故障证据并报错。
源码、任务、物理、噪声、采样、优化器配置与检查点绑定；不能修改哈希绕过不一致。

固定EVAL使用检查点保存的任务、场景与时域：

```bash
python tools/train_response_control.py @configs/agile_flight.args --mode evaluate --checkpoint frozen/20261026-u1456/best.pt --work-dir runs/pass_stop/frozen_eval
```

## 编译和梯度

CUDA编译原RK4单步及其AOT反向、任务特征、任务损失及其反向、分组参数梯度局部贡献。
原生CUDA融合GRU、Adam和外层钩子/检查保留。CPU执行是数值参考。
CPU子进程只提前准备一批不可变场景/噪声；飞行前向和反传在GPU进行。
较大连续反传保存值暂存CPU固定内存，以适配本机8 GB显卡，不截断计算图。

Time Decay默认0，保留完整H500 BPTT；显式正值只改变反向传播，不能再称为原始loss的精确梯度。
物理组固定128组，各组仅缩小超限梯度，再合并进行Adam；生产默认只有一次原生反传图遍历。
旧VJP及16维基础Actor/悬停损失作为权重导入和独立数值参照保留，没有额外推荐训练配置。

详见 [编译与预取](docs/compiled_prefetch.md)、[分组梯度](docs/group_gradient_probe.md)、[物理覆盖](docs/physics_coverage.md)。

## 监视和回放

```bash
python tools/monitor_response_training.py --run-dir runs/pass_stop/from_frozen
python tools/short_eval_replay.py --run-dir runs/pass_stop/from_frozen --device cuda --poll-seconds 2
python tools/play_response_short.py --replay runs/pass_stop/from_frozen/short_eval/playlist.json
```

监视器只读日志；播放器只读已导出的NPZ，不导入Torch或占用CUDA推理。
独立导出工具负责用已保存的检查点生成固定EVAL回放，播放窗口显示对应Actor更新号。
详见 [GUI说明](docs/training_gui.md)。

## 文件职责

| 文件 | 责任 |
| --- | --- |
| `pass_stop.py` / `pass_stop_physics.py` | 三点任务、Goal69特征、区域几何、速度下限、时间/末端loss |
| `env_raptor.py` / `response_noise.py` | 原RK4、电机、刚体、硬边界及测量带 |
| `response_policy.py` / `response_task.py` | 原生Actor基础特征、状态/轨迹、通用展开和数值参照 |
| `response_adjoints.py` / `response_groups.py` / `response_grad_probe.py` | 完整反传、物理组、一次原生反传探针 |
| `response_training.py` / `response_audit.py` | Adam事务、校验、检查点与可回放审计证据 |
| `compiled_physics.py` / `compiled_execution.py` / `strict_inductor.py` | CUDA编译边界与精度约定 |
| `response_sampling.py` / `scene_prefetch.py` / `tensor_storage.py` | 物理覆盖、CPU一批预取和反传暂存 |
| `tools/profile_response_training.py` | 对实际任务进行只读前向/反传剖析 |
| `tools/replay_response_update.py` | 同源码/后端的完整更新与Adam回放 |
| `tools/audit_pose_velocity.py` | 原动力学/位姿损失的定向数值检查 |

## 验证与冻结版本

```bash
python -m pytest -q tests
```

保留原物理、测量、位姿/速率、状态机、完整梯度、分组裁剪、CPU预取、Adam回滚、精确续训及GUI只读测试。
CPU测试不代表CUDA吞吐；本轮简化不启动正式训练或长程评估。
清理范围和数值兼容依据见 [清理记录](docs/aft_cleanup.md)。

冻结标签 `frozen-aft-20261026-u1456` 保留原始完整版本；`frozen/20261026-u1456/`里的清单与历史证据描述该标签。
当前简化分支删除了未使用的独立工具，因此不要用冻结标签的全仓文件清单检查当前分支文件是否齐全。
清理阶段保持训练源码绑定不变；随后新增随机初态/中点任务改变了源码绑定。
旧检查点精确续训应切回冻结标签。当前随机任务使用独立目录、随机权重和全新Adam。

第三方来源与MIT声明保留在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [物理来源说明](docs/raptor_reference.md)。

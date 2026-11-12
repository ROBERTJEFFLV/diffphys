# 速度与机体z轴正交

用户明确要求速度垂直机体z轴。此前将速度垂直机臂平面，等价于平行机体z轴，方向理解相反；旧运行已保存退出，源码和检查点保留。

当前使用 `configs/gradient_repaired_flight.args`，目录为 `runs/pass_stop/gradient_repaired_time3_lr001_seed7/`。梯度设计实验修复了低速减速捷径，重新设计了PASS/STOP连续罚分；硬任务条件不变。配置、实验对照、验证及限制见 [梯度修复记录](gradient_design_20261027.md)。

前一轮使用 `configs/velocity_z_time3_lr001_flight.args`，学习率为0.001、时间权重为3、EVAL间隔50次更新。目录为 `runs/pass_stop/velocity_z_orthogonal_time3_lr001_seed7/`。该轮从前一运行第2003次更新建立显式学习率分支，保留Actor、Adam、RNG和更新计数；随后在第3062次更新保存退出用于梯度实验。旧检查点和 `source_before_gradient_design.tar.gz` 保留。

前一阶段使用 `configs/velocity_z_time3_flight.args`，仅将时间权重从10改为3。目录 `runs/pass_stop/velocity_z_orthogonal_time3_seed7/` 只导入正确任务此前第116次更新的Actor权重，重新建立Adam和运行计数。32场景H64的float64生产轨迹检查得到时间项原始梯度范数比恰好0.3，其余损失分量逐项相同；证据见 `reports/velocity_z_time3_20261027/`。裁剪和Adam后的参数步长不保证乘0.3。

随后按用户要求把常规EVAL间隔从10次更新改为50次，配置显式设置 `--development-every 50`。此项是运行控制，不改变源码/目标/优化器绑定；当前运行用 `--resume` 精确续训，保留Actor、Adam、RNG与更新计数。GUI日志刷新频率独立于EVAL间隔，原GUI与导出器继续连接同一运行目录。切换检查点与命令记在运行目录的 `eval_interval_change.json`。

令中点为P、终点为S，h=(S−P)/‖S−P‖，b=R(q)e₃为实际机体z轴。首次穿点的条件为：速率≥1.5 m/s，速度方向与+h相差不超过1°，速度与b的夹角偏离90°不超过1°。实际速度通过Actor和原动力学生成，更高速率没有任务上限惩罚。

目标机体z轴选为竖直向上方向在h的正交补中的归一化投影，使其与h正交；恰好竖直的h改选世界x轴，避免零向量。这个公开目标选取不更改四旋翼推力方向。机体平面正反等价、绕z轴旋转自由；保留原平面姿态容差1°。Actor仍自行决定路径、动作及正反姿态。

有效速度段使用单位速度与实际机体z轴的正交误差，仍直接检查实际b和实际v。低速区增加缺失方向罚，固定姿态和方向时，降低速度不能再降低角度罚分。具体公式见梯度修复记录；低于1.5 m/s仍不能通过，角度容差仍为1°。

用于定义穿点时刻的导航门面经过P、法向为h，保留首次正向交点验收。2026-10-28按用户要求将有效球半径改为 `max(2L, 0.1 m)`；旧回放保留当时记录的半径。它与机体平面是两个不同平面：导航门面垂直速度，机体z轴垂直速度。首次测量出球仍永久切换到STOP，不等姿态/速度达标，不重置GRU。中点不新增角速度约束；STOP仍使用原完整姿态和停稳检查。

围栏仍为每轴±50L；起终点在[-40L,40L]立方体表面，中点在[-30L,30L]内部。原随机初始速度、姿态、角速度、电机、三个速度历史槽、成员身份、128物理组、噪声和延迟均保留。Actor仍为GRUCell(69,64)→Linear(64,4)，保留完整BPTT、原分组反传、裁剪和Adam；当前时间权重为3，其余损失权重和STOP阈值不变。

```bash
python tools/train_response_control.py @configs/gradient_repaired_flight.args --init-checkpoint reports/gradient_design_20261027/runs/anchored_log_w2/latest.pt
```

持续训练使用运行目录内的原生产函数包装器，取消更新数/时间上限，直到用户喊停：

```bash
python runs/pass_stop/gradient_repaired_time3_lr001_seed7/continuous_train_response_control.py @configs/gradient_repaired_flight.args --init-checkpoint reports/gradient_design_20261027/runs/anchored_log_w2/latest.pt
```

修正几何关系时的第一轮目录为 `runs/pass_stop/velocity_z_orthogonal_time10_from_scratch_seed7/`：该轮从随机Actor、空Adam和更新0开始，完成116次更新后按用户要求保存退出，再迁移到当前权重3配置。任务版本为 `pass-region-min-speed-v3-unoriented-plane-velocity-z-orthogonal-first-crossing-v3`，不修改旧检查点绑定。原规模3072 TRAIN、128组×24、256 EVAL、H500、100 Hz保持。

GUI金色长线表示目标机体z轴，绿色箭头表示朝终点的目标速度；两者正交。金色方框为机体平面示意，大小不改变真实有效球。目标夹角与当前策略的角度误差分开显示。旧回放根据原记录保留旧任务显示语义。

几何修正的完整CPU回归651通过、25个CUDA检查跳过；另行64场景H8的CUDA原生/编译前向与梯度对照通过。完整3072 TRAIN和256 EVAL在float32与float64下均通过目标z轴正交检查，保留原初态和物理成员。该轮生产H500反传和Adam更新已实际运行，证据保存在 `reports/velocity_z_task_20261027/`；时间权重3的变更另保存在上述新目录。几何检查、数值/梯度正确性和学会任务是不同证据；短检查不代表策略已经学会。

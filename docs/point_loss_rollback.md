# 点位loss回退

按用户要求，撤回此前为抑制绕过途径点而引入的对数罚分和PASS/STOP耦合。pass_stop.py逐字恢复为 reports/gradient_design_20261027/sources/angular_fix/pass_stop.py 的实现：PASS位置、姿态、最低速率、速度朝向、机体z正交以及STOP七项约束，均恢复原来的加权平方超差罚分；未穿点时的共同见证选择也恢复原来的平方和。

保留plane_tasks.py的低速方向评分修正，降低速度不能减轻错误方向/正交罚分。Actor、三个公开目标、点位规划、任务硬阈值、首次事件、2L区域切换、完整BPTT、分组裁剪、时间权重3、余弦调度和锁页缓存释放保持。

Actor始终接收三个点的数据。“当前目标”在首次测量轨迹离开中点2L球后永久切到STOP，不以姿态或速率达标作为切换前提。终点可见性与loss计算是不同机制。旧版STOP罚分本来始终生效，因此这次回退不等于证明绕过途径点已解决。

当前入口为 configs/quadratic_point_flight.args，运行目录 runs/pass_stop/quadratic_points_hover_best_cosine_seed7。继续仅导入根目录best.pt的78950次悬停权重，新建Adam并从更新0开始；CosineAnnealingLR从1e-3经10000次更新降到1e-5后保持。更换loss的旧检查点不能冒充同源精确续训；此前运行和检查点全部保留。

复核证据见 reports/point_loss_rollback_20261027：rollback.patch、rollback_scope.json、专项CPU检查、完整CPU回归和原生/编译CUDA损失及梯度对照。它们验证实现与数值，不证明训练成功率。

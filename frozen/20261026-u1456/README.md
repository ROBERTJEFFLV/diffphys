# 冻结版本：2026-10-26，update 1456

本目录保存用户确认保留的一版：3072场景、100 Hz、H500、Goal69/GRU64三点机动，CUDA分块前向/反向编译，CPU场景预取和固定内存暂存。

- `latest.pt`：停止后的第1456次更新，包含完整Actor、Adam、RNG和配置，可精确续训。
- `best.pt`：按固定开发EVAL目标值选出的第1450次更新。
- 冻结标签：`frozen-aft-20261026-u1456`。修改受绑定源码后应使用权重初始化或原冻结源码，不能改写检查点哈希。
- 训练源码SHA256：`60227cbe739189e57d4b43fc925b2d8774235c7601c9d1394935ad791ad0f5bf`。
- 第1450次固定开发评估：目标值13.6514，完整任务成功78/256（30.46875%）。这批EVAL被重复用于选best，不能当作独立测试成绩或部署保证。

## 恢复

原运行环境为Python3.11、PyTorch **2.10.0+cu126**；具体NumPy、驱动等版本见`environment.json`。精确续训要求检查点绑定的PyTorch版本匹配。可视化另安装仓库的`requirements-gui.txt`。
从仓库根目录执行，训练目录选择新目录：

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True python tools/train_response_control.py @configs/agile_flight.args   --resume frozen/20261026-u1456/latest.pt   --work-dir runs/pass_stop/resumed_frozen   --updates 2000 --max-seconds 3600
```

上例累计训练到update2000或运行一小时。需要一直训练直到手动停止时，使用随冻结版本保留的原运行包装器：

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True python runs/pass_stop/local_trial_20261026_095338/continuous_train_response_control.py @configs/agile_flight.args   --resume frozen/20261026-u1456/latest.pt   --work-dir runs/pass_stop/resumed_frozen   --development-every 10 --checkpoint-every 10
```

Ctrl+C或SIGTERM会在当前更新完成后保存并退出；数值错误仍按原训练器保存故障状态并报错。
固定GPU步长、噪声、任务、物理分组和损失全部沿用冻结配置；不要将CPU执行等同于CUDA性能验证。

## 验证与备份

`source_manifest.json`记录发布源码的逐文件摘要，`manifest.json`记录训练配置和模型摘要，`SHA256SUMS`覆盖本目录内容。
本次新跑CPU回归608通过、25跳过；CUDA专项8项通过的记录来自2026-10-23，之后该源码实际训练至1456。
保留的连续运行包装器与监督进程源码位于原`runs/pass_stop/local_trial_20261026_095338/`路径，以保持相对导入不变。
`captured_launch.json`为原机路径的审计记录；其他机器请使用上面的可移植命令。
完整训练日志、所有历史评估检查点、回放、审计证据、原输入论文/补丁包及完整Git备份另保留在原机的冻结备份目录中。

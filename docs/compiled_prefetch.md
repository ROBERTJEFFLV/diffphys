# CUDA 编译与 CPU 场景预取

生产入口仍为 `tools/train_response_control.py @configs/agile_flight.args`。
本机建议在命令前设置 `TORCHINDUCTOR_COMPILE_THREADS=1`，限制首次编译的 CPU 内存并发。
CUDA 自动采用分块编译；CPU 使用原生数值参考。已移除旧训练参数 `--rotation-backend`，不会再只编译旋转矩阵。

## 计算分工

| 部分 | 执行方式 |
| --- | --- |
| 原 RK4 单步物理与该步反向 | strict Inductor + AOTAutograd |
| 测量特征、Goal69 特征、阶段张量、Linear 读出 | strict Inductor；可微部分生成反向 |
| 三点任务完整张量损失与反向 | strict Inductor + AOTAutograd |
| 单次原生反传中的 GRU/Linear 分组参数局部贡献 | strict Inductor；保留原 FP64 分组累计 |
| 原生 CUDA GRUCell 前向、反向 | PyTorch 原生融合内核 |
| Adam、分组裁剪检查、有效行压缩、钩子、审计与存盘 | 原生调度 |
| 下一批飞机参数、初始状态、不可变测量/扰动带 | 一个持久 spawn CPU 子进程提前准备 |

PyTorch 2.10 的 `aten._thnn_fused_gru_cell` 不支持本项目完整图捕获所需的 FakeTensor。
实际编译错误完整保存在 `reports/native_gru_compile_limit.json`；已完成的 CUDA 验证见 `reports/compiled_execution_cuda_final.log`。
保留原生 GRU，不使用私有调度器或改写门控公式代替前向/状态反传。
整数事件前缀累计 `cumsum` 在图内调用原生 CUDA 算子，避开 PyTorch 2.10 的 H500 无梯度评估 SplitScan 代码生成错误。
失败回归与修复后检查分别保存在 `reports/h500_eval_scan_red.log` 和 `reports/h500_eval_scan_green.log`。
Dynamo 的有限编译变体上限设为64，覆盖 FP32/FP64、前向/梯度模式、紧凑行布局和单行终止阶段；不关闭 guards。
原默认8变体失败证据为 `reports/compiled_recompile_limit_red.log`。
其余编译错误仍直接报错；没有静默退回原生计算。

外层 H500 循环保留有效行压缩、任务状态与一次原生分组反传钩子。
完整物理计算图、GRU 记忆及三个历史速度槽跨全部步保留。
编译不会把硬切换变成光滑事件，也不改变 Time Decay。配置默认仍为 0。

## CPU 预取与显存

CPU 一次只排队一批，按已提交的 update 索引确定种子。
主进程取走当前批后提交下一批，GPU 更新 Actor 时 CPU 生成下一批；不会用过期权重提前飞出整条轨迹。
不会修改当前批或共享噪声带。工作进程限定线程数，禁止初始化 CUDA，退出时等待并回收。
源代码、后端、存储策略与采样约定均绑定 checkpoint；旧源码 checkpoint 只能作权重初始化，不能伪装精确续训。

3072 × H500 在本机 8 GB GPU 上直接保存全部中间量会 OOM，证据为 `reports/compiled_prefetch_h500_3072.log`。
CUDA 训练现在将不少于 65536 元素的连续反传保存值暂存至 CPU 固定内存，反向需要时返回原 GPU/dtype/stride。
非连续视图保留 GPU，以满足编译反向的布局约定；不截断计算图，也不更换积分器。
此方式会增加主机内存和 PCIe 传输，不能把小批次的加速比直接套到 3072 场景。

## 实测与边界

同一 hover 权重扩展到 Goal69、相同场景与噪声，128 场景 × 100 步：

| 执行 | 前向+反传中位数 | GPU 峰值分配 |
| --- | ---: | ---: |
| 原生参考 | 0.9153 s | 130.1 MB |
| 编译后 | 0.3166 s | 129.0 MB |

两组都启用相同 CPU 暂存策略。该小批次约加速 2.89 倍；本次编译及预热另耗约 32.46 秒。
这是冻结权重的计算吞吐，不含 Adam、审计存盘或收敛结论。
原始输出为 `reports/compiled_prefetch_native_final.json` 和 `reports/compiled_prefetch_compiled_final.json`。早期基线保留作审计。
当前代码已完成 3072 场景 × H500 的 3 次真实训练更新，工作目录为 `runs/pass_stop/compiled_prefetch_smoke_20261023_195918`。
后两次完整更新为 3.861 和 2.825 秒，GPU 峰值分配约 3.50 GB；首轮约89.7秒包含新图编译，不能算作稳定吞吐。
CPU 每批生成约0.4秒，后续取批等待约0.000011秒，上传与交付约0.035秒；子进程从未初始化CUDA，退出后已回收。
每轮向CPU暂存约6.7 GB中间值，主训练进程最大RSS约11.86 GB，另有CPU采样进程及其他应用的内存。
第三次完整CUDA更新已独立回放，梯度、更新后模型、Adam状态与RNG逐位相同；证据为 `reports/compiled_prefetch_cuda_replay.json`。
旧的冻结权重吞吐检查也保留在 `reports/compiled_prefetch_h500_3072_offload.json`，但最终代码应以上述真实更新记录为准。

CPU 损失重构与修改前快照在 FP32/FP64 的逐场景成本、指标、分项和全部参数梯度逐位一致，见 `reports/compilation_refactor_identity.json`。
CUDA 编译允许浮点舍入差异，由前向、梯度及 Adam 对照检查，不声称与原生逐位一致。

完整验证结果及 3072 场景冒烟运行位置在 `reports/compiled_prefetch_current.json` 指向的报告目录记录。
CPU完整回归608通过、25跳过；CUDA专项8通过，另9项最终CPU契约/预取/剖析检查通过。
本次训练在3次更新预算处正常结束；末次固定EVAL的任务成功率仍为0。该工作验证执行与梯度，不代表新三点任务已训练成功。

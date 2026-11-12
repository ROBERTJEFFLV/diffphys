# 悬停BEST权重初始化与余弦学习率

当前配置为 `configs/quadratic_point_flight.args`，点位loss已按用户要求回退原二次评分，详见 docs/point_loss_rollback.md。仅导入仓库根目录 `best.pt` 的悬停 Actor 权重（实际训练更新78950）；更新计数、Adam、余弦调度和随机采样从0开始。使用现有权重导入机制，将原GRU的16个输入列复制到当前69列，其余53列置0，循环层和读出层权重保留。TRAIN 3072（128组×24）、固定 EVAL 256、H500、每50次更新 EVAL、时间权重3。

```bash
python tools/train_response_control.py @configs/quadratic_point_flight.args --updates 10000 --max-seconds 86400
```

CLI 的更新数与时间预算仍是显式上限。本机持续运行沿用运行目录中的 continuous_train_response_control.py 包装入口，按用户授权持续到喊停；10000 是学习率衰减长度，不是停训条件。

这是权重初始化的新训练，不是原悬停任务的精确续训。原始best.pt、旧运行及其检查点保留。先前随机初始化运行在切换前协作停止并保存，其配置仍为 configs/gradient_repaired_cosine_flight.args。

悬停BEST初始化后的首次尝试也因内存压力被systemd-oomd终止，已提交7次更新并保留原运行。当前运行目录为 runs/pass_stop/quadratic_points_hover_best_cosine_seed7；此前锁页缓存修正后的运行已在第98次保存停止。训练在每次更新的计算图释放后，归还不再使用的CPU锁页缓存；不改变反传、采样或参数更新。日志tensor_storage字段记录释放前后的pinned_reserved_before_bytes和pinned_reserved_after_bytes。这里使用本机PyTorch 2.10提供的torch._C._host_emptyCache接口，升级PyTorch时须复核其可用性；torch.cuda.empty_cache只释放GPU缓存，不能代替它。

本机由独立用户服务 aft-hover-best-cosine.service 管理训练，避免与终端/GUI共用同一内存管理分组（cgroup）。可用 `systemctl --user status aft-hover-best-cosine.service` 查看，或用 `systemctl --user stop aft-hover-best-cosine.service` 协作停止并保存。

锁页缓存修改后：CPU回归689通过、26个CUDA检查跳过；另有5项本机CUDA存储检查通过。相同BEST、种子、场景和H500下，前三次3072场景更新的Actor、Adam、RNG和进入Adam的梯度均与修改前逐位一致。已完成的早期更新每次归还约8GB锁页缓存；短程检查不代表长期内存或策略学习稳定性已经获证。

使用原生 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-5)`，Adam 初始 LR 为 `1e-3`。调度单位为成功提交的 Adam 更新，不是物理时间步或 EVAL 次数。每次 Adam 与审计写入成功后才调用 scheduler.step()，达到 T_max 后不再推进调度器，防止继续训练时余弦回升。

| 已完成更新 | 下一次更新 LR |
|---|---:|
| 0 | 0.001 |
| 5000 | 0.000505 |
| 10000及以后 | 0.00001 |

history.jsonl 的 learning_rate 是该次 Adam 使用的值，next_learning_rate 是下一次使用的值。完整检查点保存调度状态；同源同配置恢复不重置曲线。更改调度参数会被精确恢复检查拒绝。审计胶囊保留当前 Adam 步实际使用的 LR，记录后才推进下一次 LR，因此原逐更新回放语义不变。

旧运行于2026-10-27 21:15:31被Linux全局OOM杀死（PID3061890），与先前500–600次的有限数值策略退化不同。内核证据保存在 reports/cosine_lr_20261027/previous_run_kernel_oom.log。100批原规模采样/上传检查没有复现共享内存持续增长，不能据此认定整个训练的内存问题已修复，也不能认为余弦衰减已经解决长期策略退化。

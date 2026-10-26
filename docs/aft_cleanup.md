# AFT主线清理，2026-10-26

从`700944c42d98d263a5234001c3635c4f3cdd43ed`清理未激活且没有主线依赖的独立工作流。
冻结标签`frozen-aft-20261026-u1456`、`frozen/`中的权重/清单、运行包装器和现有本地运行资料均保留。

## 删除范围

| 文件 | 删除原因 |
| --- | --- |
| `response_long_eval.py` | 独立长时悬停模拟/移动目标评估，不是当前三点任务 |
| `tools/evaluate_response_long.py` | 上述评估的单独入口 |
| `tools/play_response_long.py` | 上述旧协议的导出/播放器 |
| `tests/test_long_eval.py` | 只测试已删除的长时评估 |
| `tests/test_long_eval_player.py` | 只测试已删除的长时播放器 |
| `configs/response_long_eval.args` | 已删除评估的配置 |
| `docs/long_hover_eval.md` | 已删除评估的使用文档 |
| `configs/response_raptor_multi_airframe.args` | AFT依赖的重复悬停预设，生效值已合入`configs/agile_flight.args` |
| `tools/benchmark_group_backward.py` | 只支持基础16维Actor的独立计时脚本；保留实际AFT任务剖析器和独立VJP回归 |
| `tools/audit_pass_stop_time.py` | 旧的独立合成时间项审计；保留真实任务时间项/梯度回归 |

10个文件合计1594行。排除测试、冻结资料和生成运行后的Python运行源码，由7338行降至6248行，净减少1090行，约14.9%。
这是源码大小变化，不是吞吐或收敛速度测量。

## 保留范围与必要的接线

当前主线仍是三点Goal69 Actor、原RK4/电机/测量、H500物理反传、分组裁剪、CUDA编译、CPU预取/暂存、Adam、审计、固定EVAL和short回放。
16维基础Actor/悬停损失、原力与脉冲核、显式VJP保留，分别用于权重导入、共享特征和独立数值/梯度验证；没有额外推荐训练预设。
原数值契约、校准、来源/许可证及历史性能证据不删除。

`configs/agile_flight.args`现在自包含，所有生效值与原冻结版相同。
监视器只发现/启动`short-eval-v1`回放，拒绝已移除的长时协议；播放器保持CPU只读。
保留的实际任务剖析器补上原生69维检查点读取，并验证读取不修改权重或执行Adam更新。
README、项目指引和GUI文档改为当前AFT入口，历史协议/性能文档明确标注适用范围。

## 验证证据

机器可读结果见[aft_cleanup_validation.json](aft_cleanup_validation.json)。本地完整原始日志位于`reports/aft_cleanup_20261026_123239/`及`reports/aft_cleanup_full_cpu.log`。

- 完整剩余CPU回归：`python -m pytest -q tests`，589通过、25跳过，94.48秒。安装Pygame，SDL使用dummy显示，GUI冒烟未跳过；CUDA专项在该CPU环境中跳过。
- 21个`response_training.SOURCE_FILES`逐文件与清理前核对，零改动；总源摘要仍为`60227cbe739189e57d4b43fc925b2d8774235c7601c9d1394935ad791ad0f5bf`。
- 23个冻结/运行/保护契约资料逐文件摘要不变。
- PyTorch2.10.0+cu126下，从冻结`latest.pt`执行累计更新上限1456的真实CLI恢复。没有执行新训练更新或飞行评估；保存后仍为1456，Actor、Adam、RNG、绑定及参数名逐位一致。

零更新检查命令：

```bash
python tools/train_response_control.py @configs/agile_flight.args \
  --resume frozen/20261026-u1456/latest.pt \
  --work-dir reports/aft_cleanup_20261026_123239/zero_update_resume \
  --updates 1456 --max-seconds 120 --development-every 10 --checkpoint-every 10
```

本轮没有正式训练、长程评估或CUDA吞吐复测，不给出新的学习表现或部署安全结论。
需要恢复已删除工具时，使用冻结标签或Git历史；无需修改检查点摘要。

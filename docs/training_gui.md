# AFT监视器与固定EVAL回放

监视器读`history.jsonl`、`evaluation.jsonl`和进程/退出记录；播放器读已导出的NPZ。
两者都不导入Torch、载入Actor、生成新评估或修改训练状态。

## 打开窗口

在装有`requirements-gui.txt`的环境运行：

```bash
python tools/monitor_response_training.py --run-dir runs/pass_stop/from_frozen
```

窗口区分最后完成的TRAIN、最后固定EVAL、运行/正常停止/中断/失败。
只有完整JSONL行才进入曲线。读取量和保留点数有上限，日志轮转/回滚不会保留虚假的未来更新。
没有进程信息时明确显示未知状态，不用日志年龄推断训练崩溃。
关闭窗口不会停止训练；训练收到Ctrl+C或SIGTERM后会保存当前更新并退出。

## 生成与打开回放

导出是独立工具的显式行为，在训练机器的CUDA环境运行：

```bash
python tools/short_eval_replay.py --run-dir runs/pass_stop/from_frozen --device cuda --poll-seconds 2
```

它用已提交检查点重新执行原固定EVAL，原子发布`short_eval/playlist.json`及对应NPZ。
每次导出用短生命GPU子进程，导出之间释放CUDA上下文。
回放的Actor更新号与源摘要写入元数据；有对应EVAL记录时核对指标。
训练未保存的中间Actor不会冒充回放里的Actor，固定EVAL也不会冒充连续TRAIN飞行。

只导出一次可以增加`--once`。只读播放：

```bash
python tools/play_response_short.py --replay runs/pass_stop/from_frozen/short_eval/playlist.json
```

也可以在监视器的已保存回放列表中点击播放，或使用`--replay PATH`指定播放列表。
自动发现只检查指定目录及一层子目录中的`playlist.json`/`short_eval/playlist.json`，不递归扫描所有训练文件。
没有已导出NPZ时显示缺失，监视器不会自动训练、推理或导出。

## 三点任务显示

AFT回放记录轨迹、姿态、电机命令、真实有效帧、三个固定点、当前目标/阶段、区域半径和最低速率。
播放器显示入球前/球内/终点阶段，以及实际速率相对下限的不足量；出球不会清除此前的区域评分。
区域半径`2L`与靠近中点中心的0.1 m容差分别显示。
播放到真实越界/终止帧后不把冻结填充帧当作继续飞行。
旧的同协议short回放仍可读取；未记录的遥测明确标为未记录。

主线只接受`short-eval-v1`播放列表。已移除另一套长时移动目标/表面力悬停评估及其播放协议。
若需要历史工作流，请切到冻结标签`frozen-aft-20261026-u1456`；当前分支不会把旧长时协议误交给AFT播放器。

## 只读与复核

渲染可在CPU机器上进行。复制播放列表及它指向的NPZ即可，不需要训练检查点、原悬停基座或CUDA。
播放器验证NPZ摘要、数组形状、三点命令字段和最低速率元数据，损坏或不完整的数据会明确报错。
`--screenshot PATH`和`--max-frames N`用于有限GUI检查，不会运行新的飞行。
测试覆盖无Torch导入、无训练目录修改、元数据扫描上限、失败帧边界和场景切换。

本工具显示仿真记录，不是部署控制器或安全监督器。

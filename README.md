# AGCD

AGCD 联合优化体素机器人的形态和控制器。控制器恢复自
[ModularEvoGym 的 transformer_ppo](https://github.com/Yuxing-Wang-THU/ModularEvoGym/tree/fbc9b7d86f65001efd097ba28ad7f42542658a9e/examples/transformer_ppo)，
对应提交 `fbc9b7d86f65001efd097ba28ad7f42542658a9e`。

## 环境与运行

使用已按 ModularEvoGym 配置、安装好仿真器的 Conda 环境：

```bash
conda activate modularevogym
python run_AGCD.py
```

在本仓库根目录执行。控制器源码已包含在本仓库中，运行不需要导入旁边的
ModularEvoGym checkout，但需要环境中安装的 ModularEvoGym 版 `evogym`。

入口保留现有批量实验行为：依次运行 `Walker-v0`、`DownStepper-v0`，每个任务运行一次，
种群 20、并发工作进程 20、成熟度上限 25，并启用注意力引导及 MFP。
两个任务均使用 `get_par` 提供的 500 次单个体 PPO 更新预算和 50000 次总更新预算。
这些计数指 PPO 更新次数，每次更新采集 128 个仿真步。
入口原有的参数覆盖和任务循环保持不变；命令行中的 `--env` 等参数不会覆盖该循环。

结果写入 `result/AGCD/<任务名>/0/`，包括结构、控制器、`config.json`、
`out.txt`、`table.csv` 和 `all_agents.json`。入口会跳过已经存在的实验目录；
重跑前请先移动需要保留的旧目录。

## 控制器与继承

- 使用原版单一观测投影、可学习的一维位置编码、Transformer，以及拼接原始局部观测的解码器。
- 保留原版价值聚合、动作分布及掩码处理；观测不需要额外的 `body` 字段。
- `PPOAgent.uni_act(..., return_attention=True)` 返回 Actor 第一层注意力，供 AGCD 的形态演化使用。
- AGCD 的 PPO 设置保持为 rollout 128、4 个 epoch、4 个 mini-batch，保留单环境 buffer 修复。
- 后代始终深拷贝最相似父代的最佳控制器。最佳权重在评估成绩改善时保存，与对应的注意力一致。
  `--inherit` 已删除，配置不再包含 `inherit`。

此前二维位置编码或分离 x/y 编码器训练出的检查点与恢复后的网络结构不同，
不做自动迁移。请使用新的实验目录重新训练。

## 验证

基础检查覆盖两个默认环境、不同机器人尺寸、真实观测、掩码、序列尺寸调整和有限梯度：

```bash
python tests/test_controller.py -v
```

与原仓库进行数值一致性检查时，额外提供本地参考仓库路径：

```bash
python tests/test_controller.py --reference-root /home/dyc/桌面/work/ModularEvoGym -v
```

此项使用相同权重逐项比较动作均值、标准差、价值、动作对数概率、熵和注意力。
参考仓库仅供此检查使用。

缩小预算完成真实训练和演化全流程：

```bash
python tests/smoke_agcd.py
```

默认验证两个任务，每个任务 4 个初始个体、2 个工作进程、两代演化和精英补训。
仅在验证脚本中将 rollout 缩至 16、epoch 缩至 1、mini-batch 数缩至 2；
正式训练默认值不变。验证会检查实际参数更新、交叉和变异继承的权重及存储独立性、
最佳权重与检查点一致性、注意力、输出文件，并加载模型及归一化统计完成真实评估。

输出目录默认自动创建在系统临时目录，路径在开始时打印。
也可使用 `--output-root result/my-smoke-run` 指定一个尚未存在的验证目录，
或用 `--env Walker-v0` 只验证一个任务。缩小预算验证用于确认流程可运行，不用于评估收敛性能。

## Git

`.gitignore` 忽略缓存、本地编辑器配置、虚拟环境、构建文件、实验输出、日志和本地凭据。
已跟踪的 Python 缓存和 `.vscode` 设置已移出索引，本地文件保留。
第三方源码及测试资源继续跟踪，Git 身份、远程仓库和分支跟踪配置沿用现有设置。

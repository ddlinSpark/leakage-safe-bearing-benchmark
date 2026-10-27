# 复现说明

## 环境

- Python 3.10+
- PyTorch 2.0+，需匹配本机 CUDA/CPU 环境
- 参考开发环境：CUDA 13.0、RTX 5080
- 完整依赖见 `requirements.txt`

先检查环境：

```bash
python -m pip install -e .
spindle-check-gpu
```

Rebuilding the DOCX additionally requires Pandoc. Set `PANDOC_EXE` if the
executable is not on `PATH`.

## 推荐流程

所有命令在仓库根目录执行。先通过 `--help` 确认当前版本参数：

```bash
spindle-build-raw --help
spindle-build-windows --help
spindle-train-backbone --help
spindle-train-calibrator --help
spindle-self-train --help
spindle-postprocess --help
spindle-predict --help
```

主流程为：

1. 构建并清洗数据集。
2. 构建滑窗索引。
3. 训练原始 CWT/残差注意力主干。
4. 训练校准器并生成净化伪标签。
5. 迭代 self-training；R3 增加 GRU 时序分支。
6. 对最终输出执行后处理和测试评价。
7. 按需组合图表或运行推理。

## 复现约束

- 原始数据和训练产物不随仓库分发，必须从归档或数据拥有方获取。
- 模型权重只与本项目版本和训练配置匹配。
- `self_train_relabel.py` 的源主干、校准器和输出目录必须显式对应，不能复用
  其他轮次的权重。
- 脚本保留历史超参数默认值以便追踪实验，但复现论文结果时应以对应
  `run_summary.json` 中的配置为准。
- 由于按工位时间顺序划分数据，不应改用随机划分后直接比较论文指标。

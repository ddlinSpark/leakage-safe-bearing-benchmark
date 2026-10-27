# Spindle CNN — Leakage-Safe Bearing Fault Diagnosis

本仓库保存主轴系统轴承故障诊断的实验代码，并明确区分历史工业数据试验与
公开数据确认实验。历史十类结果依赖规则生成的标签，不能作为真实故障准确率；
论文修订版只把公开数据上的真实标签结果作为主要证据。

## 公共基准结果

修订实验使用 Paderborn 轴承数据中的振动与两路电流信号。测试同时留出未见
轴承和未见工况，窗口之间不重叠。

| 方法 | Accuracy | Macro-F1 | Macro-F1 标准差 |
| --- | ---: | ---: | ---: |
| Raw 1D-CNN | 0.409 | 0.308 | 0.072 |
| CWT 2D-CNN | 0.400 | 0.291 | 0.088 |
| CWT-GRU | 0.391 | 0.284 | 0.106 |
| CWT-GRU + calibrated self-training | 0.350 | 0.240 | 0.103 |

主比较为四折工况留出 × 三个随机种子。完整结果见
[docs/results/paderborn](docs/results/paderborn/README.md)。自训练没有提高
严格外推条件下的宏 F1；温度缩放改善了概率校准，但不改变预测类别。

原稿中的 96% 私有数据结果保留为历史试验记录，不再作为论文主结果。

## 代码结构

```text
src/spindle_cnn/
├── data/           # 数据清洗、STFT、滑窗索引和时频图预览
├── final/          # 最终主干、校准、self-training 和推理
├── public_benchmark/ # CWRU/Paderborn 泄漏安全基准与论文图表
├── baselines/      # GADF/Swin 对照实验
├── reporting/      # 指标汇总和图表组合
├── document_tools/ # Word 文档生成工具
└── environment/    # GPU 环境检查
```

实验说明见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)，数据说明见
[docs/DATASET.md](docs/DATASET.md)，复现步骤见
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。公开基准实验与泄漏安全
协议见 [docs/PUBLIC_BENCHMARK.md](docs/PUBLIC_BENCHMARK.md)。

## 安装

建议使用 Python 3.10+。请先根据本机 CUDA 版本安装匹配的 PyTorch，再安装本项目：

```bash
python -m pip install -e .
```

也可以使用 `python -m pip install -r requirements.txt`。

## 快速验证

```bash
spindle-check-gpu
spindle-build-raw --help
spindle-train-backbone --help
spindle-predict --help
spindle-cwru-benchmark --help
spindle-pu-cache --help
spindle-public-summary --help
```

推理示例：

```bash
spindle-predict --npy window.npy --op PPOP40_1
spindle-predict --csv batch.csv --batch --device cpu
```

模型目录需包含 `run_summary.json`、`best_model.pt` 和校准器文件。默认路径为
`outputs/self_train_v3_gru` 与 `outputs/self_train_v3_gru_calibrated`。

修订论文位于 `docs/paper/主轴故障诊断方法_SCI修订版.docx`，该二进制文档
默认不进入 Git。重建命令见
[docs/PUBLIC_BENCHMARK.md](docs/PUBLIC_BENCHMARK.md)。

## 数据与产物

原始数据、处理后的数据集、模型权重和图像产物默认位于仓库根目录的
`data/`、`dataset_final/`、`outputs/` 和 `figures/`。这些体积较大的文件已
通过 `.gitignore` 排除。

如需将运行目录放在其他位置，可设置 `SPINDLE_CNN_ROOT`：

```powershell
$env:SPINDLE_CNN_ROOT = "D:\spindle_runtime"
```

本地归档、比赛投稿副本和压缩包位于 `artifacts/local/`，不会进入 Git。

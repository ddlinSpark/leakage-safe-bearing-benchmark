# Spindle CNN — 主轴故障弱监督十分类

本仓库保存主轴多变量时序故障诊断的可运行实验代码。最终方案使用多小波
CWT、残差注意力、双向 GRU、self-training 和条件校准器完成十分类。

## 最终结果

| 版本 | 方案 | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| R3 GRU | CWT + GRU + Self-training ×2 + Calibrator | 0.9605 | 0.9585 |

最终测试结果来自本地实验结果目录，不随代码仓库提交。

## 代码结构

```text
src/spindle_cnn/
├── data/           # 数据清洗、STFT、滑窗索引和时频图预览
├── final/          # 最终主干、校准、self-training 和推理
├── baselines/      # GADF/Swin 对照实验
├── reporting/      # 指标汇总和图表组合
├── document_tools/ # Word 文档生成工具
└── environment/    # GPU 环境检查
```

实验说明见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)，数据说明见
[docs/DATASET.md](docs/DATASET.md)，复现步骤见
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

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
```

推理示例：

```bash
spindle-predict --npy window.npy --op PPOP40_1
spindle-predict --csv batch.csv --batch --device cpu
```

模型目录需包含 `run_summary.json`、`best_model.pt` 和校准器文件。默认路径为
`outputs/self_train_v3_gru` 与 `outputs/self_train_v3_gru_calibrated`。

## 数据与产物

原始数据、处理后的数据集、模型权重和图像产物默认位于仓库根目录的
`data/`、`dataset_final/`、`outputs/` 和 `figures/`。这些体积较大的文件已
通过 `.gitignore` 排除。

如需将运行目录放在其他位置，可设置 `SPINDLE_CNN_ROOT`：

```powershell
$env:SPINDLE_CNN_ROOT = "D:\spindle_runtime"
```

本地归档、比赛投稿副本和压缩包位于 `artifacts/local/`，不会进入 Git。

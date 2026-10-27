# 实验代码说明

## 最终链路

最终实验位于 `src/spindle_cnn/final/`：

| 模块 | 职责 |
| --- | --- |
| `train_cwt_residual_attention_compare.py` | 主干训练、CWT 构建、评价和缓存 |
| `train_logit_calibrator.py` | Affine、condition-aware、confusion-aware 校准器 |
| `self_train_relabel.py` | TTA 推理、伪标签重建和主干重训 |
| `postprocess_cwt_calibration.py` | 温度缩放、向量缩放、isotonic 和局部偏置 |
| `ensemble_calibrators.py` | R2/R3 校准器集成分析 |
| `predict.py` | 单样本、CSV 批量和 NumPy 推理 |
| `models/logit_calibrator.py` | 校准器网络定义 |

最终主干的输入是 9 个传感器序列，以及由 `morl/mexh/gaus4` 多小波融合得到的
CWT 图。`ResidualAttentionGRUClassifier` 将 CWT 残差注意力特征和双向 GRU
时序特征拼接，再经过分类头输出十分类 logits。

## 对照实验

`src/spindle_cnn/baselines/` 保留两套 GADF/Swin 路线，用于结果对比：

- `train_gadf_swin_gam_classifier.py`
- `train_gafd_gru_swin_cbam_classifier.py`

这两套脚本不是最终 R3 的依赖。运行它们需要与最终实验相同的数据表和滑窗索引。

## 数据与分析工具

- `data/build_dataset_from_raw.py`：从原始 `sbxx.csv`、`dyxx.csv` 生成完整数据集。
- `data/clean_data1_for_modeling.py`：清洗单表数据。
- `data/build_final_dataset_with_stft.py`：增加 STFT 概览与数据集元信息。
- `data/build_window_sample_index.py`：建立训练、验证、测试滑窗索引。
- `data/plot_six_time_frequency_overviews.py`：绘制六类时序变换总览。
- `reporting/`：汇总训练曲线、混淆矩阵和版本指标。

已移除的 `rebuild_windows.py` 只是一次性路径补丁，其逻辑已由
`build_dataset_from_raw.py` 和 `build_window_sample_index.py` 覆盖。

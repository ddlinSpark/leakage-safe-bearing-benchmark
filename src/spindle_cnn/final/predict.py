"""主轴故障推理脚本。用法：

spindle-predict --csv sample.csv                         # 单样本 CSV
spindle-predict --csv batch.csv --batch                  # 批量推理
spindle-predict --npy window.npy --op PPOP40_1           # numpy 数组
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..paths import OUTPUTS_DIR
from . import postprocess_cwt_calibration as post
from . import train_cwt_residual_attention_compare as base
from .models.logit_calibrator import ConditionAwareCalibrator

FEATURE_COLUMNS = base.FEATURE_COLUMNS
CLASS_NAMES = base.CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="主轴故障十分类推理")
    parser.add_argument(
        "--model-dir",
        default=str(OUTPUTS_DIR / "self_train_v3_gru"),
        help="主干模型输出目录（默认 R3 GRU，Accuracy 0.9605）",
    )
    parser.add_argument(
        "--calibrator-dir",
        default=str(OUTPUTS_DIR / "self_train_v3_gru_calibrated"),
        help="校准器输出目录（默认 R3 GRU 校准器）",
    )
    parser.add_argument("--csv", help="输入 CSV 文件路径，需包含 9 列传感器数据和 op 列")
    parser.add_argument("--npy", help="输入 .npy 文件路径，shape [window_size, 9]")
    parser.add_argument("--op", default="PPOP40_1", help="工位标识，用于条件校准器")
    parser.add_argument("--batch", action="store_true", help="CSV 文件包含多个窗口（按行）")
    parser.add_argument(
        "--tta-time-shifts", type=str, default="0,-2,2", help="TTA 时间平移"
    )
    parser.add_argument("--tta-scale-shifts", type=str, default="0", help="TTA 尺度平移")
    parser.add_argument("--device", default="cuda", help="推理设备")
    return parser.parse_args()


def load_models(model_dir: Path, calibrator_dir: Path, device: torch.device):
    """加载主干和校准器。"""
    summary = json.loads((model_dir / "run_summary.json").read_text(encoding="utf-8"))
    calib_details = json.loads(
        (calibrator_dir / "calibrator_details.json").read_text(encoding="utf-8")
    )
    condition_mapping = calib_details.get("condition_mapping", {})

    backbone = post.build_model(summary, device=device)
    backbone.load_state_dict(
        torch.load(model_dir / "best_model.pt", map_location=device)
    )
    backbone.eval()

    calibrator = ConditionAwareCalibrator(
        num_classes=len(CLASS_NAMES),
        num_conditions=len(condition_mapping),
        embedding_dim=calib_details.get("embedding_dim", 8),
        hidden_dim=calib_details.get("hidden_dim", 32),
        dropout=calib_details.get("dropout", 0.1),
    ).to(device)
    calibrator.load_state_dict(
        torch.load(calibrator_dir / "best_calibrator.pt", map_location=device)
    )
    calibrator.eval()

    return backbone, calibrator, condition_mapping, summary


@torch.no_grad()
def predict_single(
    raw_seq: np.ndarray,
    op: str,
    backbone: torch.nn.Module,
    calibrator: torch.nn.Module,
    condition_mapping: dict,
    summary: dict,
    shift_pairs: list,
    device: torch.device,
):
    """单样本推理，返回预测类别、置信度和各类概率。"""
    # CWT 时频图
    cwt_wavelets = [str(w) for w in summary["cwt_wavelets"]]
    window_size = int(summary["window_size"])
    cwt_scales = int(summary["gadf_points"])
    gradient_mix = float(summary.get("cwt_gradient_mix", 0.15))
    add_diff = bool(summary.get("cwt_add_diff", False))

    if raw_seq.shape[0] != window_size:
        # 重采样到目标长度
        raw_seq = base.resample_series(raw_seq.astype(np.float32), window_size)
    raw_seq = raw_seq.astype(np.float32)
    if raw_seq.ndim == 1:
        raw_seq = raw_seq.reshape(-1, len(FEATURE_COLUMNS))

    cwt_img = base.build_cwt_stack(raw_seq, cwt_scales, cwt_wavelets, gradient_mix, add_diff=add_diff)
    cwt_tensor = torch.from_numpy(cwt_img).unsqueeze(0).to(device)
    raw_tensor = torch.from_numpy(raw_seq).unsqueeze(0).to(device)

    # TTA
    logits_sum = None
    for shift_y, shift_x in shift_pairs:
        cwt_tta = post.shift_with_replicate(cwt_tensor, shift_y=shift_y, shift_x=shift_x)
        logits, _, _ = backbone(raw_tensor, cwt_tta)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    tta_logits = logits_sum / max(len(shift_pairs), 1)

    # 校准
    cond_id = torch.tensor(
        [condition_mapping.get(op, 0)], dtype=torch.long, device=device
    )
    calibrated = calibrator(tta_logits, cond_id)

    probs = F.softmax(calibrated, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs[pred], probs


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model_dir = Path(args.model_dir)
    calibrator_dir = Path(args.calibrator_dir)
    backbone, calibrator, condition_mapping, summary = load_models(
        model_dir, calibrator_dir, device
    )
    print(f"Model: {summary['model_name']} | Wavelets: {summary['cwt_wavelets']}")

    shift_pairs = [
        (int(sy), int(sx))
        for sy in args.tta_scale_shifts.split(",")
        for sx in args.tta_time_shifts.split(",")
    ]
    condition_id = condition_mapping.get(args.op, 0)
    print(f"OP: {args.op} (id={condition_id}) | TTA: {len(shift_pairs)} shifts")

    # 加载输入
    if args.csv:
        import pandas as pd

        df = pd.read_csv(args.csv)
        if not args.batch:
            if "op" in df.columns:
                args.op = str(df.iloc[0]["op"])
            raw_seq = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            pred, conf, probs = predict_single(
                raw_seq, args.op, backbone, calibrator,
                condition_mapping, summary, shift_pairs, device,
            )
            print(f"\nPrediction: {CLASS_NAMES[pred]} (confidence: {conf:.4f})")
            print("\nAll class probabilities:")
            for i, p in enumerate(probs):
                bar = "█" * int(p * 40)
                print(f"  {CLASS_NAMES[i]:<30s}  {p:6.2%}  {bar}")
        else:
            # 批量：按 op 分组后推理
            import pandas as pd

            predictions = []
            for (op,), group in df.groupby("op", sort=False):
                seq = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
                for i in range(len(seq)):
                    start = max(0, i - summary["window_size"] + 1)
                    window = seq[start : i + 1]
                    if len(window) < summary["window_size"]:
                        pad = np.zeros(
                            (summary["window_size"] - len(window), len(FEATURE_COLUMNS)),
                            dtype=np.float32,
                        )
                        window = np.concatenate([pad, window], axis=0)
                    pred, conf, _ = predict_single(
                        window, str(op), backbone, calibrator,
                        condition_mapping, summary, shift_pairs, device,
                    )
                    predictions.append(
                        {"op": op, "row": i, "prediction": CLASS_NAMES[pred], "confidence": conf}
                    )
            for p in predictions:
                print(f"  op={p['op']} row={p['row']} → {p['prediction']} ({p['confidence']:.4f})")

    elif args.npy:
        raw_seq = np.load(args.npy).astype(np.float32)
        pred, conf, probs = predict_single(
            raw_seq, args.op, backbone, calibrator,
            condition_mapping, summary, shift_pairs, device,
        )
        print(f"\nPrediction: {CLASS_NAMES[pred]} (confidence: {conf:.4f})")
        print("\nAll class probabilities:")
        for i, p in enumerate(probs):
            bar = "█" * int(p * 40)
            print(f"  {CLASS_NAMES[i]:<30s}  {p:6.2%}  {bar}")

    else:
        print("请通过 --csv 或 --npy 指定输入文件。使用 --help 查看用法。")


if __name__ == "__main__":
    main()

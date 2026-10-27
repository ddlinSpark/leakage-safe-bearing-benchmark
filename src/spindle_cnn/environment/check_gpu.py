"""Check PyTorch CUDA availability and optionally run a matrix multiplication."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 PyTorch、CUDA 和 GPU 计算能力")
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="只报告设备信息，不执行矩阵乘法测试",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    print("=" * 50)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"当前设备数量: {torch.cuda.device_count()}")
        print(f"当前设备名称: {torch.cuda.get_device_name(0)}")
        print(f"当前设备算力: {torch.cuda.get_device_capability(0)}")

        if not args.skip_compute:
            print("\n--- 开始 GPU 计算测试 ---")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.mm(x, y)
            print(f"GPU 计算成功! 输出张量形状: {z.shape}")
            print(f"设备: {z.device}")
            print("--- GPU 测试通过 ---")
    else:
        print("\nCUDA 不可用，请检查 NVIDIA 驱动、PyTorch 的 CUDA 版本和运行环境。")

    print("=" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx.py — 把训练好的 eegnet.pth 导出为 ONNX（固定输入形状）

流程
----
  1. 加载 model/eegnet.pth（state_dict）
  2. 重建 EEGNet 模型并置为 eval 模式
  3. dummy input (1,1,22,500)，torch.onnx.export 导出，**不设置动态轴**（固定形状）
  4. （可选）用 onnxruntime 验证 ONNX 输出与 PyTorch 输出一致

输出
----
  model/eegnet.onnx

用法
----
  python3 model/export_onnx.py
  python3 model/export_onnx.py --opset 12   # 指定 ONNX opset（默认 12，RKNN 兼容好）

说明
----
  - opset 建议 12~13，rknn-toolkit2 对这两个版本解析最稳；新版也支持到 17。
  - 若导出后 onnxruntime 校验未装，会自动跳过校验并提示（不影响 .onnx 生成）。
  - 后续 convert_rknn.py 依赖本脚本产出的 eegnet.onnx。

依赖：torch, onnx, onnxruntime（可选，用于校验）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT = MODEL_DIR / "eegnet.pth"
ONNX_PATH = MODEL_DIR / "eegnet.onnx"

sys.path.insert(0, str(MODEL_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch → ONNX 导出")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset 版本（默认 12）")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 onnxruntime 输出一致性校验")
    args = parser.parse_args()

    import torch  # noqa: PLC0415

    from eegnet import EEGNet  # noqa: PLC0415

    if not CHECKPOINT.exists():
        raise SystemExit(
            f"✗ 权重文件缺失：{CHECKPOINT}\n"
            "  请先训练模型：python3 model/train.py"
        )

    # ---- 加载模型 ----
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = EEGNet(n_channels=8, n_classes=2, input_length=500,
                   F1=8, D=2, F2=16, kernel_length=64, dropout=0.5)
    model.load_state_dict(state)
    model.eval()
    print(f"已加载权重：{CHECKPOINT}（参数量 {model.num_params():,}）")

    # ---- 导出 ONNX（固定形状，无动态轴）----
    dummy = torch.randn(1, 1, 8, 500, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["eeg_input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        # 关键：不传 dynamic_axes，即所有轴固定为导出时的形状
    )
    print(f"✓ ONNX 已导出：{ONNX_PATH}")

    # ---- 校验：PyTorch vs ONNX 输出 ----
    if not args.no_verify:
        _verify_onnx(model, dummy, ONNX_PATH)

    print("\n下一步：python3 model/calibration.py（生成 INT8 校准集）")
    return 0


def _verify_onnx(model, dummy, onnx_path: Path) -> None:
    """用 onnxruntime 跑一遍，对比 PyTorch 输出，打印最大误差。"""
    try:
        import onnx  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError:
        print("  · 未安装 onnx/onnxruntime，跳过输出一致性校验。")
        print("    可执行：pip install onnx onnxruntime -i "
              "https://pypi.tuna.tsinghua.edu.cn/simple")
        return

    onnx.checker.check_model(str(onnx_path))
    print("  ✓ ONNX 模型结构检查通过")

    import torch  # noqa: PLC0415
    with torch.no_grad():
        ref = model(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"eeg_input": dummy.numpy()})[0]

    max_err = float(np.abs(ref - out).max())
    print(f"  PyTorch 输出 shape: {ref.shape}, ONNX 输出 shape: {out.shape}")
    print(f"  最大绝对误差: {max_err:.2e}")
    if np.argmax(ref) == np.argmax(out):
        print("  ✓ 分类结果一致，导出正确")
    else:
        print("  ⚠ 分类结果不一致，请检查导出参数")


if __name__ == "__main__":
    sys.exit(main())

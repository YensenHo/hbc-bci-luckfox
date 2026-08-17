#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_features.py — 导出「到 flatten 的 features ONNX」+「fc 权重 C 数组」

P2-1 方案：NPU 只算到 240 维特征（避开输出层 INT8 量化误差），
最后一层 240→2 的 fc 用 CPU（C 代码 FP32 matmul，480 个乘加零成本）。

输出：
  model/eegnet_features.onnx   到 flatten（输出 240 维特征）
  board/fc_weights.h           fc 权重（2×240）+ bias（2）C 数组

用法：
  python3 model/export_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT / "model"
BOARD_DIR = PROJECT / "board"
CHECKPOINT = MODEL_DIR / "eegnet.pth"
FEAT_ONNX = MODEL_DIR / "eegnet_features.onnx"
FC_HEADER = BOARD_DIR / "fc_weights.h"

sys.path.insert(0, str(MODEL_DIR))


def main() -> int:
    import torch  # noqa: PLC0415
    from eegnet import EEGNet  # noqa: PLC0415

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = EEGNet(n_channels=8, n_classes=2, input_length=500,
                   F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)
    model.load_state_dict(state)
    model.eval()

    # ---- 1. 导出 features ONNX（到 flatten，240 维）----
    dummy = torch.randn(1, 1, 8, 500, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(FEAT_ONNX),
        input_names=["eeg_input"], output_names=["features"],
        opset_version=12, do_constant_folding=True,
    )
    print(f"✓ features ONNX 已导出：{FEAT_ONNX}")

    # ---- 2. 导出 fc 权重（2×240 + bias 2）----
    w = model.fc.weight.data.cpu().numpy()   # (2, 240, 1, 1)
    b = model.fc.bias.data.cpu().numpy()     # (2,)
    w = w.reshape(2, 240)
    n_in, n_out = w.shape[1], w.shape[0]

    lines = [
        "/* fc_weights.h — 自动生成自 eegnet.pth 的 fc 层（Conv2d(240,2,1x1) → 等价 Linear(240,2)）",
        " * 生成命令：python3 model/export_features.py  — 勿手改 */",
        f"#define FC_IN  {n_in}",
        f"#define FC_OUT {n_out}",
        "static const float fc_w[FC_OUT][FC_IN] = {",
    ]
    for i in range(n_out):
        row = ", ".join(f"{v:.9g}f" for v in w[i])
        lines.append(f"    {{{row}}},")
    lines.append("};")
    lines.append(f"static const float fc_b[FC_OUT] = {{ {', '.join(f'{v:.9g}f' for v in b)} }};")
    FC_HEADER.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ fc 权重已导出：{FC_HEADER}（{n_out}×{n_in} + bias {n_out}）")

    # ---- 3. 自检：CPU 版 features+fc 与完整 forward 一致 ----
    with torch.no_grad():
        feat = model.forward_features(dummy).numpy()      # (1, 240)
        logits_ref = model(dummy).numpy()                 # (1, 2)
        logits_calc = feat @ w.T + b                       # (1, 2)
    max_err = float(np.abs(logits_ref - logits_calc).max())
    print(f"  ✓ fc 拆 CPU 自检：logits 最大误差 {max_err:.2e}（应 ≈0）")
    if max_err > 1e-4:
        raise SystemExit("✗ fc 拆 CPU 自检失败，权重导出有误")
    return 0


if __name__ == "__main__":
    sys.exit(main())

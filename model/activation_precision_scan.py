#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""activation_precision_scan.py — 解码器中间激活的数值精度扫描（P1-④ 完整版）

在 EEGNet 每个中间激活（ELU 输出）后插 forward hook，round 到不同位深，
扫全 1296 试次，报「激活位深 → vs FP32 argmax 一致率」。
与 weight_precision_scan.py（权重位深）合起来，定位「解码器整体需要多少 bit」。

用法：python3 model/activation_precision_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT / "model"
sys.path.insert(0, str(MODEL_DIR))


def quantize_tensor(w: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 32:
        return w.astype(np.float32)
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(w)) / qmax
    if scale == 0:
        return w.astype(np.float32)
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return (q * scale).astype(np.float32)


def main() -> int:
    import torch  # noqa: PLC0415
    from eegnet import EEGNet  # noqa: PLC0415

    d = np.load(PROJECT / "data" / "processed" / "eval.npz")
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)

    ckpt = torch.load(MODEL_DIR / "eegnet.pth", map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    def run_at_act_bits(bits: int):
        model = EEGNet(n_channels=8, n_classes=2, input_length=500,
                       F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)
        model.load_state_dict(state)
        model.eval()

        hooks = []

        def make_hook(bits_):
            def hook(module, inp, out):
                return torch.from_numpy(quantize_tensor(out.detach().cpu().numpy(), bits_)).to(out.dtype)
            return hook

        # 只 hook ELU 输出（激活函数的输出，是最关键的中间激活）
        for m in model.modules():
            if isinstance(m, torch.nn.ELU):
                hooks.append(m.register_forward_hook(make_hook(bits)))

        preds = []
        try:
            with torch.no_grad():
                for i in range(0, len(X), 256):
                    out = model(torch.from_numpy(X[i:i + 256]))
                    preds.append(out.argmax(dim=1).numpy())
        finally:
            for h in hooks:
                h.remove()
        return np.concatenate(preds)

    # FP32 基线
    pred_fp32 = run_at_act_bits(32)
    N = len(y)
    fp32_acc = (pred_fp32 == y).sum()

    print(f"=== 解码器中间激活位深扫描（{N} 试次，ELU 输出 round）===\n")
    print(f"{'激活位深':>8} | {'准确率':>8} | {'vs FP32 一致率':>14}")
    print("-" * 40)
    print(f"{'FP32':>8} | {100.0 * fp32_acc / N:7.2f}% | {'100.00%':>12}")
    for bits in [16, 14, 12, 10, 8, 6]:
        pred = run_at_act_bits(bits)
        acc = (pred == y).sum()
        agree = (pred == pred_fp32).sum()
        print(f"{'int' + str(bits):>8} | {100.0 * acc / N:7.2f}% | {100.0 * agree / N:12.2f}%")

    print("\n（结合 weight_precision_scan：权重 int10 保精度、激活需 X bit，合起来即解码器精度下限）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

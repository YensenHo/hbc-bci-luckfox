#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weight_precision_scan.py — 解码器权重的数值精度扫描（P1-④）

把 EEGNet 权重量化到不同位深（float→int16/12/10/8，带逐张量 scale），
扫全 1296 试次，报「位深 → vs FP32 argmax 一致率」。

目的：把 NPU 失败翻成卖点——「HBC 传输侧 5bit 就够（已做），
但解码器权重需要 X bit 才能 ≥99% 一致，NPU 的单-scale INT8 天然装不下，
所以 CPU FP32 才是正确部署」。

用法：python3 model/weight_precision_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT / "model"
sys.path.insert(0, str(MODEL_DIR))


def quantize_tensor(w: np.ndarray, bits: int) -> np.ndarray:
    """把张量量化到 bits 位（对称量化，带 scale），返回反量化后的近似值。"""
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
    X = d["X"].astype(np.float32)   # (1296,1,8,500)
    y = d["y"].astype(np.int64)

    ckpt = torch.load(MODEL_DIR / "eegnet.pth", map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    def run_at_bits(bits: int):
        model = EEGNet(n_channels=8, n_classes=2, input_length=500,
                       F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)
        model.load_state_dict(state)
        model.eval()
        # 量化所有权重（含 BN 参数，BN 在 eval 下已折叠进卷积）
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(torch.from_numpy(quantize_tensor(p.data.cpu().numpy(), bits)))
        # 全量推理
        correct = 0
        logits_all = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i:i + 256])
                out = model(xb)
                logits_all.append(out.numpy())
                pred = out.argmax(dim=1).numpy()
                correct += int((pred == y[i:i + 256]).sum())
        return correct, len(X)

    # FP32 基线
    fp32_correct, N = run_at_bits(32)
    print(f"=== 解码器权重位深扫描（{N} 试次）===\n")
    print(f"{'位深':>6} | {'准确率':>8} | {'vs FP32 一致率':>14}")
    print("-" * 40)
    print(f"{'FP32':>6} | {100.0 * fp32_correct / N:7.2f}% | {'100.00%':>12}")
    for bits in [16, 14, 12, 10, 8, 6]:
        correct, _ = run_at_bits(bits)
        acc = 100.0 * correct / N
        print(f"{'int' + str(bits):>6} | {acc:7.2f}% | {100.0 * correct / fp32_correct:12.2f}%")

    print("\n（vs FP32 一致率 = 该位深与 FP32 预测完全相同的试次占比）")
    print("结论：找到一致率 ≥99% 的最小位深，即解码器所需的精度下限。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

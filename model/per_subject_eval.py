#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""per_subject_eval.py — 按被试分组的准确率统计（P1-⑥）

用 eval_subject_idx.npy 把全量 1296 试次按 9 个被试分组，
分别报每被试准确率 + 均值±标准差，回答「76.39% 是不是某个好被试的幻觉」。

用法：python3 model/per_subject_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT / "model"
sys.path.insert(0, str(MODEL_DIR))


def main() -> int:
    import torch  # noqa: PLC0415
    from eegnet import EEGNet  # noqa: PLC0415

    d = np.load(PROJECT / "data" / "processed" / "eval.npz")
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    subj = np.load(PROJECT / "data" / "processed" / "eval_subject_idx.npy")

    ckpt = torch.load(MODEL_DIR / "eegnet.pth", map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = EEGNet(n_channels=8, n_classes=2, input_length=500,
                   F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)
    model.load_state_dict(state)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out = model(torch.from_numpy(X[i:i + 256]))
            preds.append(out.argmax(dim=1).numpy())
    pred = np.concatenate(preds)

    print("=== 按被试分组的准确率（9 被试 × 144 试次）===\n")
    print(f"{'被试':>5} | {'准确率':>8} | {'正确/总数':>10}")
    print("-" * 32)
    accs = []
    for s in range(1, 10):
        mask = subj == s
        correct = int((pred[mask] == y[mask]).sum())
        n = int(mask.sum())
        acc = 100.0 * correct / n
        accs.append(acc)
        print(f"A{s:02d}   | {acc:7.2f}% | {correct:>4}/{n:<4}")
    accs = np.array(accs)
    print("-" * 32)
    print(f"均值      | {accs.mean():7.2f}%")
    print(f"标准差    | {accs.std():7.2f}%")
    print(f"最低~最高 | {accs.min():.2f}%~{accs.max():.2f}%")
    print(f"\n全量 1296 试次整体准确率: {100.0 * (pred == y).sum() / len(y):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

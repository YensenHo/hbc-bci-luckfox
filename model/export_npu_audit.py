#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_npu_audit.py — 导出 NPU 全测试集对账数据（X + 真值 y + PyTorch CPU logits）

供板子 NPU 逐样本比对，验证 +44*scale workaround 是否全局成立。

输出（data/npu_audit/）：
    X.bin            N × 4000 float32（8ch×500 展平）
    y.bin            N × int32
    cpu_logits.bin   N × 2 float32（PyTorch eval 模式 logits，真值）

用法：
  python3 model/export_npu_audit.py --n 288
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT / "model"
PROCESSED = PROJECT / "data" / "processed"
OUT = PROJECT / "data" / "npu_audit"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=288)
    parser.add_argument("--activation", type=str, default="elu",
                        help="激活函数 elu/relu（与训练时的 activation 一致）")
    args = parser.parse_args()
    N = args.n

    sys.path.insert(0, str(MODEL_DIR))
    from eegnet import EEGNet  # noqa: PLC0415

    d = np.load(PROCESSED / "eval.npz")
    X = d["X"][:N].astype(np.float32)   # (N,1,8,500)
    y = d["y"][:N].astype(np.int32)

    model = EEGNet(n_channels=8, n_classes=2, activation=args.activation)
    ckpt = torch.load(MODEL_DIR / "eegnet.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        cpu_logits = model(torch.from_numpy(X)).numpy().astype(np.float32)
    cpu_pred = cpu_logits.argmax(1)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "X.bin").write_bytes(X.ravel().tobytes())
    (OUT / "y.bin").write_bytes(y.tobytes())
    (OUT / "cpu_logits.bin").write_bytes(cpu_logits.tobytes())

    print(f"✓ 导出 {N} 个试次 → {OUT}/")
    print(f"  类别分布[左/右] = {np.bincount(y, minlength=2).tolist()}")
    print(f"  CPU(PyTorch) 预测准确率基准：{(cpu_pred == y).mean()*100:.2f}%")
    print(f"  CPU logits 范围: [{cpu_logits.min():.3f}, {cpu_logits.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

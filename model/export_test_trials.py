#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_test_trials.py — 导出 N 个评估集试次为二进制，供板载融合闭环读取

二进制格式（小端）：
    int32  N（试次数）
    每个试次: float32×4000（X，8ch×500 展平）+ int32×1（y 标签 0/1）

用法
----
  python3 model/export_test_trials.py --n 100 --out data/fusion_test.bin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=str, default=str(PROJECT / "data" / "fusion_test.bin"))
    args = parser.parse_args()

    d = np.load(PROCESSED / "eval.npz")
    X = d["X"][: args.n].astype(np.float32)      # (N,1,8,500)
    y = d["y"][: args.n].astype(np.int32)
    N = X.shape[0]

    with open(args.out, "wb") as f:
        f.write(np.int32(N).tobytes())
        for i in range(N):
            f.write(X[i].ravel().tobytes())       # 4000 float32
            f.write(np.int32(y[i]).tobytes())

    print(f"✓ 导出 {N} 个试次 → {args.out}")
    print(f"  类别分布[左/右] = {np.bincount(y, minlength=2).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

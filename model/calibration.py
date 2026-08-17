#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibration.py — 生成 RKNN INT8 量化校准集

原理
----
RKNN Toolkit2 的 INT8 量化需要一个「校准集」：一批有代表性的真实输入样本，
用来统计每层激活的数值范围（min/max），从而把 FP32 权重/激活映射到 INT8。
校准集越贴近真实推理输入，量化精度损失越小。

本脚本
------
  1. 从训练集 train.npz 随机取 N 个样本（默认 200，可 --num 调整）
  2. 每个样本单独存成一个 .npy（形状 (1,1,22,500) float32，与模型输入一致）
  3. 写一个 calib_data.txt，每行一个 .npy 的**绝对路径**

输出
----
  model/calib_data/0000.npy ... 00NN.npy
  model/calib_data.txt

用法
----
  python3 model/calibration.py                 # 默认 200 个样本
  python3 model/calibration.py --num 500       # 500 个样本
  python3 model/calibration.py --seed 42

说明
----
  - 采样用的是「已 z-score 标准化」的输入（与推理输入一致），
    因此 convert_rknn.py 里 mean_values=[[0]]、std_values=[[1]]（即不做额外归一）。
  - 本脚本可在任意平台运行（只需 numpy），生成的 txt 拿到 Linux x86 主机上给
    convert_rknn.py 用。

依赖：numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("BCI_DATA_ROOT", str(PROJECT_ROOT / "data")))
PROCESSED_DIR = Path(os.environ.get("BCI_DATA_PROCESSED", str(DATA_ROOT / "processed")))
MODEL_DIR = Path(__file__).resolve().parent

CALIB_DIR = MODEL_DIR / "calib_data"
CALIB_TXT = MODEL_DIR / "calib_data.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 RKNN INT8 量化校准集")
    parser.add_argument("--num", type=int, default=500,
                        help="校准样本数（默认 500）")
    parser.add_argument("--seed", type=int, default=42,
                        help="采样随机种子（默认 42）")
    args = parser.parse_args()

    train_npz = PROCESSED_DIR / "train.npz"
    if not train_npz.exists():
        raise SystemExit(
            f"✗ 训练集缺失：{train_npz}\n"
            "  请先下载并预处理数据：\n"
            "    python3 data/download_bcic2a.py\n"
            "    python3 data/preprocess.py"
        )

    with np.load(train_npz, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)

    n_total = len(X)
    n_pick = min(args.num, n_total)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_total, size=n_pick, replace=False)
    idx = np.sort(idx)

    # 清理旧校准数据，避免残留文件导致 txt 与实际不一致
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    for old in CALIB_DIR.glob("*.npy"):
        old.unlink()

    lines: list[str] = []
    for i, src_idx in enumerate(idx):
        sample = X[src_idx]                       # (1, 8, 500)
        sample = sample[np.newaxis, ...]          # (1, 1, 8, 500) —— 确保单样本带 batch 维
        sample = np.ascontiguousarray(sample, dtype=np.float32)
        path = CALIB_DIR / f"{i:04d}.npy"
        np.save(path, sample)
        # 写相对 MODEL_DIR 的路径：rknn build 读 txt 时按「txt 所在目录」解析，
        # 故 txt 里不能带 model/ 前缀（否则拼成 model/model/...）
        lines.append(str(path.relative_to(MODEL_DIR)))

    CALIB_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"训练集样本总数 : {n_total}")
    print(f"校准样本数     : {n_pick}")
    print(f"样本形状       : {sample.shape} {sample.dtype}")
    print(f"校准目录       : {CALIB_DIR}")
    print(f"✓ 校准集清单已写入：{CALIB_TXT}（{len(lines)} 行绝对路径）")
    print("\n下一步（Linux x86 / WSL2 主机）：python3 model/convert_rknn.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

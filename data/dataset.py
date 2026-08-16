#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py — PyTorch Dataset 加载器

读取 preprocess.py 生成的 .npz（X: (n,1,22,500) float32, y: (n,) int64），
封装成 torch.utils.data.Dataset 子类，供 train.py 直接喂给 DataLoader。

用法
----
  from dataset import EEGDataset
  ds = EEGDataset("data/processed/train.npz")
  x, y = ds[0]          # x: (1,22,500) float32, y: 标量 long

依赖：torch, numpy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    EEG 试次数据集（torch.utils.data.Dataset 子类）。

    参数
    ----
    npz_path : str | Path
        preprocess.py 输出的 .npz 文件路径（需含 X、y 两个数组）。
    """

    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"✗ 数据集文件不存在：{self.npz_path}\n"
                "  请先运行预处理：python3 data/preprocess.py"
            )
        with np.load(self.npz_path, allow_pickle=False) as data:
            if "X" not in data or "y" not in data:
                raise KeyError(
                    f"✗ {self.npz_path.name} 缺少 X/y 字段，实际字段：{list(data.keys())}"
                )
            self.X = np.asarray(data["X"], dtype=np.float32)
            self.y = np.asarray(data["y"], dtype=np.int64)

        if self.X.ndim != 4:
            raise ValueError(
                f"✗ X 应为 (n,1,22,500) 四维数组，实际 {self.X.shape}"
            )
        if len(self.y) != len(self.X):
            raise ValueError(
                f"✗ X({len(self.X)}) 与 y({len(self.y)}) 样本数不一致"
            )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (x, y)。x 为 float32 (1,22,500)，y 为 int64 标量。"""
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y

    @property
    def shape(self) -> tuple:
        return tuple(self.X.shape)

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1


if __name__ == "__main__":
    # 简单自检：加载训练集打印形状与类别分布
    import sys

    if len(sys.argv) < 2:
        print("用法：python3 data/dataset.py data/processed/train.npz")
        sys.exit(0)

    ds = EEGDataset(sys.argv[1])
    print(f"文件：{ds.npz_path}")
    print(f"形状：X={ds.shape}, y 长度={len(ds)}, 类别数={ds.n_classes}")
    x, y = ds[0]
    print(f"首样本：x={tuple(x.shape)} {x.dtype}, y={y.item()} {y.dtype}")

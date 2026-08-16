#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""visualize_eeg.py — 可视化真实脑电数据（数据探索工具）

从 data/processed/eval.npz 抽取真实试次，画 8 通道脑电波形图。
这是真实脑电数据的可视化产物（BCIC IV 2a 公开数据集，运动想象）。

用法：
  python3 data/visualize_eeg.py              # 默认第 42 个试次
  python3 data/visualize_eeg.py --trial 0    # 指定试次
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"
OUT = PROJECT / "data" / "eeg_waveform.png"
FS = 250  # Hz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=42)
    args = parser.parse_args()

    d = np.load(PROCESSED / "eval.npz")
    X, y = d["X"], d["y"]
    trial = X[args.trial, 0]                 # (8, 500) 一个真实试次
    label = "左手" if y[args.trial] == 0 else "右手"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    BG, GRAY, WHITE = "#0D1117", "#8FA3BF", "#E8ECEF"
    ch_names = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
    colors = ["#00B4D8", "#4FC3F7", "#00B4D8", "#E76F51", "#FFB74D",
              "#E76F51", "#81C784", "#81C784"]
    t = np.arange(500) / FS

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    for i in range(8):
        yv = trial[i] - trial[i].mean()
        offset = (7 - i) * 40
        ax.plot(t, yv + offset, color=colors[i], lw=1.1, alpha=0.95)
        ax.text(-0.06, offset, ch_names[i], ha="right", va="center",
                color=GRAY, fontsize=11, weight="bold")
    ax.set_xlim(0, 2)
    ax.set_ylim(-60, 7 * 40 + 60)
    ax.set_yticks([])
    ax.set_xlabel("时间 (秒)", color=GRAY, fontsize=13)
    ax.set_title(f"大脑在想「{label}」的时候，8 个通道的信号长这样", color=WHITE,
                 fontsize=18, weight="bold", pad=14)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=GRAY, labelsize=11)
    ax.grid(axis="x", color=GRAY, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=BG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ 脑电波形已保存：{OUT}（试次 {args.trial}，标签 {label}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py — BCIC IV 2a 预处理（.mat 版）：8 通道二分类运动想象

数据源变更说明（2026-08）
------------------------
  .gdf 源已从 Graz 服务器移除（404），改为下载 .mat（lampx.tugraz.at）。
  .mat 结构（struct_as_record=False, squeeze_me=True）：
    m['data'] 是 9 元素数组：
      data[0..2] : 基线/睁眼/闭眼/运动记录（无试次，跳过）
      data[3..8] : 6 个 MI run，每个含 X(96735,25)、trial(48,)、y(48,)、fs=250
  - X 单位 μV，形状 (样本数, 25)，25 通道 = 22 EEG + 3 EOG
  - trial = 试次起点（t=0 注视，1-indexed）；cue 在 trial+500 样本（t=2s）
  - y = 标签 1~4（1=左手 2=右手 3=双脚 4=舌头）

二分类 + 8 通道（对齐 ADS1299 硬件，Codex 收敛结论）
----------------------------------------------------
  - 只取 y∈{1,2}（左/右手）→ 映射为 0/1
  - 只取 8 个运动区电极：FC3 FCz FC4 C3 Cz C4 CP3 CP4
    （在 25 通道里索引 [1,3,5,7,9,11,13,17]）
  - 分段：cue 后 [0.5, 2.5]s = [trial+625, trial+1125]，共 500 样本
  - 滤波：0.5–50 Hz 带通 + 50 Hz 陷波（scipy，4 阶 Butterworth）
  - 每段 z-score 标准化

输出
----
  data/processed/train.npz  训练集（A{nn}T 会话）
  data/processed/eval.npz   评估集（A{nn}E 会话）
  X shape (n, 1, 8, 500) float32, y shape (n,) int64（0/1）

用法
----
  python3 data/preprocess.py --subjects 1 2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("BCI_DATA_ROOT", str(PROJECT_ROOT / "data")))
RAW_DIR = Path(os.environ.get("BCI_DATA_RAW", str(DATA_ROOT / "raw")))
PROCESSED_DIR = Path(os.environ.get("BCI_DATA_PROCESSED", str(DATA_ROOT / "processed")))

FS = 250
T_CUE_S = 2.0                  # cue 相对 trial 起点的秒数
WIN = (0.5, 2.5)               # cue 后 [0.5, 2.5]s
START = int((T_CUE_S + WIN[0]) * FS)   # 625
STOP = int((T_CUE_S + WIN[1]) * FS)    # 1125
N_SAMPLES = STOP - START               # 500
N_CHANNELS = 8

# 25 通道顺序（22 EEG + 3 EOG）
CH_25 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
         "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP3", "CP1", "CPz", "CP2", "CP4",
         "P1", "Pz", "P2", "POz",
         "EOG1", "EOG2", "EOG3"]
# 8 个运动区电极（二分类左/右手最关键的区域）
MOTOR_8 = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
MOTOR_IDX = [CH_25.index(c) for c in MOTOR_8]   # [1,3,5,7,9,11,13,17]

KEEP_LABELS = {1: 0, 2: 1}     # 左手→0, 右手→1


def _load_scipy():
    try:
        from scipy.io import loadmat  # noqa: PLC0415
        from scipy.signal import butter, filtfilt, iirnotch  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "✗ 未安装 scipy。请执行：\n"
            "    pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            f"  原始错误：{exc}"
        ) from exc
    return loadmat, butter, filtfilt, iirnotch


def _filter_band(x: np.ndarray, b, a) -> np.ndarray:
    """沿时间轴（axis=1）做零相位带通滤波。"""
    from scipy.signal import filtfilt  # noqa: PLC0415
    return filtfilt(b, a, x, axis=1)


def _process_session(mat_path: Path, loadmat, butter, filtfilt, iirnotch
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """读取一个会话 .mat，返回 (X, y) 或 None。"""
    print(f"  处理 {mat_path.name} ...")
    m = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    data = m["data"]

    # 带通 0.5–50 Hz + 50 Hz 陷波（对整段连续数据滤波，再分段，避免边界效应）
    b_band, a_band = butter(4, [0.5, 50.0], btype="band", fs=FS)
    b_notch, a_notch = iirnotch(50.0, 30.0, fs=FS)

    X_list, y_list = [], []
    for run in data:
        trial = np.asarray(run.trial).ravel()
        if trial.size == 0:
            continue                      # 跳过无试次的基线 run
        X = np.asarray(run.X, dtype=np.float64)   # (n_samples, 25)
        y = np.asarray(run.y).ravel()

        # 滤波（全通道 22 EEG，EOG 丢弃前先滤，顺序无关）
        X = filtfilt(b_band, a_band, X, axis=0)
        X = filtfilt(b_notch, a_notch, X, axis=0)

        for onset, label in zip(trial, y):
            label = int(label)
            if label not in KEEP_LABELS:
                continue                  # 只要左/右手
            start = int(onset) + START
            stop = int(onset) + STOP
            if stop > X.shape[0]:
                continue                  # 越界段丢弃
            seg = X[start:stop, MOTOR_IDX]        # (500, 8)
            X_list.append(seg.T)                  # (8, 500)
            y_list.append(KEEP_LABELS[label])

    if not X_list:
        return None
    X = np.stack(X_list).astype(np.float32)       # (n, 8, 500)
    y = np.asarray(y_list, dtype=np.int64)

    # 每段 z-score（跨通道+时间整体）
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True)
    X = (X - mean) / (std + 1e-8)
    X = X[:, np.newaxis, :, :]                    # (n, 1, 8, 500)
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser(description="BCIC IV 2a 预处理（.mat → 8ch 二分类 npz）")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    args = parser.parse_args()

    loadmat, butter, filtfilt, iirnotch = _load_scipy()
    subjects = sorted(set(args.subjects))

    train_X, train_y, eval_X, eval_y = [], [], [], []
    for subj in subjects:
        for sess, buckets in (("T", (train_X, train_y)), ("E", (eval_X, eval_y))):
            p = RAW_DIR / f"A{subj:02d}{sess}.mat"
            if not p.exists():
                print(f"  ⚠ 缺失 {p.name}，跳过")
                continue
            res = _process_session(p, loadmat, butter, filtfilt, iirnotch)
            if res is not None:
                buckets[0].append(res[0]); buckets[1].append(res[1])

    if not train_X and not eval_X:
        raise SystemExit("✗ 无数据。请先下载：python3 data/download_bcic2a.py")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _save(name, Xs, ys):
        if not Xs:
            print(f"  ⚠ {name}: 无数据，跳过")
            return
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        np.savez_compressed(PROCESSED_DIR / name, X=X.astype(np.float32), y=y.astype(np.int64))
        counts = np.bincount(y, minlength=2)
        print(f"  ✓ {name}: X={X.shape}, y={y.shape}")
        print(f"      类别分布[左/右] = {counts.tolist()}")

    print("\n保存预处理结果 ...")
    _save("train.npz", train_X, train_y)
    _save("eval.npz", eval_X, eval_y)
    print(f"\n✓ 完成，输出目录：{PROCESSED_DIR}")
    print("  下一步：python3 model/train.py")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

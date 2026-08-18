#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_subject_index.py — 重建 eval.npz 的 subject 索引（P1-⑥）

eval.npz 是 9 个 subject 的 E 会话按顺序拼接的，但没保存 subject 边界。
本脚本复用 preprocess._process_session 重跑 E 会话，输出 subject 索引数组，
与 eval.npz 试次顺序一一对应（供按被试分组统计用）。

用法：python3 data/export_subject_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "data"))
sys.path.insert(0, str(PROJECT))

from preprocess import _load_scipy, _process_session, RAW_DIR, PROCESSED_DIR  # noqa: E402


def main() -> int:
    loadmat, butter, filtfilt, iirnotch = _load_scipy()
    subj_idx = []
    counts = []
    for subj in range(1, 10):
        p = RAW_DIR / f"A{subj:02d}E.mat"
        if not p.exists():
            print(f"  ⚠ 缺失 {p.name}")
            continue
        res = _process_session(p, loadmat, butter, filtfilt, iirnotch)
        if res is None:
            print(f"  ⚠ {p.name} 无有效试次")
            continue
        n = res[0].shape[0]
        subj_idx.extend([subj] * n)
        counts.append((subj, n))
        print(f"  subject {subj}: {n} 试次")

    subj_idx = np.asarray(subj_idx, dtype=np.int64)
    # 校验：和 eval.npz 的试次数一致
    eval_n = np.load(PROCESSED_DIR / "eval.npz")["X"].shape[0]
    print(f"\n总试次: {len(subj_idx)}（eval.npz = {eval_n}）")
    assert len(subj_idx) == eval_n, "试次数不一致！"
    np.save(PROCESSED_DIR / "eval_subject_idx.npy", subj_idx)
    print(f"✓ 已保存 subject 索引 → {PROCESSED_DIR / 'eval_subject_idx.npy'}")
    print(f"  每 subject 试次数: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

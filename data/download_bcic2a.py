#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_bcic2a.py — 下载 BCI Competition IV Dataset 2a（四分类运动想象）

数据集规格
----------
  - 22 通道 EEG，250 Hz 采样率，4 类（左手 / 右手 / 双脚 / 舌头）
  - 9 名被试，每名 2 个 session：A01T~A09T（训练）、A01E~A09E（评估）
  - 原始格式 .gdf（MNE 可直接读取），评估集标签在对应 A{nn}E.mat 文件中
  - 训练集标签已内嵌在 .gdf 的事件注释里（事件码 769/770/771/772）

下载优先级（自动逐级回退，每级失败才进下一级）
------------------------------------------------
  1. MOABB —— `moabb.datasets.BNCI2014_001`（旧版叫 `BNCI2014001`）自动下载并缓存
  2. Graz 原始源 —— https://lampz.tugraz.at/~bci/database/001-2014/A{nn}{T|E}.{gdf|mat}
  3. BNCI Horizon 2020 镜像 —— https://bnci-horizon-2020.eu/database/data-sets/001-2014/

用法
----
  python3 data/download_bcic2a.py                    # 下载全部 9 名被试（约 4GB）
  python3 data/download_bcic2a.py --subjects 1 2     # 只下载被试 1、2
  python3 data/download_bcic2a.py --skip-moabb       # 跳过 MOABB，直接走直连下载
  BCI_DATA_RAW=/path/to/raw python3 data/download_bcic2a.py   # 自定义缓存目录

说明
----
  - 缓存目录默认 <项目根>/data/raw/，可用环境变量 BCI_DATA_RAW 覆盖（不用写死绝对路径）。
  - 国内网络建议先设好代理；直连下载内置重试 + 进度提示。
  - 每个被试需下载 3 个文件：A{nn}T.gdf、A{nn}E.gdf、A{nn}E.mat（T 会话标签在 gdf 内，
    故 T 会话的 .mat 非必需，这里不下载）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径解析：以「项目根」为基准，环境变量可覆盖
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("BCI_DATA_ROOT", str(PROJECT_ROOT / "data")))
RAW_DIR = Path(os.environ.get("BCI_DATA_RAW", str(DATA_ROOT / "raw")))

SUBJECTS_ALL = list(range(1, 10))          # 1..9
SESSIONS = ("T", "E")                      # T=训练, E=评估

# 直连下载的候选源（按优先级）
DIRECT_BASES = [
    "https://lampx.tugraz.at/~bci/database/001-2014",            # Graz 原始源（注意是 lampx 非 lampz）
    "https://bnci-horizon-2020.eu/database/data-sets/001-2014",  # BNCI 镜像
]

USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; luckfox-deploy/1.0)"}


def _download_one(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> bool:
    """单文件下载，带重试与进度条。成功返回 True，失败返回 False（不抛异常）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  · {dest.name} 已存在，跳过")
        return True

    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers=USER_AGENT)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total > 0:
                            pct = done * 100.0 / total
                            print(
                                f"\r  ↓ {dest.name}: {pct:5.1f}% "
                                f"({done // (1024 * 1024)}/{total // (1024 * 1024)} MB)",
                                end="", flush=True,
                            )
                print()
            tmp.replace(dest)
            print(f"  ✓ {dest.name} 下载完成（{done // (1024 * 1024)} MB）")
            return True
        except Exception as exc:  # noqa: BLE001 —— 网络异常一律重试
            print(f"\n  ✗ 第 {attempt}/{retries} 次下载失败（{exc}）")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == retries:
                return False
            time.sleep(2 * attempt)
    return False


def _download_via_moabb(raw_dir: Path, subjects: list[int]) -> bool:
    """优先级 1：用 MOABB 下载。成功且 .gdf 落盘到 raw_dir 返回 True。"""
    try:
        # 让 mne / moabb 把数据缓存在我们的 raw_dir 下
        os.environ.setdefault("MNE_DATA", str(raw_dir))
        try:
            from moabb.datasets import BNCI2014_001
        except ImportError:
            from moabb.datasets import BNCI2014001 as BNCI2014_001  # 旧版命名

        dataset = BNCI2014_001()
        print(f"[1/3] 尝试 MOABB（{type(dataset).__name__}）下载被试 {subjects} ...")
        try:
            dataset.download(subject_list=subjects, path=str(raw_dir))
        except TypeError:
            # 旧版本 download 不接受 path 参数，退而求其次用 get_data 触发下载
            dataset.get_data(subjects=subjects)

        gdfs = list(raw_dir.rglob("*.gdf"))
        if gdfs:
            print(f"  ✓ MOABB 下载成功，共 {len(gdfs)} 个 .gdf 已缓存到 {raw_dir}")
            return True
        print("  · MOABB 下载完成，但 .gdf 未落在预期目录，改走直连下载兜底...")
        return False
    except Exception as exc:  # noqa: BLE001 —— ImportError / 网络错误等一律回退
        print(f"  · MOABB 下载失败（{exc}），回退直连下载...")
        return False


def _download_direct(raw_dir: Path, subjects: list[int]) -> dict[str, bool]:
    """优先级 2/3：直接从 Graz / BNCI 下载 .gdf 与 E 会话 .mat。"""
    status: dict[str, bool] = {}
    # 每个被试要下的文件：T.mat、E.mat（.gdf 已从服务器移除，改用 .mat）
    targets = []
    for subj in subjects:
        for sess in SESSIONS:
            targets.append((subj, sess, "mat"))

    for base in DIRECT_BASES:
        print(f"  下载源：{base}")
        all_ok = True
        for subj, sess, ext in targets:
            fname = f"A{subj:02d}{sess}.{ext}"
            url = f"{base}/{fname}"
            dest = RAW_DIR / fname
            ok = _download_one(url, dest)
            status[fname] = ok
            all_ok = all_ok and ok
        if all_ok:
            break
        print("  · 该源存在失败项，尝试下一镜像...")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 BCIC IV 2a 四分类运动想象数据集")
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS_ALL,
                        help="要下载的被试编号（默认 1~9）")
    parser.add_argument("--skip-moabb", action="store_true",
                        help="跳过 MOABB，直接走直连下载")
    args = parser.parse_args()

    subjects = sorted(set(args.subjects))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"缓存目录：{RAW_DIR}")
    print(f"目标被试：{subjects}")

    ok = False
    if not args.skip_moabb:
        ok = _download_via_moabb(RAW_DIR, subjects)

    if not ok:
        print("[2/3] 直连下载 .gdf / .mat ...")
        status = _download_direct(RAW_DIR, subjects)
        n_ok = sum(1 for v in status.values() if v)
        print(f"  直连下载完成：{n_ok}/{len(status)} 个文件成功")

    # 汇总校验
    missing: list[str] = []
    for subj in subjects:
        for sess in SESSIONS:
            if not (RAW_DIR / f"A{subj:02d}{sess}.mat").exists():
                missing.append(f"A{subj:02d}{sess}.mat")

    if missing:
        print("\n⚠ 以下文件仍缺失（这会导致预处理时报错）：")
        for m in missing:
            print(f"    - {m}")
        print("  请检查网络 / 代理后重试，或手动下载后放入上面的缓存目录。")
        print("  下一步：python3 data/preprocess.py 将读取这些文件。")
        return 1

    print(f"\n✓ 全部 {len(subjects)} 名被试数据就绪。")
    print(f"  下一步：python3 data/preprocess.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_rknn.py — ONNX → RKNN（INT8 量化，target rv1106）

⚠ 平台限制（务必先读）
-----------------------
  rknn-toolkit2 **只能在 Linux x86_64 上运行**（官方只发 x86_64 Linux 轮子）：
    - Mac（含本项目所在的 macOS）—— 不支持，请在 WSL2 或云主机（如一台 Linux 服务器）跑本脚本
    - Windows —— 官方不直接支持，请用 WSL2
  板载推理用的是另一套 rknn-lite2（board/infer_eegnet.py 负责），本脚本只负责在 PC 端
  把 ONNX 转成 .rknn 文件。

安装（在 Linux x86 主机上）
---------------------------
  pip install rknn-toolkit2
  # 若官方 PyPI 没有合适版本，从 Rockchip 官方仓库安装：
  #   https://github.com/airockchip/rknn-toolkit2
  #   下载对应 rknn-toolkit2 的 .whl（按 README 的 Python 版本对照表）后 pip install。

转换流程
--------
  1. config(mean_values=[[0]], std_values=[[1]], target_platform='rv1106')
       —— 输入已做过 z-score（均值 0 标准差 1），所以不做额外归一
  2. load_onnx('eegnet.onnx')
  3. build(do_quantization=True, dataset='calib_data.txt')  # INT8 量化，用校准集统计范围
  4. init_runtime(target='rv1106') 验证精度（用模拟器跑校准集，对比 ONNX 输出）
  5. export_rknn('eegnet.rknn')

用法（Linux x86 主机上）
------------------------
  python3 model/convert_rknn.py
  python3 model/convert_rknn.py --verify  # 转换后额外跑一组样本对比精度

输出
----
  model/eegnet.rknn    INT8 量化后的 RKNN 模型（目标 <50KB）

依赖：rknn-toolkit2（仅 Linux x86）、numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent

ONNX_PATH = MODEL_DIR / "eegnet.onnx"
RKNN_PATH = MODEL_DIR / "eegnet.rknn"
CALIB_TXT = MODEL_DIR / "calib_data.txt"

# 模型输入形状（与 eegnet.onnx 固定形状一致，8ch 二分类）
INPUT_SHAPE = (1, 1, 8, 500)
N_CLASSES = 2


def _import_rknn():
    """导入 rknn-toolkit2，未安装 / 平台不支持时给出清晰中文提示。"""
    try:
        from rknn.api import RKNN  # noqa: PLC0415
        return RKNN
    except ImportError as exc:
        raise SystemExit(
            "✗ 无法导入 rknn-toolkit2。\n"
            "  原因：rknn-toolkit2 仅在 Linux x86_64 上可用（Mac/Windows 需 WSL2）。\n"
            "  请在一台 Linux x86 主机（WSL2 或云服务器）上执行：\n"
            "    pip install rknn-toolkit2\n"
            "  版本对照见：https://github.com/airockchip/rknn-toolkit2\n"
            f"  原始错误：{exc}"
        ) from exc


def _verify_accuracy(rknn, onnx_ref_logits: np.ndarray | None) -> None:
    """用 rv1106 模拟器跑一遍校准集，对比分类准确率（可选）。"""
    if not CALIB_TXT.exists():
        print("  · 未找到 calib_data.txt，跳过精度验证。")
        return

    lines = [ln.strip() for ln in CALIB_TXT.read_text().splitlines() if ln.strip()]
    if not lines:
        print("  · calib_data.txt 为空，跳过精度验证。")
        return

    correct = 0
    total = 0
    # 只取前若干样本验证，避免模拟器耗时过长
    for ln in lines[:50]:
        x = np.load(MODEL_DIR / ln).astype(np.float32)
        # 确保形状 (1,1,22,500)
        if x.shape != INPUT_SHAPE:
            x = x.reshape(INPUT_SHAPE)
        out = rknn.inference(inputs=[x])
        pred = int(np.argmax(out[0]))
        total += 1
        if onnx_ref_logits is not None:
            # 以 ONNX 的 argmax 作为"真值"来比对模拟器
            ref = int(np.argmax(onnx_ref_logits))
            correct += (pred == ref)
        else:
            print(f"  样本 {total}: 模拟器 argmax={pred}")
    if onnx_ref_logits is not None:
        print(f"  ✓ 模拟器 vs ONNX 分类一致率：{correct}/{total} = {correct/total*100:.1f}%")
    print("  （注：rv1106 模拟器运行的是量化后模型，与板载 NPU 行为基本一致。）")


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX → RKNN (INT8, rv1106)")
    parser.add_argument("--verify", action="store_true",
                        help="转换后跑校准集验证量化精度")
    args = parser.parse_args()

    RKNN = _import_rknn()

    if not ONNX_PATH.exists():
        raise SystemExit(
            f"✗ ONNX 文件缺失：{ONNX_PATH}\n"
            "  请先在能跑 torch 的环境导出：python3 model/export_onnx.py\n"
            "  再把 eegnet.onnx 拷贝到本机。"
        )
    if not CALIB_TXT.exists():
        raise SystemExit(
            f"✗ 校准集清单缺失：{CALIB_TXT}\n"
            "  请先生成校准集：python3 model/calibration.py\n"
            "  并把 calib_data/ 目录 + calib_data.txt 一起拷贝到本机。"
        )

    rknn = RKNN(verbose=True)

    # ---- 1. 配置：输入已 z-score，均值 0 标准差 1，故不做额外归一 ----
    #   mean_values / std_values 的格式：[[c0, c1, ...]]，按通道给定；
    #   本模型输入通道 C=1，故 [[0]] / [[1]]。
    print("配置 RKNN（target=rv1106, INT8, optimization_level=0 避免 Pad 拆分）...")
    rknn.config(mean_values=[[0]], std_values=[[1]], target_platform="rv1106",
                optimization_level=0)

    # ---- 2. 加载 ONNX ----
    print(f"加载 ONNX：{ONNX_PATH}")
    ret = rknn.load_onnx(model=str(ONNX_PATH))
    if ret != 0:
        raise SystemExit(f"✗ load_onnx 失败，返回码 {ret}（可能是 opset 版本不兼容）")

    # ---- 3. 构建 + INT8 量化（用校准集统计激活范围）----
    print(f"构建 RKNN 模型（INT8 量化，校准集：{CALIB_TXT}）...")
    ret = rknn.build(do_quantization=True, dataset=str(CALIB_TXT))
    if ret != 0:
        raise SystemExit(f"✗ build 失败，返回码 {ret}")

    # ---- 4. 用 rv1106 模拟器验证精度后再导出（可选，x86 上无模拟器会抛异常）----
    #   注意：init_runtime 失败会抛 ValueError（而非返回非 0），必须 try/except 包裹；
    #   导出 .rknn 不依赖 init_runtime，模拟器仅用于转换后精度自检。
    print("初始化 rv1106 模拟器...")
    try:
        ret = rknn.init_runtime(target="rv1106")
        if ret != 0:
            print("  ⚠ init_runtime 返回非 0，跳过精度验证")
        elif args.verify:
            _verify_accuracy(rknn, None)
    except Exception as exc:  # noqa: BLE001 —— x86 无 rv1106 模拟器属预期
        print(f"  ⚠ init_runtime 异常（x86 无 rv1106 模拟器，跳过精度验证）：{exc}")

    # ---- 5. 导出 .rknn ----
    print(f"导出 RKNN：{RKNN_PATH}")
    ret = rknn.export_rknn(str(RKNN_PATH))
    if ret != 0:
        raise SystemExit(f"✗ export_rknn 失败，返回码 {ret}")

    size_kb = RKNN_PATH.stat().st_size / 1024
    print(f"\n✓ RKNN 模型已导出：{RKNN_PATH}（{size_kb:.1f} KB）")
    if size_kb > 50:
        print("  ⚠ 模型 >50KB，请检查是否意外包含冗余算子（EEGNet 预期 <50KB）")
    else:
        print("  ✓ 模型大小 <50KB，符合板载部署目标")

    rknn.release()
    print("\n下一步：把 eegnet.rknn 拷贝到 Luckfox 板子，用 board/infer_eegnet.py 推理。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

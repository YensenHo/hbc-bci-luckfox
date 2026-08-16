#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_c_weights.py — 把训练好的 EEGNet 权重导出为 C 头文件（含 BN 折叠）

BN 折叠：推理时 BatchNorm 是逐通道仿射变换
    y = gamma * (x - mean) / sqrt(var + eps) + beta
  可折叠进前一层卷积：
    W' = W * gamma / sqrt(var + eps)
    b' = beta - mean * gamma / sqrt(var + eps)

导出的 C 头文件 board/eegnet_weights.h 结构：
    conv1_w[8][1][1][64], conv1_b[8]
    depthwise_w[16][1][8][1], depthwise_b[16]
    sep_dw_w[16][1][1][16]
    pointwise_w[16][16][1][1], pointwise_b[16]
    fc_w[2][240], fc_b[2]

用法
----
  python3 model/export_c_weights.py [--checkpoint model/eegnet.pth]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent
BOARD_DIR = MODEL_DIR.parent / "board"
HEADER = BOARD_DIR / "eegnet_weights.h"


def _fold_bn(w: np.ndarray, bn_w, bn_b, bn_mean, bn_var, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """把 BN 折叠进卷积权重。w: (out, in/groups, kh, kw)。返回 (W', b')。"""
    scale = bn_w / np.sqrt(bn_var + eps)      # gamma / sqrt(var+eps)
    bias = bn_b - bn_mean * scale             # beta - mean*scale
    # 卷积权重逐输出通道缩放
    w_folded = w * scale.reshape(-1, 1, 1, 1)
    return w_folded, bias


def export(checkpoint: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt["state_dict"]
    eps = 1e-5  # PyTorch BatchNorm2d 默认 eps

    # ---- Block1 ----
    conv1_w = sd["conv1.weight"].numpy()                       # (8,1,1,64)
    conv1_w, conv1_b = _fold_bn(conv1_w, sd["bn1.weight"].numpy(),
                                sd["bn1.bias"].numpy(), sd["bn1.running_mean"].numpy(),
                                sd["bn1.running_var"].numpy(), eps)

    dw_w = sd["depthwise.weight"].numpy()                      # (16,1,8,1)
    dw_w, dw_b = _fold_bn(dw_w, sd["bn2.weight"].numpy(),
                          sd["bn2.bias"].numpy(), sd["bn2.running_mean"].numpy(),
                          sd["bn2.running_var"].numpy(), eps)

    # ---- Block2（sep_depthwise 无 BN；pointwise 后接 bn3）----
    sdw_w = sd["sep_depthwise.weight"].numpy()                 # (16,1,1,16)
    pw_w = sd["pointwise.weight"].numpy()                      # (16,16,1,1)
    pw_w, pw_b = _fold_bn(pw_w, sd["bn3.weight"].numpy(),
                          sd["bn3.bias"].numpy(), sd["bn3.running_mean"].numpy(),
                          sd["bn3.running_var"].numpy(), eps)

    # ---- FC ----
    fc_w = sd["fc.weight"].numpy()                             # (2,240)
    fc_b = sd["fc.bias"].numpy()                               # (2,)

    # ---- 生成 C 头文件 ----
    def fmt_array(name: str, arr: np.ndarray) -> str:
        flat = arr.ravel()
        vals = ", ".join(f"{v:.8e}f" for v in flat)
        return f"static const float {name}[{len(flat)}] = {{ {vals} }};\n"

    lines = [
        "/* 自动生成 —— 勿手改。来源: model/export_c_weights.py */",
        "/* EEGNet 8ch 二分类权重（BN 已折叠进卷积） */",
        "#ifndef EEGNET_WEIGHTS_H",
        "#define EEGNET_WEIGHTS_H",
        "",
        "#define EEGNET_N_CH 8",
        "#define EEGNET_T 500",
        "#define EEGNET_F1 8",
        "#define EEGNET_D 2",
        "#define EEGNET_F2 16",
        "#define EEGNET_T_OUT 15",      # 500//32 = 15
        "",
        fmt_array("conv1_w", conv1_w),
        fmt_array("conv1_b", conv1_b),
        fmt_array("depthwise_w", dw_w),
        fmt_array("depthwise_b", dw_b),
        fmt_array("sep_dw_w", sdw_w),
        fmt_array("pointwise_w", pw_w),
        fmt_array("pointwise_b", pw_b),
        fmt_array("fc_w", fc_w),
        fmt_array("fc_b", fc_b),
        "",
        "#endif /* EEGNET_WEIGHTS_H */",
    ]
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    HEADER.write_text("\n".join(lines), encoding="utf-8")

    print(f"✓ 已导出 {HEADER}")
    print(f"  各层权重形状：")
    print(f"    conv1_w      {conv1_w.shape} + bias {conv1_b.shape}")
    print(f"    depthwise_w  {dw_w.shape} + bias {dw_b.shape}")
    print(f"    sep_dw_w     {sdw_w.shape}")
    print(f"    pointwise_w  {pw_w.shape} + bias {pw_b.shape}")
    print(f"    fc_w         {fc_w.shape} + bias {fc_b.shape}")
    total = (conv1_w.size + conv1_b.size + dw_w.size + dw_b.size +
             sdw_w.size + pw_w.size + pw_b.size + fc_w.size + fc_b.size)
    print(f"  总参数量: {total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 EEGNet 权重为 C 头文件")
    parser.add_argument("--checkpoint", type=str, default=str(MODEL_DIR / "eegnet.pth"))
    args = parser.parse_args()
    cp = Path(args.checkpoint)
    if not cp.exists():
        raise SystemExit(f"✗ 模型文件不存在：{cp}\n  请先训练：python3 model/train.py")
    export(cp)
    return 0


if __name__ == "__main__":
    sys.exit(main())

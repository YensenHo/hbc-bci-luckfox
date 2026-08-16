#!/usr/bin/env python3
"""
infer_eegnet.py — 板载 NPU 推理（阶段 2）

⚠️ 重大修正（Codex 调研 2026-08 确认）：RV1106/RV1103 **不支持 rknn-lite2（Python）**！
   Luckfox Pico 系列板载推理只能用 **C API**（librknnmrt.so + rknn_api.h）：
   rknn_init / rknn_query / rknn_create_mem / rknn_run。
   且默认 uClibc Buildroot 镜像不带 librknnmrt.so，需随 C demo 交叉编译一起打包。

   因此本文件只作占位说明，真正推理要用 C 程序（见 model/convert_rknn.py 产出的 .rknn，
   配合 Luckfox 官方 rknn_model_zoo / luckfox_pico_rknn_example 的 C demo，
   用 SDK 工具链 arm-rockchip830-linux-uclibcgnueabihf-gcc 交叉编译 + 打包 librknnmrt.so）。

前置：PC 端（Linux x86/WSL2）用 rknn-toolkit2 把 EEGNet 转成 eegnet.rknn（target rv1106）。

预期：8ch EEGNet 单次推理 <5ms（0.5 TOPS 跑 ~2500 参数模型绰绰有余）。
"""
import sys


def main():
    print("RV1106 不支持 rknn-lite2 Python API，请使用 C API（librknnmrt.so）。")
    print("参考：https://wiki.luckfox.com/Luckfox-Pico-Ultra/RKNN/")
    print("流程：")
    print("  1) Linux x86/WSL2: rknn-toolkit2 转 eegnet.rknn（target rv1106, INT8）")
    print("  2) SDK 工具链 arm-rockchip830-linux-uclibcgnueabihf-gcc 交叉编译 C demo")
    print("  3) 打包 librknnmrt.so 随 demo 一起 adb push")
    print("  4) 板子上跑 demo（跑前 RkLunch-stop.sh 释放默认 rkipc 进程）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

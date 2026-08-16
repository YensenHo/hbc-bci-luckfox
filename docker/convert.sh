#!/bin/bash
# 一键本地完成 NPU 部署两步（转 .rknn + 交叉编译），替代 GitHub Actions
# 用法：./docker/convert.sh
# 产物：model/eegnet.rknn、board/eegnet_npu
set -euo pipefail

# 项目根目录
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="hbc-bci-npu:latest"

# 1. 构建镜像（首次需下载 torch+rknn-toolkit2，约 2GB，几分钟）
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "=== 构建镜像（首次，需下载依赖，约 2GB）==="
    docker build --platform linux/amd64 -t "$IMAGE" -f "$ROOT/docker/Dockerfile" "$ROOT"
else
    echo "=== 镜像已存在，跳过构建 ==="
fi

# 2. 挂载项目，跑转换 + 编译
echo "=== 挂载 $ROOT → /workspace，执行转换 + 编译 ==="
docker run --rm --platform linux/amd64 -v "$ROOT:/workspace" "$IMAGE" bash -c '
    cd /workspace
    echo ""
    echo "=== [1/2] 转 .rknn（INT8, rv1106）==="
    python3 model/convert_rknn.py

    echo ""
    echo "=== [2/2] 交叉编译 NPU 推理程序 ==="
    arm-rockchip830-linux-uclibcgnueabihf-gcc -O2 -I board \
        board/infer_eegnet_rknn.c board/eegnet_infer.c \
        -L board -lrknnmrt -lm -Wl,-rpath,/oem/usr/lib \
        -o board/eegnet_npu
    file board/eegnet_npu
'

echo ""
echo "=== 完成 ==="
echo "  model/eegnet.rknn  ← INT8 量化模型"
echo "  board/eegnet_npu   ← ARM 二进制（推板子用）"
echo ""
echo "推板子："
echo "  adb push model/eegnet.rknn /root/eegnet.rknn"
echo "  adb push board/eegnet_npu /root/eegnet_npu"
echo "  adb shell \"chmod +x /root/eegnet_npu && /root/eegnet_npu /root/eegnet.rknn\""

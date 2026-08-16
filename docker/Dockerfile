# hbc-bci NPU 本地构建镜像（x86_64 Linux）
#
# 用途：在 Apple Silicon Mac 上本地完成两件 Mac 做不了的事：
#   1. rknn-toolkit2 转 .rknn（rknn-toolkit2 只有 x86_64 wheel）
#   2. Luckfox uClibc 工具链交叉编译 NPU 推理程序
#
# 构建（首次需下载 torch+rknn-toolkit2，约 2GB，几分钟）：
#   docker build --platform linux/amd64 -t hbc-bci-npu -f docker/Dockerfile .
# 或直接跑一键脚本：./docker/convert.sh

# 注意：x86_64 platform 由构建命令指定（docker build --platform linux/amd64），
# 不写在 FROM 里（避免 BuildKit 常量警告）。
FROM ubuntu:22.04

# 基础工具 + Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev git file ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# rknn-toolkit2（x86_64 wheel）+ onnx 1.14.1（rknn-toolkit2 2.3.2 内部用 onnx.mapping，
# onnx>=1.16 已移除该 API，必须降级）
# 清华 pip 镜像（GFW 下快）
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        rknn-toolkit2==2.3.2 \
    && pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "onnx==1.14.1"

# Luckfox uClibc 工具链（sparse clone，只取工具链目录）
RUN git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/LuckfoxTECH/luckfox-pico.git /opt/luckfox \
    && cd /opt/luckfox \
    && git sparse-checkout set tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf

ENV PATH="/opt/luckfox/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin:${PATH}"

WORKDIR /workspace

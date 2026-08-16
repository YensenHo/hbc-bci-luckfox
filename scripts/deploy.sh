#!/usr/bin/env bash
# ============================================================
# deploy.sh — 一键部署板载脚本到 Luckfox Pico Ultra
#
# 用法：
#   bash scripts/deploy.sh              # 默认 172.32.0.93
#   LUCKFOX_HOST=172.32.0.100 bash scripts/deploy.sh
# ============================================================
set -euo pipefail

HOST="${LUCKFOX_HOST:-172.32.0.93}"
USER="root"
REMOTE="/root"
# 脚本所在目录的上一级 = 项目根
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOARD_DIR="$ROOT/board"

echo "==> 目标板子: ${USER}@${HOST}  (远程目录 ${REMOTE})"

# ---- 上传板载脚本 ----
FILES=(
  "lif_sim.py"
  "lif_net.c"
  "infer_eegnet.py"
  "eeg_capture.py"
  "hbc_tx.py"
)

for f in "${FILES[@]}"; do
  echo "==> 上传 ${f}"
  scp "${BOARD_DIR}/${f}" "${USER}@${HOST}:${REMOTE}/"
done

# ---- 编译 C 版 LIF ----
echo "==> 编译 lif_net.c"
ssh "${USER}@${HOST}" "cd ${REMOTE} && gcc -O2 lif_net.c -o lif_net && echo 'lif_net 编译完成'"

# ---- 运行验证 LIF ----
echo "==> 运行 LIF 验证"
ssh "${USER}@${HOST}" "cd ${REMOTE} && python3 lif_sim.py && ./lif_net"

echo ""
echo "部署完成。后续步骤："
echo "  1) 上传 eegnet.rknn:  scp model/eegnet.rknn ${USER}@${HOST}:${REMOTE}/"
echo "  2) 板载 NPU 推理:     ssh ${USER}@${HOST} 'python3 /root/infer_eegnet.py'"

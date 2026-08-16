#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — 训练 EEGNet 四分类运动想象模型

数据协议
--------
  - 训练集 = data/processed/train.npz（全部 A{nn}T 会话，标签来自 .gdf 事件码）
  - 评估集 = data/processed/eval.npz （全部 A{nn}E 会话，标签来自 .mat）
  - 评估集既用于每 epoch 的验证，也作为最终测试集（本项目无独立 hold-out 集）。

训练配置（严格按规格）
----------------------
  - 优化器 Adam lr=0.001，损失 CrossEntropyLoss，batch=16，epoch=50
  - 早停 patience=10（按验证集准确率）
  - 每个 epoch 打印 train/val 准确率与 loss，目标 >70%

输出
----
  model/eegnet.pth          最优模型权重（state_dict）
  model/accuracy_report.txt 准确率报告（含每 epoch 历史与超参数）

用法
----
  python3 model/train.py
  python3 model/train.py --epochs 50 --batch-size 16 --lr 0.001
  python3 model/train.py --no-cuda   # 强制 CPU（Mac MPS / CUDA 自动选择）

依赖：torch, numpy（数据已由 preprocess.py 预处理好）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径与导入
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("BCI_DATA_ROOT", str(PROJECT_ROOT / "data")))
PROCESSED_DIR = Path(os.environ.get("BCI_DATA_PROCESSED", str(DATA_ROOT / "processed")))
MODEL_DIR = Path(__file__).resolve().parent

CHECKPOINT = MODEL_DIR / "eegnet.pth"
REPORT_TXT = MODEL_DIR / "accuracy_report.txt"
REPORT_JSON = MODEL_DIR / "accuracy_report.json"

sys.path.insert(0, str(MODEL_DIR))   # 保证 from eegnet import EEGNet 可用
sys.path.insert(0, str(DATA_ROOT))   # 保证 from dataset import EEGDataset 可用


def _pick_device(no_cuda: bool):
    """选择训练设备：CUDA → MPS（Mac）→ CPU。"""
    import torch  # noqa: PLC0415
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 EEGNet（BCIC IV 2a 四分类）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    from eegnet import EEGNet  # noqa: PLC0415
    from dataset import EEGDataset  # noqa: PLC0415

    # ---- 数据校验 ----
    train_npz = PROCESSED_DIR / "train.npz"
    eval_npz = PROCESSED_DIR / "eval.npz"
    if not train_npz.exists():
        raise SystemExit(
            f"✗ 训练集缺失：{train_npz}\n"
            "  请先下载并预处理数据：\n"
            "    python3 data/download_bcic2a.py\n"
            "    python3 data/preprocess.py"
        )
    if not eval_npz.exists():
        raise SystemExit(
            f"✗ 评估集缺失：{eval_npz}\n  请先运行：python3 data/preprocess.py"
        )

    # ---- 随机种子 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 设备与数据加载 ----
    device = _pick_device(args.no_cuda)
    train_ds = EEGDataset(train_npz)
    eval_ds = EEGDataset(eval_npz)
    print(f"设备     : {device}")
    print(f"训练集   : {train_ds.shape}（{len(train_ds)} 试次）")
    print(f"评估集   : {eval_ds.shape}（{len(eval_ds)} 试次）")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)

    # ---- 模型 ----
    model = EEGNet(n_channels=8, n_classes=2, input_length=500,
                   F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)
    model = model.to(device)
    print(f"参数量   : {model.num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 训练循环 ----
    history: list[dict] = []
    best_acc = 0.0
    best_epoch = -1
    best_state = None
    patience_counter = 0

    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)
        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        # 验证
        model.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_n += xb.size(0)
        val_acc = val_correct / val_n
        dt = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
        })
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.2f}% "
              f"val_acc={val_acc*100:5.2f}% | {dt:.1f}s")

        # 早停 + 保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n早停：验证集准确率连续 {args.patience} 个 epoch 未提升。")
                break

    # ---- 保存最优模型与报告 ----
    if best_state is None:
        raise SystemExit("✗ 训练未产生任何有效模型（可能数据异常）。")

    torch.save({"state_dict": best_state,
                "epoch": best_epoch,
                "val_acc": best_acc,
                "model_args": dict(n_channels=8, n_classes=2, input_length=500,
                                   F1=8, D=2, F2=16, kernel_length=63, dropout=0.5)},
               CHECKPOINT)
    print(f"\n✓ 最优模型已保存：{CHECKPOINT}（epoch {best_epoch}，val_acc={best_acc*100:.2f}%）")

    target = 0.75
    lines = [
        "EEGNet 训练报告 — BCIC IV 2a 8ch 二分类（左/右手）",
        "=" * 50,
        f"最佳验证准确率 : {best_acc*100:.2f}%（epoch {best_epoch}）",
        f"目标 (>75%)     : {'达成 ✓' if best_acc >= target else '未达成 ✗'}",
        f"训练集样本数    : {len(train_ds)}",
        f"评估集样本数    : {len(eval_ds)}",
        f"参数量          : {model.num_params():,}",
        f"设备            : {device}",
        f"超参数          : lr={args.lr}, batch={args.batch_size}, "
        f"epochs={args.epochs}, patience={args.patience}, seed={args.seed}",
        "-" * 50,
        "逐 epoch 记录 (epoch, train_loss, train_acc, val_acc):",
    ]
    for h in history:
        lines.append(f"  {h['epoch']:3d}  {h['train_loss']:.6f}  "
                     f"{h['train_acc']*100:5.2f}%  {h['val_acc']*100:5.2f}%")
    report = "\n".join(lines)
    REPORT_TXT.write_text(report + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(history, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"  报告已写入：{REPORT_TXT}")
    print(f"  报告已写入：{REPORT_JSON}")
    print("\n下一步：python3 model/export_onnx.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_c_infer.py — 验证 C 版 EEGNet 推理与 PyTorch 输出一致

方法：取 eval.npz 前 N 个真实脑电样本，分别跑 PyTorch（eval 模式）和
      编译后的 C 版，比较 logits 与预测类别。这是 C 移植的正确性门槛。

用法
----
  python3 model/verify_c_infer.py [--n 20]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT / "model"
BOARD_DIR = PROJECT / "board"
PROCESSED = PROJECT / "data" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    N = args.n

    sys.path.insert(0, str(MODEL_DIR))
    from eegnet import EEGNet  # noqa: PLC0415

    # ---- PyTorch 推理 ----
    model = EEGNet(n_channels=8, n_classes=2)
    ckpt = torch.load(MODEL_DIR / "eegnet.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    d = np.load(PROCESSED / "eval.npz")
    Xs = d["X"][:N].astype(np.float32)          # (N,1,8,500)
    with torch.no_grad():
        logits_py = model(torch.from_numpy(Xs)).numpy()
    pred_py = logits_py.argmax(1)

    # ---- C 推理（编译 + 运行）----
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "input.bin").write_bytes(Xs.tobytes())
        test_c = tmp / "test.c"
        test_c.write_text(
            '#include <stdio.h>\n'
            '#include <stdlib.h>\n'
            'extern int eegnet_infer(const float *x, float *logits);\n'
            'int main(void) {\n'
            '    FILE *f = fopen("' + str(tmp / "input.bin") + '", "rb");\n'
            f'    for (int n = 0; n < {N}; n++) {{\n'
            '        float x[8*500];\n'
            '        fread(x, sizeof(float), 8*500, f);\n'
            '        float logits[2];\n'
            '        eegnet_infer(x, logits);\n'
            '        printf("%.6f %.6f\\n", logits[0], logits[1]);\n'
            '    }\n'
            '    fclose(f);\n'
            '    return 0;\n'
            '}\n'
        )
        bin_path = tmp / "eegnet_test"
        r = subprocess.run(
            ["cc", "-O2", str(BOARD_DIR / "eegnet_infer.c"), str(test_c),
             "-o", str(bin_path), "-lm"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("✗ C 编译失败：")
            print(r.stderr)
            return 1
        out = subprocess.run([str(bin_path)], capture_output=True, text=True).stdout
        logits_c = np.array([[float(v) for v in line.split()] for line in out.strip().split("\n")])
        pred_c = logits_c.argmax(1)

    # ---- 对比 ----
    max_diff = float(np.abs(logits_py - logits_c).max())
    n_match = int((pred_py == pred_c).sum())
    print(f"logits 最大绝对误差 : {max_diff:.2e}")
    print(f"预测一致率         : {n_match}/{N} ({n_match/N*100:.0f}%)")
    if max_diff < 1e-3 and n_match == N:
        print("✓ C 推理与 PyTorch 一致（BN 折叠正确）")
        _save_consistency_figure(logits_py, logits_c, N, max_diff, MODEL_DIR)
        return 0
    print("✗ C 推理与 PyTorch 不一致，需排查移植")
    return 1


def _save_consistency_figure(logits_py, logits_c, n, max_diff, model_dir: Path) -> None:
    """验证时把 C vs PyTorch 的 logits 逐点对比画成散点图（真实验证产物）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor="#1B2A4A")
        ax.set_facecolor("#243555")
        ax.scatter(logits_py[:, 0], logits_c[:, 0], color="#00B4D8", s=50, label="logits[0]")
        ax.scatter(logits_py[:, 1], logits_c[:, 1], color="#E76F51", s=50, label="logits[1]")
        lo = min(logits_py.min(), logits_c.min()) - 0.2
        hi = max(logits_py.max(), logits_c.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "--", color="#8FA3BF", lw=1.5, label="y=x（完全一致）")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("PyTorch logits", color="white", fontsize=12)
        ax.set_ylabel("C 推理 logits", color="white", fontsize=12)
        ax.set_title(f"C 移植 vs PyTorch：{n} 个真实试次 logits 逐点重合（误差 {max_diff:.2e}）",
                     color="white", fontsize=14, weight="bold", pad=12)
        ax.tick_params(colors="white", labelsize=11)
        for sp in ax.spines.values():
            sp.set_color("#8FA3BF")
        ax.legend(facecolor="#243555", edgecolor="#8FA3BF", labelcolor="white")
        fig.tight_layout()
        fig_path = model_dir / "verify_consistency.png"
        fig.savefig(fig_path, facecolor="#1B2A4A", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  验证一致性图已保存：{fig_path}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 一致性图保存失败（不影响验证）：{e}")


if __name__ == "__main__":
    sys.exit(main())

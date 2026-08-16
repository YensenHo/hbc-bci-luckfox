#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eegnet.py — EEGNet 模型定义（Lawhern et al. 2018, J. Neural Eng.）

针对 BCIC IV 2a 四分类运动想象的标准配置：
  - F1=8, D=2, F2=16, kernel_length=64（250Hz 下 0.25s）, dropout=0.5
  - 输入 (1, 22, 500)（C=1 图像通道, H=22 EEG 电极, W=500 时间点）
  - 输出 4 类（左手/右手/双脚/舌头）
  - 参数量约 2.5k，INT8 量化后 <50KB，适合 Luckfox RV1106 NPU

结构（对照原文）：
  Block1: Conv2d(1,F1,(1,64)) → BatchNorm → DepthwiseConv2d(F1,D*F1,(22,1),groups=F1)
          → BatchNorm → ELU → AvgPool(1,4) → Dropout
  Block2: SeparableConv2d(D*F1,F2,(1,16)) [depthwise + pointwise]
          → BatchNorm → ELU → AvgPool(1,8) → Dropout
  分类头: Flatten → Linear(F2 * (T//32), 4)

实现说明：偶数长度卷积核的 'same' 填充是非对称的（左 floor((k-1)/2)、右 k//2），
PyTorch 的 nn.Conv2d 只支持对称填充，因此用 ZeroPad2d + Conv2d(padding=0) 显式实现，
保证时序维度严格保持 T→T//4→T//32，且 ONNX/RKNN 兼容性最好。

用法
----
  前向自检（打印输出形状，应为 torch.Size([1, 4])）：
    python3 model/eegnet.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet 四分类运动想象模型。"""

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 2,
        input_length: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_length = input_length
        self.F1, self.D, self.F2 = F1, D, F2
        self.kernel_length = kernel_length

        # ---- Block 1：时间卷积 + 空间 depthwise 卷积 ----
        # 'same' 填充（偶数核 64：左 31、右 32）
        pl1, pr1 = kernel_length // 2 - 1, kernel_length // 2
        self.pad1 = nn.ZeroPad2d((pl1, pr1, 0, 0))
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # depthwise：每组一个 (n_channels, 1) 卷积核，把 22 个电极压成 1
        self.depthwise = nn.Conv2d(
            F1, D * F1, (n_channels, 1), groups=F1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # ---- Block 2：separable 卷积（depthwise (1,16) + pointwise (1,1)）----
        pl2, pr2 = 16 // 2 - 1, 16 // 2   # 7, 8
        self.pad2 = nn.ZeroPad2d((pl2, pr2, 0, 0))
        self.sep_depthwise = nn.Conv2d(
            D * F1, D * F1, (1, 16), groups=D * F1, padding=0, bias=False
        )
        self.pointwise = nn.Conv2d(D * F1, F2, (1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # ---- 分类头：T → T//4 → T//32 ----
        self.temporal_out = input_length // 32
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * self.temporal_out, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, n_classes) 未归一化 logits。"""
        # Block 1
        x = self.pad1(x)
        x = self.conv1(x)          # (B, F1, C, T)
        x = self.bn1(x)
        x = self.depthwise(x)      # (B, D*F1, 1, T)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)          # (B, D*F1, 1, T//4)
        x = self.dropout1(x)

        # Block 2
        x = self.pad2(x)
        x = self.sep_depthwise(x)  # (B, D*F1, 1, T//4)
        x = self.pointwise(x)      # (B, F2, 1, T//4)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)          # (B, F2, 1, T//32)
        x = self.dropout2(x)

        # 分类头
        x = self.flatten(x)        # (B, F2 * T//32)
        x = self.fc(x)             # (B, n_classes)
        return x

    def num_params(self) -> int:
        """可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # ---- 前向自检：dummy input (1,1,8,500) → 输出 (1,2) ----
    model = EEGNet()
    model.eval()
    dummy = torch.randn(1, 1, 8, 500, dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy)
    print(f"输入形状 : {tuple(dummy.shape)}")
    print(f"输出形状 : {tuple(out.shape)}   （应为 (1, 2)）")
    print(f"参数量   : {model.num_params():,} 个")
    assert tuple(out.shape) == (1, 2), f"输出形状错误：{tuple(out.shape)}"
    print("✓ 前向测试通过")

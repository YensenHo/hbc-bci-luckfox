#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eegnet.py — EEGNet 模型定义（Lawhern et al. 2018, J. Neural Eng.）

针对 BCIC IV 2a 四分类运动想象的标准配置：
  - F1=8, D=2, F2=16, kernel_length=63（250Hz 下 0.252s，奇数核适配 NPU）, dropout=0.5
  - 输入 (1, 22, 500)（C=1 图像通道, H=22 EEG 电极, W=500 时间点）
  - 输出 4 类（左手/右手/双脚/舌头）
  - 参数量约 2.5k，INT8 量化后 <50KB，适合 Luckfox RV1106 NPU

结构（对照原文）：
  Block1: Conv2d(1,F1,(1,64)) → BatchNorm → DepthwiseConv2d(F1,D*F1,(22,1),groups=F1)
          → BatchNorm → ELU → AvgPool(1,4) → Dropout
  Block2: SeparableConv2d(D*F1,F2,(1,16)) [depthwise + pointwise]
          → BatchNorm → ELU → AvgPool(1,8) → Dropout
  分类头: Flatten → Linear(F2 * (T//32), 4)

实现说明：为适配 RV1106 NPU（不支持 Pad 算子），改用**奇数核 + 对称 'same' padding**
（conv1 核 63、sep_depthwise 核 15），padding 作为 Conv2d 内置属性导出，不产生独立 Pad op。

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
        kernel_length: int = 63,
        dropout: float = 0.5,
        activation: str = "elu",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_length = input_length
        self.F1, self.D, self.F2 = F1, D, F2
        self.kernel_length = kernel_length

        # ---- Block 1：时间卷积 + 空间 depthwise 卷积 ----
        # 奇数核 + 手动补零（valid conv）：RV1106 NPU 的 Conv 不支持 padding，
        # 若用 Conv2d(padding=...) 会在 ONNX→RKNN 时被拆成独立 Pad op（NPU 不支持）。
        # 故改用 forward 里 torch.cat 手动补零 + Conv(padding=0)，数学等价。
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=0, bias=False)
        self.pad1_amt = kernel_length // 2
        self.bn1 = nn.BatchNorm2d(F1)
        # depthwise：每组一个 (n_channels, 1) 卷积核，把 8 个电极压成 1
        self.depthwise = nn.Conv2d(
            F1, D * F1, (n_channels, 1), groups=F1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(D * F1)
        # 激活函数：elu（原版）/ relu（诊断：排除 ELU 负半轴指数对 INT8 量化的影响）
        self.elu = nn.ELU() if activation == "elu" else nn.ReLU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # ---- Block 2：separable 卷积（depthwise (1,15) + pointwise (1,1)）----
        # 奇数核 15 + 手动补零（valid conv），同样避免 Pad op
        self.sep_depthwise = nn.Conv2d(
            D * F1, D * F1, (1, 15), groups=D * F1, padding=0, bias=False
        )
        self.pad2_amt = 7
        self.pointwise = nn.Conv2d(D * F1, F2, (1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # ---- 分类头：T → T//4 → T//32 ----
        # 用 Conv2d(1x1) 代替 Linear：Linear 在 ONNX 里是 Gemm，
        # rknn-toolkit2 2.3.2 对 Gemm 的量化有 bug（logits[0] 恒定偏移 -44），
        # Conv2d(1x1) 走 Conv 量化路径（NPU 原生优化），规避该 bug。
        self.temporal_out = input_length // 32
        self.flatten = nn.Flatten()
        self.fc = nn.Conv2d(F2 * self.temporal_out, n_classes, (1, 1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, F2*T//32) 的 240 维特征（flatten 后，fc 之前）。

        供 NPU 只算到特征层（P2-1：fc 拆到 CPU），避免输出层 INT8 量化误差。
        """
        # Block 1
        B, C, H, T = x.shape
        pad1 = x.new_zeros(B, C, H, self.pad1_amt)
        x = torch.cat([pad1, x, pad1], dim=3)   # (B, C, H, T + 2*pad1)
        x = self.conv1(x)          # valid conv → (B, F1, C, T)
        x = self.bn1(x)
        x = self.depthwise(x)      # (B, D*F1, 1, T)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)          # (B, D*F1, 1, T//4)
        x = self.dropout1(x)

        # Block 2
        B, C, H, T = x.shape
        pad2 = x.new_zeros(B, C, H, self.pad2_amt)
        x = torch.cat([pad2, x, pad2], dim=3)   # (B, C, H, T//4 + 2*pad2)
        x = self.sep_depthwise(x)  # valid conv → (B, D*F1, 1, T//4)
        x = self.pointwise(x)      # (B, F2, 1, T//4)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)          # (B, F2, 1, T//32)
        x = self.dropout2(x)
        x = self.flatten(x)        # (B, F2 * T//32) = (B, 240)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, T) → 输出 (B, n_classes) 未归一化 logits。"""
        x = self.forward_features(x)             # (B, 240)
        # 分类头：reshape (B, C, 1, 1) → conv1x1 → (B, n_classes)
        x = x.view(x.shape[0], self.F2 * self.temporal_out, 1, 1)  # (B, C, 1, 1)
        x = self.fc(x)             # (B, n_classes, 1, 1)
        x = x.view(x.shape[0], self.n_classes)  # (B, n_classes)
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

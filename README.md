# HBC-BCI：国产 ¥338 芯片的端侧脑机接口 × 人体通信

> **一块 ¥338 的国产 AI 芯片，完成脑电解码 + 人体通信回传的融合闭环。**
> EEGNet 端侧推理 **1.2ms**（15× 加速），EQS-HBC 人体信道作为回传链路——不用 RF，用人体本身传神经信号。

---

## 结果速览

| 指标 | 数值 | 说明 |
|---|---|---|
| EEGNet 8ch 运动想象二分类 | **76.31%** | BCIC IV 2a 真实脑电，9 被试 |
| 端侧 NPU 推理 | **1.2 ms** | vs 纯 CPU 18.5ms，**15× 加速** |
| EQS-HBC 体表信道 | **33.1 dB @ 1MHz** | IT'IS 组织参数，与 Sen 组 BP-QBC 实测吻合 |
| 端到端鲁棒性 | **接收 SNR ≥18dB 恢复直连精度** | 量化 + 信道损失不影响分类 |

---

## 一句话解释这个项目

脑机接口（BCI）的无线化是必然趋势，但主流方案（RF 射频）功耗高、还要天线对准。**EQS-HBC（电准静态人体通信）用人体本身作为信道**，功耗是 RF 的约万分之一。

这个项目把这条链路做到了**一块 ¥338 的国产芯片**上：

```
头皮电极 → ADS1299(采集) → RV1106 NPU(EEGNet 解码) → AD9833(HBC 1MHz 载波) → 人体信道 → 接收
```

---

## 目录结构

```
hbc-bci-luckfox/
├── board/           # 板载代码（C + Python）
│   ├── eegnet_infer.c        # EEGNet 纯 C 移植（与 PyTorch 逐位一致）
│   ├── infer_eegnet_rknn.c   # RV1106 NPU 推理（zero-copy + INT8）
│   ├── hbc_channel.c/h       # EQS-HBC 信道模型（IT'IS Cole-Cole 分层）
│   ├── hbc_bci_fusion.c      # 端到端融合闭环（脑电→信道→解码）
│   ├── eeg_capture.py        # ADS1299 8ch EEG 采集（SPI）
│   └── hbc_tx.py             # AD9833 1MHz HBC 载波（OOK）
├── model/           # 模型训练 + ONNX/RKNN 转换
│   ├── eegnet.py / train.py / eegnet.pth
│   ├── export_onnx.py / convert_rknn.py / calibration.py
├── data/            # BCIC IV 2a 下载 + 预处理 + IT'IS 解析
├── scripts/         # 部署 + 验证脚本
└── config.yaml      # 全局配置（板子/采样率/模型/HBC 参数）
```

配套仿真包（信道模型 + 端到端 SNR 扫描 + 结果图）见 [hbc-bci-sim](https://github.com/YensenHo/hbc-bci-sim)。

---

## 快速开始

### 硬件（总价 ~¥500）
- Luckfox Pico Ultra（RV1106，¥338，0.5 TOPS NPU）
- ADS1299 模块（8ch 脑电前端，¥60-80）
- AD9833 模块（DDS，¥12）

### 软件流程
```bash
# 1. 训练 EEGNet（PC，需要 torch）
python3 model/train.py                    # → eegnet.pth（76.31%）

# 2. 导出 ONNX（Mac/PC）
python3 model/export_onnx.py              # → eegnet.onnx

# 3. 转 RKNN（需 Linux x86，rknn-toolkit2）
python3 model/convert_rknn.py             # → eegnet.rknn (INT8, <50KB)

# 4. 部署到板子（USB RNDIS）
bash scripts/deploy.sh                    # scp 上传 + 编译
scp model/eegnet.rknn root@172.32.0.93:/root/

# 5. 板载 NPU 推理
#    交叉编译 board/infer_eegnet_rknn.c 后上板执行
```

---

## 关键技术细节

### 1. EEGNet 纯 C 移植（与 PyTorch 逐位一致）
- 全手写 C 前向（Conv2d/Depthwise/BatchNorm/ELU/AvgPool），logits 最大误差 **1.08e-6**
- 奇数核 + 手动补零，规避 RV1106 NPU 不支持的 Pad 算子

### 2. NPU 部署（15× 加速）
- zero-copy 模式 + INT8 量化 + NHWC 布局
- 模型 <50KB，单次推理 1.2ms

### 3. EQS-HBC 信道建模
- IT'IS V4.2 4-Cole-Cole 色散模型，分层 R-C 梯形网络
- 组织参数与 Sen 组 BP-QBC（Nature Electronics 2023）实测精确吻合（εr=860/σ=0.163 @1MHz）

---

## 诚实边界（重要）

- **信道是模型，不是测量**：33.1dB 来自参数化信道模型（IT'IS + 返回路径电容），**不代表穿过真实人体的实测传输**。
- **端到端融合是模型驱动**：脑电 → 信道 → 解码的闭环，信道部分是软件模型，不是真实体耦合传输。
- **真实站得住的是**：真实脑电（BCIC2a）驱动的软件信号链在 RV1106 上跑通 + EEGNet 精确移植 + 量化/信道损失小到不影响分类。

**真实体耦合传输实验（信号源 + 电极 + 示波器测真实损耗）是下一步，完成后"模型"才升级为"测量"。**

---

## 定位声明（诚实版）

- **不声称首创 EQS-HBC**（Purdue Sen 组 2018 首创，BP-QBC 上 Nature Electronics 2023）
- **不声称首创端侧 EEGNet**（ETH Zurich PULP 组已系统做过 ARM/RISC-V 部署）
- **可辩护的定位**：公开可检索范围内，**首个在国产低成本 NPU（RV1106，¥338）上端侧完成 EEGNet 脑电解码、并以 EQS-HBC 人体信道作为回传链路的融合闭环系统**——单项各有先例，"国产 NPU 端侧神经解码 × 体耦合传输闭环"这个组合无公开先例。

---

## 许可

代码：MIT。组织参数来自 IT'IS V4.2（学术用途需遵循 IT'IS 许可）。

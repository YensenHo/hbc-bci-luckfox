# Luckfox Pico Ultra — HBC-BCI 端侧验证 · 部署包

> 目标：把 PPT 里的 **LIF 脉冲仿真 + EEGNet 解码 + HBC 信号生成** 全部跑到 Luckfox 上。
> 对应文档：`../20260812/Luckfox部署指南.md`、`../20260812/EEG_数据集与模型设计.md`、`../20260812/Luckfox_Mac刷机手册.md`

---

## 一、目录结构

```
luckfox-deploy/
├── config.yaml            # 全局配置（板子/IP/采样率/模型参数/HBC载波）
├── requirements.txt       # PC 端 + 板载依赖
├── README.md              # 本文件
├── data/                  # 【Codex 负责】数据集下载 + 预处理 + 加载器
│   ├── download_bcic2a.py #   BCIC IV 2a 四分类运动想象下载（MOABB→Graz→BNCI 三级回退）
│   ├── preprocess.py      #   MNE 滤波(0.5-50Hz)+陷波(50Hz)+分段[0.5,2.5]s+标准化 → .npz
│   └── dataset.py         #   PyTorch Dataset 加载器
├── model/                 # 【Codex 负责】EEGNet 训练 + ONNX/RKNN 转换
│   ├── eegnet.py          #   EEGNet 模型（F1=8 D=2 F2=16, ~2500 参数）
│   ├── train.py           #   训练脚本（目标 >70% 四分类准确率）
│   ├── export_onnx.py     #   PyTorch → ONNX（固定输入 1×22×500）
│   ├── calibration.py     #   生成 RKNN INT8 量化校准集
│   └── convert_rknn.py    #   ONNX → RKNN（target rv1106, INT8, <50KB）
├── board/                 # 【Hermes 负责】板载脚本
│   ├── lif_sim.py         #   LIF 单神经元仿真（Python，阶段 1）
│   ├── lif_net.c          #   LIF 10^4 神经元网络（C 加速，阶段 1）
│   ├── infer_eegnet.py    #   NPU 推理（rknn-lite，阶段 2）
│   ├── eeg_capture.py     #   ADS1299 8ch EEG 采集（SPI，阶段 3）
│   └── hbc_tx.py          #   AD9833 1MHz HBC 载波 + OOK（阶段 4）
└── scripts/
    └── deploy.sh          # 一键 scp 上传 + C 编译 + LIF 验证
```

---

## 二、快速开始（按阶段推进）

### 阶段 0 — 刷机 + 连接（见 `../20260812/Luckfox_Mac刷机手册.md`）

板子出厂预装 factory test image，插 USB-C 即可启动，无需先刷机：

```bash
ssh root@172.32.0.93     # 密码 luckfox
```

### 阶段 1 — LIF 模型（第 1 天跑通）

```bash
bash scripts/deploy.sh                       # 一键上传 + 编译 + 运行验证
# 预期：lif_sim.py 输出 spikes≈2-3 次/100ms；lif_net 输出 10^4 神经元耗时 <2s
```

### 阶段 2 — EEGNet NPU 部署（PC 端 + 板端）

PC 端（训练 + 导出）：

```bash
cd data  && python3 download_bcic2a.py      # 下载 BCIC IV 2a（约 4GB）
python3 preprocess.py                        # 预处理 → .npz
cd ../model && python3 train.py              # 训练 EEGNet → eegnet.pth
python3 export_onnx.py                       # → eegnet.onnx
python3 calibration.py                       # 生成校准集
# 下面这条在 WSL2/云主机跑（rknn-toolkit2 仅 Linux x86）：
python3 convert_rknn.py                      # → eegnet.rknn (INT8, <50KB)
```

板端（推理）：

```bash
scp model/eegnet.rknn root@172.32.0.93:/root/
ssh root@172.32.0.93 "python3 /root/infer_eegnet.py"
# 预期：四分类结果 + 推理 < 5ms/次
```

### 阶段 3 — EEG 采集（硬件 ¥150-200）

接线见 `config.yaml` 的 `hardware` 段 + `board/eeg_capture.py` 顶部注释。
板子使能 SPI 后：`python3 /root/eeg_capture.py`

### 阶段 4 — HBC 发射

`python3 /root/hbc_tx.py` → AD9833 输出 1 MHz EQS 载波，OOK 调制。

### 阶段 5 — 完整 TDD 链路 Demo

```
头皮电极 → ADS1299(采集) → SPI → Luckfox(EEGNet解码) → AD9833(HBC发射) → 腕部接收
```
TDD 时隙：99% 采集+解码，1% HBC 发送（BP-QBC 同款策略）。端到端延迟目标 <50ms。

---

## 三、数据集说明

| 数据集 | 通道 | 采样率 | 任务 | 优先级 |
|--------|------|--------|------|--------|
| **BCIC IV 2a**（本包训练用） | 22 | 250Hz | 四分类运动想象 | ⭐ 主 |
| BCIC IV 2b | 3 | 250Hz | 二分类 | 备 |
| PhysioNet eegmmidb | 64 | 160Hz | 运动/想象 | 备 |
| High Gamma | 128 | 500Hz | 实际运动 | 高通道压力测试 |
| MOABB | 聚合 12 个 | - | 多基线对比 | 论文 baseline |

下载地址与格式详见 `../20260812/EEG_数据集与模型设计.md`。

---

## 四、常见坑

| 问题 | 解决 |
|------|------|
| LIF 不发放（spikes=0） | 注入电流 I 必须 > 8.5nA（稳态 -65+10I > 阈值 20），见 lif_sim.py 注释 |
| rknn 转换报错 | rknn-toolkit2 只在 Linux x86 跑，Mac 用 WSL2/云主机 |
| 64MB 内存不够 | Pico Ultra 是 512MB 无压力；EEGNet 仅 ~50KB |
| SPI 设备不存在 | 设备树使能 SPI1，wiki.luckfox.com 有 overlay 教程 |
| PWM 到不了 1MHz | 别用 PWM 生成载波，用 AD9833 DDS（hbc_tx.py 已实现） |
| ADS1299 读不到数据 | 检查 DRDY 下降沿 + SPI mode=1(CPOL=0,CPHA=1) |
| pip 超时 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

---

## 五、参考

- Luckfox Wiki: https://wiki.luckfox.com/
- RKNN Toolkit2: https://github.com/airockchip/rknn-toolkit2
- BP-QBC 论文: `../20260812/papers/BP-QBC_2205.08540.pdf`
- 技术方案总纲: `../docs/4_总体技术方案_BCI-BAN-AI闭环系统.docx`

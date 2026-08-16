#!/usr/bin/env python3
"""
eeg_capture.py — ADS1299 8 通道脑电采集（阶段 3，EEG 前端）

用途：通过 SPI 读取 ADS1299（TI 8 通道 24bit 生物电模拟前端）的脑电数据。

ADS1299 协议要点：
  - SPI 模式 CPOL=0, CPHA=1（mode 1），DRDY 低电平表示数据就绪
  - RDATA(0x12)：读 1 次 → 返回 3 字节状态 + 8 通道 × 3 字节 = 27 字节
  - 24bit 有符号补码，LSB = VREF / (2^23 - 1) / GAIN
  - 命令：SDATAC(0x11) 停止连续读，RDATAC(0x10) 开始连续读，
          WREG(0x40|addr) 写寄存器，RREG(0x20|addr) 读寄存器

硬件接线（Luckfox Pico Ultra，SPI0 引脚，见 config.yaml spi 段）：
    ADS1299 SCLK  → SPI0_CLK  (GPIO1_PC1)
    ADS1299 DIN   → SPI0_MOSI (GPIO1_PC2)
    ADS1299 DOUT  → SPI0_MISO (GPIO1_PC3)
    ADS1299 CS    → SPI0_CS0  (GPIO1_PC0)
    ADS1299 DRDY  → GPIO 中断 (下降沿触发)
    GND/3.3V      → 对应电源

用法（板子上，需先使能 SPI0：luckfox-config → SPI → enable）：
    python3 /root/eeg_capture.py

⚠️ 本脚本依赖真实硬件，需接 ADS1299 模块后验证；DRDY 中断用 GPIO 轮询
   简化实现（生产环境应改用 /sys/class/gpio 中断或内核驱动）。
"""
import time
import spidev

# ---- ADS1299 命令字 ----
CMD_WAKEUP = 0x02
CMD_STANDBY = 0x04
CMD_RESET = 0x06
CMD_START = 0x08
CMD_STOP = 0x0A
CMD_RDATAC = 0x10
CMD_SDATAC = 0x11
CMD_RDATA = 0x12
WREG = lambda a: 0x40 | a   # 写寄存器
RREG = lambda a: 0x20 | a   # 读寄存器

# ---- 寄存器地址 ----
REG_CONFIG1 = 0x01  # 采样率
REG_CONFIG2 = 0x02
REG_CONFIG3 = 0x03
REG_CH1SET = 0x04   # CH1SET..CH8SET = 0x04..0x0B

N_CHANNELS = 8
VREF = 4.5          # V（ADS1299 内部基准，视硬件而定）
GAIN = 24           # 可编程增益（CHnSET 低 3 位）


class ADS1299:
    def __init__(self, bus=0, device=0, max_speed=2_000_000):
        # ⚠️ Pico Ultra 只有 SPI0（无 SPI1），用 bus=0 而非 1
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed
        self.spi.mode = 1           # CPOL=0, CPHA=1
        self._reset()
        self._configure()

    def _cmd(self, cmd):
        self.spi.xfer2([cmd])

    def _write_reg(self, addr, value):
        self.spi.xfer2([WREG(addr), 0x00, value])

    def _reset(self):
        self._cmd(CMD_RESET)
        time.sleep(0.1)

    def _configure(self):
        """配置采样率 250Hz + 8 通道使能 + 增益 24"""
        self._cmd(CMD_SDATAC)
        # CONFIG1: 采样率 250 SPS（默认 0x96 对应 250SPS，视晶振而定）
        self._write_reg(REG_CONFIG1, 0x96)
        # CONFIG3: 内部参考使能
        self._write_reg(REG_CONFIG3, 0xEC)
        # CH1SET..CH8SET: 正常输入 + 增益 24（0x60 | (gain 编码 0b110 = 24)）
        gain_bits = {1: 0b000, 2: 0b001, 4: 0b010, 6: 0b011,
                     8: 0b100, 12: 0b101, 24: 0b110}[GAIN]
        for ch in range(1, N_CHANNELS + 1):
            self._write_reg(REG_CH1SET + (ch - 1), 0x60 | gain_bits)
        self._cmd(CMD_START)
        self._cmd(CMD_RDATAC)       # 进入连续读模式

    def read_sample(self):
        """读一次完整采样：27 字节 = 3 状态 + 8×3 数据，返回 8 通道电压 (V)"""
        raw = self.spi.xfer2([0x00] * (3 + N_CHANNELS * 3))
        data = raw[3:]              # 丢弃 3 字节状态
        values = []
        for ch in range(N_CHANNELS):
            b = data[ch * 3:(ch + 1) * 3]
            code = (b[0] << 16) | (b[1] << 8) | b[2]
            if code & 0x800000:     # 24bit 有符号补码
                code -= 0x1000000
            voltage = code * VREF / (2 ** 23 - 1) / GAIN
            values.append(voltage)
        return values

    def close(self):
        self.spi.close()


if __name__ == "__main__":
    ads = ADS1299()
    try:
        print("ADS1299 采集启动：8 通道 @ 250Hz")
        for i in range(10):
            vals = ads.read_sample()
            # 打印每通道 μV（四舍五入到整数，脑电 μV 量级）
            print(f"样本 {i}: " + " ".join(f"ch{c}={int(v * 1e6):>6}μV"
                                           for c, v in enumerate(vals, 1)))
            time.sleep(0.004)       # 250Hz 周期 4ms
    finally:
        ads.close()

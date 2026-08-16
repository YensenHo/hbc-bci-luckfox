#!/usr/bin/env python3
"""
hbc_tx.py — AD9833 DDS 生成 EQS-HBC 载波（阶段 4，把 Luckfox 变成「HBC 发射机」）

用途：通过 SPI 控制 AD9833 模块生成 1 MHz EQS-HBC 载波，支持 OOK 调制
      （数据 1 = 有载波，数据 0 = 关断）。1 MHz = BP-QBC 论文使用的载波频率。

⚠️ 为什么不用板载 PWM：RV1106 PWM 无法稳定输出 1 MHz，AD9833 DDS（¥12）才是正解。
   AD9833 频率字：freq_word = hz * 2^28 / 25MHz（板载 25MHz 晶振）

硬件接线（Luckfox Pico Ultra，SPI0 引脚，与 ADS1299 分时复用总线）：
    AD9833 SDATA  → SPI0_MOSI (GPIO1_PC2)
    AD9833 SCLK   → SPI0_CLK  (GPIO1_PC1)
    AD9833 FSYNC  → GPIO 软件 CS（与 ADS1299 分时复用 SPI0）
    AD9833 OUT    → 放大后接 HBC 电极

用法（板子上，需先使能 SPI0：ls /dev/spidev* 应看到 spidev0.x）：
    python3 /root/hbc_tx.py
"""
import time
import spidev

# ---- AD9833 寄存器/控制字 ----
DDS_CLOCK_HZ = 25_000_000          # 25MHz 晶振
CTRL_RESET = 0x2100                # B28=1, RESET=1
CTRL_RUN = 0x2000                  # RESET=0（退出复位，开始输出）
REG_FREQ0 = 0x4000                 # FREQ0 寄存器选择（bit15-14 = 01）
REG_PHASE0 = 0xC000                # PHASE0 = 0


class AD9833:
    def __init__(self, bus=0, device=1, max_speed=1_000_000):
        # ⚠️ Pico Ultra 只有 SPI0（无 SPI1），用 bus=0 而非 1
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed
        self.spi.mode = 0           # AD9833: CPOL=0, CPHA=0
        self._write16(CTRL_RESET)   # 上电复位

    def _write16(self, word16):
        """写 16 位控制字（高字节在前）"""
        self.spi.xfer2([(word16 >> 8) & 0xFF, word16 & 0xFF])

    def set_freq(self, hz):
        """设置 FREQ0 输出频率"""
        fword = int(hz * (1 << 28) / DDS_CLOCK_HZ)   # 频率字
        msb = (fword >> 14) & 0x3FFF                 # 高 14 位
        lsb = fword & 0x3FFF                         # 低 14 位
        self._write16(REG_FREQ0 | lsb)
        self._write16(REG_FREQ0 | msb)
        self._write16(REG_PHASE0)
        self._write16(CTRL_RUN)

    def off(self):
        """关断输出（复位 = 无载波）"""
        self._write16(CTRL_RESET)

    def close(self):
        self.spi.close()


def ook_tx(dds, bits, bit_period=0.001):
    """OOK 开关键控：bit=1 发载波，bit=0 关断。bit_period 秒/bit"""
    for b in bits:
        if b:
            dds.set_freq(1_000_000)   # 1 MHz ON
        else:
            dds.off()                 # 载波关断
        time.sleep(bit_period)
    dds.off()


if __name__ == "__main__":
    dds = AD9833()
    try:
        dds.set_freq(1_000_000)       # 1 MHz EQS 载波
        print("HBC TX ON: 1 MHz（BP-QBC 同款载波）")
        time.sleep(2)

        # OOK 演示：发送比特流 0b10110010
        print("OOK 发送: 10110010")
        ook_tx(dds, [1, 0, 1, 1, 0, 0, 1, 0], bit_period=0.05)
    finally:
        dds.off()
        dds.close()
        print("HBC TX OFF")

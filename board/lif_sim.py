#!/usr/bin/env python3
"""
LIF 脉冲神经元仿真 — Luckfox Pico Ultra 板载验证脚本（阶段 1）

用途：验证 LIF 积分发放神经元模型在端侧芯片上的运行正确性与实时性，
     为「LIF-EEGNet 适配 SNN 硬件」的 PPT 承诺提供第一块实证。

用法（板子上）：
    ssh root@172.32.0.93
    python3 /root/lif_sim.py

⚠️ 关键陷阱（已踩坑记录）：
    vrest=-65, R=10, vth=20 时，稳态膜电位 v_ss = vrest + R*I。
    只有 I > (vth - vrest)/R = 8.5 才会发放！I=3 之类"看似合理"的
    电流值永远不产生 spike（静默 bug）。本脚本用 I=10，保证发放。

    理论发放周期：T_fire = tau * ln((v_ss - vrest)/(v_ss - vth))
                = 20 * ln(100/15) ≈ 37.9ms  → 100ms 内约 2-3 个 spike
"""
import time


class LIF:
    """Leaky Integrate-and-Fire 单神经元（欧拉法离散）"""

    def __init__(self, tau=20.0, R=10.0, vth=20.0, vrest=-65.0):
        self.tau = tau      # 膜时间常数 (ms)
        self.R = R          # 膜电阻 (Mohm)
        self.vth = vth      # 发放阈值 (mV)
        self.vrest = vrest  # 静息电位 (mV)
        self.v = vrest      # 当前膜电位

    def step(self, dt, I_ext):
        """
        单步仿真。dv/dt = (-(v - vrest) + R * I_ext) / tau
        返回 1 表示本步发放 spike（发放后膜电位重置到 vrest）。
        """
        self.v += dt * (-(self.v - self.vrest) + self.R * I_ext) / self.tau
        if self.v >= self.vth:
            self.v = self.vrest
            return 1
        return 0


if __name__ == "__main__":
    n = LIF()
    spikes = 0
    DT = 0.001          # ms
    T_STEPS = 100000    # 100ms 仿真

    t0 = time.time()
    for _ in range(T_STEPS):
        spikes += n.step(DT, 10.0)  # ⚠️ I=10 才能发放
    t1 = time.time()

    elapsed_ms = (t1 - t0) * 1000
    print(f"spikes={spikes}, 100ms 仿真耗时={elapsed_ms:.1f}ms")
    print(f"预期发放 ~2-3 次/100ms（T_fire≈37.9ms），实际 {spikes} 次")
    # 纯 Python 在 1.2GHz A7 上约 1-3 秒；要跑 10^4 神经元规模请用 C 版 lif_net.c

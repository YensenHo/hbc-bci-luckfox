#!/usr/bin/env python3
"""
verify.py — Luckfox 部署包自检（纯 stdlib，无第三方依赖）

用途：验证板载脚本语法、config.yaml 结构、SPI0 引脚配置，并用 stub spidev
      实测 ADS1299/AD9833 的寄存器/解码逻辑（无需真实硬件）。

用法：
    python3 scripts/verify.py          # 全量自检，输出 PASS/FAIL 汇总
    exit code 0 = 全部通过

说明：这是 ad-hoc 冒烟验证（语法 + stub 逻辑），不代表真实硬件/数据集下的行为。
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "board")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


# ---- 1. 板载脚本语法 ----
for f in ("eeg_capture.py", "hbc_tx.py", "infer_eegnet.py", "lif_sim.py"):
    r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(BOARD, f)],
                       capture_output=True, text=True)
    check(f"py_compile {f}", r.returncode == 0, r.stderr.strip()[:100])

# ---- 1b. model/ 脚本语法 ----
for f in ("eegnet.py", "train.py", "export_c_weights.py", "export_test_trials.py", "verify_c_infer.py"):
    r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(ROOT, "model", f)],
                       capture_output=True, text=True)
    check(f"py_compile model/{f}", r.returncode == 0, r.stderr.strip()[:100])

# ---- 1c. data/ 脚本语法 ----
for f in ("download_bcic2a.py", "preprocess.py", "dataset.py", "parse_itis.py"):
    r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(ROOT, "data", f)],
                       capture_output=True, text=True)
    check(f"py_compile data/{f}", r.returncode == 0, r.stderr.strip()[:100])

# ---- 1d. scripts/ 脚本语法 ----
for f in ("verify.py", "return_path_cap.py"):
    r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(ROOT, "scripts", f)],
                       capture_output=True, text=True)
    check(f"py_compile scripts/{f}", r.returncode == 0, r.stderr.strip()[:100])

# ---- 2. config.yaml 结构 ----
cfg = open(os.path.join(ROOT, "config.yaml")).read()
check("config.yaml bus=0 + SPI0 引脚", "bus: 0" in cfg and "GPIO1_PC1" in cfg)
check("config.yaml 无 bus:1 残留", "bus: 1" not in cfg)

# ---- 3. infer_eegnet.py 输出 C API 提示 ----
r = subprocess.run([sys.executable, os.path.join(BOARD, "infer_eegnet.py")],
                   capture_output=True, text=True)
check("infer_eegnet.py C API 提示", r.returncode == 0 and "librknnmrt" in r.stdout)

# ---- 4. ADS1299（stub spidev，验证 bus=0 + 8ch 解码）----
def _load(name: str, fake_spidev):
    sys.modules["spidev"] = fake_spidev
    spec = importlib.util.spec_from_file_location(name, os.path.join(BOARD, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fake = types.ModuleType("spidev")
class _SpiA:  # ADS1299：记录 open(bus,dev) + RDATA 返回 27 字节
    def __init__(self): self.writes, self.opened = [], None
    def open(self, bus, dev): self.opened = (bus, dev)
    def xfer2(self, data):
        self.writes.append(list(data))
        return ([0] * 3 + [0x00, 0x00, 0x7F] * 8) if len(data) == 27 else [0] * len(data)
    def close(self): pass
fake.SpiDev = _SpiA
eeg = _load("eeg_capture", fake)
ads = eeg.ADS1299()
check("ADS1299 用 SPI0(bus=0)", ads.spi.opened == (0, 0), str(ads.spi.opened))
vals = ads.read_sample()
ch1 = [w for w in ads.spi.writes if len(w) == 3 and w[0] == eeg.WREG(eeg.REG_CH1SET)]
check("ADS1299 8ch 解码", len(vals) == 8 and bool(ch1) and ch1[0][2] == 0x66, f"channels={len(vals)}")

# ---- 5. AD9833（stub spidev，验证 bus=0 + 频率字）----
fake2 = types.ModuleType("spidev")
class _SpiB:
    def __init__(self): self.writes, self.opened = [], None
    def open(self, bus, dev): self.opened = (bus, dev)
    def xfer2(self, data): self.writes.append(list(data)); return list(data)
    def close(self): pass
fake2.SpiDev = _SpiB
hbc = _load("hbc_tx", fake2)
dds = hbc.AD9833()
check("AD9833 用 SPI0(bus=0)", dds.spi.opened == (0, 1), str(dds.spi.opened))
dds.set_freq(1_000_000)
fword = int(1_000_000 * (1 << 28) / hbc.DDS_CLOCK_HZ)
check("AD9833 频率字", fword == 10737418 and dds.spi.writes[-1] == [0x20, 0x00], f"freq={fword}")

# ---- 6. C 文件编译检查（-Wall 干净）----
import shutil
_CC = shutil.which("cc") or shutil.which("clang") or "cc"
_c_srcs = ["lif_net.c", "hbc_channel.c", "hbc_bci_loop.c", "eegnet_infer.c",
           "hbc_bci_fusion.c", "hbc_bci_realtime.c", "infer_eegnet_rknn.c"]
for src in _c_srcs:
    out = f"/tmp/verify_{src}.o"
    r = subprocess.run([_CC, "-O2", "-Wall", "-c", os.path.join(BOARD, src), "-o", out],
                       capture_output=True, text=True)
    check(f"cc -Wall 编译 {src}", r.returncode == 0 and "warning" not in r.stderr.lower(),
          r.stderr.strip()[:120] if r.stderr.strip() else "clean")
    if os.path.exists(out):
        os.remove(out)


def main() -> int:
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [n for n, ok, _ in results if not ok]
    print("=" * 52)
    print(f"部署包自检: {passed}/{len(results)} 通过")
    for n, ok, d in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

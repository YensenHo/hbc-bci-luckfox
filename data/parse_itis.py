#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_itis.py — 解析 IT'IS 组织参数库 V4.2，提取多层组织的 4-Cole-Cole 参数

IT'IS 数据源：DOI 10.13099/VIP21000-04-2（下载见 data/itis/Database-V4-2.zip）
Dielectric Properties 字段（14 个，按表头顺序）：
    ef  = ε_∞（高频极限介电常数）
    del1..del4 = Δε_n（色散强度）
    tau1..tau4 = τ_n（弛豫时间，单位依次 ps/ns/µs/ms）
    alf1..alf4 = α_n（Cole-Cole 展宽参数）
    sig = σ_i（静态离子电导 S/m）

4-Cole-Cole 复介电常数（Gabriel 1996）：
    ε(ω) = ε_∞ + Σ_{n=1..4} Δε_n / (1 + (jωτ_n)^(1-α_n)) + σ_i/(jωε_0)

多层组织（对应 hbc_channel.h 的 layers）：
    皮肤 Skin / 脂肪 Fat / 颅骨 Bone (Cortical) / 脑脊液 Cerebrospinal Fluid / 脑灰质 Brain (Grey Matter)

用法
----
  python3 data/parse_itis.py [--freq 100e3 1e6 10e6]   # 打印指定频点的 ε'/σ，供交叉验证
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ITIS_DIR = Path(__file__).resolve().parent.parent / "data" / "itis"
ASCII_FILE = ITIS_DIR / "Thermal_dielectric_acoustic_MR properties_database_V4.2(ASCII).txt"

EPS0 = 8.8541878128e-12

# 多层组织 → IT'IS 里的确切名称
TARGETS = ["Skin", "Fat", "Bone (Cortical)", "Cerebrospinal Fluid", "Brain (Grey Matter)"]

# 4-Cole-Cole 各 τ 的单位（s）
TAU_UNITS = [1e-12, 1e-9, 1e-6, 1e-3]  # ps, ns, µs, ms


def parse_tissue_db(path: Path) -> dict[str, dict]:
    """解析 ASCII 库，返回 {组织名: {ef, del1..4, tau1..4(s), alf1..4, sig}}。"""
    with open(path, encoding="latin-1") as f:
        lines = f.readlines()

    # 表头：第 2 行是大分类，第 3 行是子字段名
    header_sub = [c.strip() for c in lines[2].rstrip("\r\n").split("\t")]
    try:
        i_ef = header_sub.index("ef")
        i_alf4 = header_sub.index("alf4")
    except ValueError as e:
        raise SystemExit(f"✗ 表头解析失败：{e}\n字段列表：{header_sub}") from e

    # 字段顺序（从 ef 到 alf4，共 14 个）
    keys = ["ef", "del1", "tau1", "alf1", "del2", "tau2", "alf2",
            "sig", "del3", "tau3", "alf3", "del4", "tau4", "alf4"]

    out: dict[str, dict] = {}
    for line in lines[3:]:
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) <= i_alf4:
            continue
        name = fields[1].strip()   # 行首 tab 占 fields[0]，组织名在 fields[1]
        if name not in TARGETS:
            continue
        vals = [fields[i_ef + j] for j in range(14)]
        d: dict = {}
        for k, v in zip(keys, vals):
            d[k] = float(v) if v.strip() else 0.0
        # τ 单位转换
        d["tau1"] *= TAU_UNITS[0]
        d["tau2"] *= TAU_UNITS[1]
        d["tau3"] *= TAU_UNITS[2]
        d["tau4"] *= TAU_UNITS[3]
        out[name] = d
    return out


def cole_cole(d: dict, freq: float) -> tuple[float, float]:
    """返回 (εr', σ_eff) 在某频率。σ_eff = ω ε0 εr''。"""
    w = 2.0 * np.pi * freq
    eps = complex(d["ef"], 0.0)
    for n in range(1, 5):
        de = d[f"del{n}"]
        tau = d[f"tau{n}"]
        alpha = d[f"alf{n}"]
        if de == 0.0:
            continue
        eps += complex(de) / (1.0 + (1j * w * tau) ** (1.0 - alpha))
    eps += complex(0.0, -d["sig"] / (w * EPS0))  # + σ_i/(jωε0)
    return float(eps.real), float(-eps.imag * w * EPS0)  # εr', σ_eff


def export_c_header(db: dict, out_path: Path) -> None:
    """生成 C 参数表 cole_cole_params.h（供 hbc_channel.c include）。"""
    lines = [
        "/* 自动生成自 IT'IS V4.2 (DOI 10.13099/VIP21000-04-2) — 勿手改 */",
        "/* 生成命令：python3 data/parse_itis.py --export-c */",
        "/* 多层组织 4-Cole-Cole 参数（Gabriel 1996 色散模型） */",
        "",
    ]
    lines.append(f"static const ColeColeParams ITIS_LAYERS[{len(TARGETS)}] = {{")
    for name in TARGETS:
        d = db[name]
        de = "{%g, %g, %g, %g}" % (d["del1"], d["del2"], d["del3"], d["del4"])
        tau = "{%.9e, %.9e, %.9e, %.9e}" % (d["tau1"], d["tau2"], d["tau3"], d["tau4"])
        al = "{%g, %g, %g, %g}" % (d["alf1"], d["alf2"], d["alf3"], d["alf4"])
        lines.append(f"    /* {name} */ {{ {d['ef']:g}, {de}, {tau}, {al}, {d['sig']:.6g} }},")
    lines.append("};")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ 已生成 {out_path}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="解析 IT'IS 组织参数库，提取多层组织 Cole-Cole 参数")
    parser.add_argument("--export-c", action="store_true",
                        help="生成 C 参数表 board/cole_cole_params.h")
    args = parser.parse_args()

    if not ASCII_FILE.exists():
        raise SystemExit(
            f"✗ 未找到 IT'IS ASCII 库：{ASCII_FILE}\n"
            f"  请先下载：data/itis/Database-V4-2.zip（DOI 10.13099/VIP21000-04-2）"
        )

    db = parse_tissue_db(ASCII_FILE)

    if args.export_c:
        export_c_header(db, Path(__file__).resolve().parent.parent / "board" / "cole_cole_params.h")
        return 0

    print("=" * 78)
    print("IT'IS V4.2 多层组织 · 4-Cole-Cole 参数（已做单位换算）")
    print("=" * 78)
    for name in TARGETS:
        d = db.get(name)
        if d is None:
            print(f"  ⚠ 未找到 {name}")
            continue
        print(f"\n【{name}】")
        print(f"  ε_∞ = {d['ef']:.3f}")
        print(f"  σ_i = {d['sig']:.4f} S/m")
        for n, tu in zip(range(1, 5), TAU_UNITS):
            print(f"  Δε{n} = {d[f'del{n}']:g}, τ{n} = {d[f'tau{n}']*1e6:.3f} µs, α{n} = {d[f'alf{n}']}")

    # 交叉验证：指定频点的 εr'/σ_eff
    freqs = [100e3, 1e6, 10e6, 100e6]
    print("\n" + "=" * 78)
    print("频点交叉验证（εr' / σ_eff [S/m]）")
    print("=" * 78)
    hdr = f"{'组织':<22}" + "".join(f"{f/1e6:>9.2f}MHz" for f in freqs)
    print(hdr)
    for name in TARGETS:
        d = db.get(name)
        if d is None:
            continue
        row = f"{name:<22}"
        for f in freqs:
            ep, sig = cole_cole(d, f)
            row += f"{ep:>7.0f}/{sig:>8.4f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())

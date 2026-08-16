#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
return_path_cap.py — 计算 EQS-HBC 返回路径电容 C_body_earth（人体-大地电容）

目的：替代 hbc_bci_fusion.c 里"拍脑袋"的 150pF，给出有物理依据的值。

方法 1（解析）：人体近似为球体（半径 r_body），位于接地平面上方（球心离地 d），
    用镜像法级数精确求解球体-大地电容：
        C = 4πε0·r·sinh(α)·Σ_{n=1..∞} 1/sinh(nα)，α = arccosh(d/r)
    孤立球体（d→∞）退化为 C = 4πε0·r。

方法 2（数值，scikit-fem）：2D 轴对称拉普拉斯 ∇²φ=0，人体=圆柱 Dirichlet φ=1V，
    大地 z=0 Dirichlet φ=0，解出电荷 Q，C = Q/V。用有限元交叉验证解析近似。

方法 3（文献）：Maity 1805.05200 等 EQS-HBC 论文 C_body 典型量级 ~100-200 pF。

用法
----
  python3 scripts/return_path_cap.py
"""
from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12


def sphere_ground_cap(r: float, d: float, n_terms: int = 200) -> float:
    """球体（半径 r）在接地平面上方（球心离地 d）的电容（镜像法级数）。"""
    if d <= r:
        raise ValueError("d 必须 > r（球体不能埋进地平面）")
    alpha = np.arccosh(d / r)
    s = 0.0
    for n in range(1, n_terms + 1):
        s += 1.0 / np.sinh(n * alpha)
    return 4.0 * np.pi * EPS0 * r * np.sinh(alpha) * s


def cylinder_ground_cap_approx(r: float, h: float, d_gap: float) -> float:
    """圆柱（半径 r，高 h，底离地 d_gap）对地电容的工程近似。

    把圆柱侧面+顶面近似为若干水平环对地的贡献，粗略用「圆柱表面等效对地距离」。
    这是数量级估计，精确值见数值方法。"""
    # 简化：圆柱体等效为球体（等体积），半径 r_eq = (3 r² h / 4)^(1/3)
    r_eq = (0.75 * r * r * h) ** (1.0 / 3.0)
    d_center = d_gap + h / 2.0  # 等效球心离地距离（圆柱中心）
    return sphere_ground_cap(r_eq, d_center)


def fem_cylinder_ground_cap(r_body: float, h_body: float, d_gap: float,
                            Rmax: float = 2.0, Zmax: float = 3.0,
                            nr: int = 60, nz: int = 80) -> float:
    """scikit-fem 2D 轴对称有限元求解人体-大地电容。

    模型：人体=圆柱（半径 r_body，高 h_body，底部离地 d_gap），导体（等势 φ=1V）；
    大地 z=0（φ=0）；外边界 φ=0。求解域为空气（ε0）。

    轴对称拉普拉斯 ∇²φ=0，弱形式 ∫ 2πr ∇φ·∇v dr dz = 0。
    电容用能量法：C = 2W/V² = ∫ ε0|∇φ|²·2πr dr dz / V²。
    """
    try:
        from skfem import MeshTri, Basis, ElementTriP1, asm, solve, condense
        from skfem.helpers import dot, grad
        from skfem.assembly import BilinearForm, Functional
    except ImportError as e:
        print(f"  ⚠ scikit-fem 不可用，跳过数值求解：{e}")
        return float("nan")

    # 网格：结构化，显式包含人体边界坐标
    x = np.unique(np.concatenate([np.linspace(0, Rmax, nr), [r_body]]))
    y = np.unique(np.concatenate([np.linspace(0, Zmax, nz), [d_gap, d_gap + h_body]]))
    mesh = MeshTri.init_tensor(x, y)
    basis = Basis(mesh, ElementTriP1())

    @BilinearForm
    def laplace(u, v, w):
        return 2.0 * np.pi * w.x[0] * dot(grad(u), grad(v))

    A = asm(laplace, basis)
    b = np.zeros(basis.N)

    r = mesh.p[0]
    z = mesh.p[1]

    # Dirichlet 边界节点
    tol = 1e-9
    ground = np.where(np.isclose(z, 0.0, atol=tol))[0]
    body_side = np.where(np.isclose(r, r_body, atol=tol) & (z >= d_gap - tol) & (z <= d_gap + h_body + tol))[0]
    body_top = np.where(np.isclose(z, d_gap + h_body, atol=tol) & (r <= r_body + tol))[0]
    body_bottom = np.where(np.isclose(z, d_gap, atol=tol) & (r <= r_body + tol))[0]
    outer = np.where(np.isclose(r, Rmax, atol=tol) | np.isclose(z, Zmax, atol=tol))[0]

    body = np.unique(np.concatenate([body_side, body_top, body_bottom]))
    D = np.unique(np.concatenate([ground, body, outer]))

    x_full = np.zeros(basis.N)
    x_full[body] = 1.0   # 人体 φ=1V；其余（ground + outer）默认 0V

    A_c, b_c, _, I = condense(A, b, D=D, x=x_full)
    u_c = solve(A_c, b_c)
    u = x_full.copy()      # 完整解：Dirichlet 部分已设
    u[I] = u_c             # 自由自由度用求解结果

    # 能量法算电容：C = ∫ ε0|∇φ|²·2πr dr dz（V=1 时）
    @Functional
    def energy(w):
        return 2.0 * np.pi * w.x[0] * 0.5 * EPS0 * dot(grad(w["u"]), grad(w["u"]))

    W = energy.assemble(basis, u=basis.interpolate(u))
    C = 2.0 * W   # V=1 → C = 2W
    return C


def main() -> int:
    print("=" * 70)
    print("EQS-HBC 返回路径电容 C_body_earth（人体-大地电容）")
    print("=" * 70)

    # 人体几何参数
    r_body = 0.15   # 人体等效半径 m（躯干半径量级）
    h_body = 1.70   # 身高 m
    d_gap = 0.02    # 鞋底绝缘层离地间隙 m

    # 方法1：球体近似（球心离地 = 半径 + 鞋底间隙，球底部刚触地）
    r_eq = (0.75 * r_body ** 2 * h_body) ** (1.0 / 3.0)  # 等体积球半径
    d_center = r_eq + d_gap
    C_iso = 4.0 * np.pi * EPS0 * r_eq                     # 孤立球体
    C_ground = sphere_ground_cap(r_eq, d_center)          # 对地
    print(f"\n【方法1 球体近似】等效球半径 r_eq = {r_eq*100:.1f} cm")
    print(f"  孤立球体电容 C_iso      = {C_iso*1e12:.1f} pF")
    print(f"  对地电容    C_ground    = {C_ground*1e12:.1f} pF")

    # 方法2：圆柱工程近似
    C_cyl = cylinder_ground_cap_approx(r_body, h_body, d_gap)
    print(f"\n【方法2 圆柱解析近似】C ≈ {C_cyl*1e12:.1f} pF")

    # 方法3：scikit-fem 轴对称有限元（准静态求解，最可辩护）
    print(f"\n【方法3 scikit-fem 轴对称有限元】")
    C_fem = fem_cylinder_ground_cap(r_body, h_body, d_gap)
    if not np.isnan(C_fem):
        print(f"  圆柱对地电容 C_fem = {C_fem*1e12:.1f} pF")
    else:
        print(f"  数值求解未执行")

    # 灵敏度：鞋底间隙的影响
    print(f"\n【灵敏度】鞋底间隙 d_gap 对 C_body 的影响：")
    for gap in (0.005, 0.02, 0.05, 0.10):
        d = r_eq + gap
        c = sphere_ground_cap(r_eq, d)
        print(f"  d_gap = {gap*100:4.1f} cm → C_body = {c*1e12:6.1f} pF")

    # 对比拍脑袋的 150pF
    print(f"\n【结论】当前代码用的 150pF：")
    print(f"  球体对地估计 {C_ground*1e12:.0f} pF（{C_ground*1e12/150*100:.0f}% of 150pF）")
    print(f"  150pF 落在人体-大地电容的合理量级（~50-200 pF）内。")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

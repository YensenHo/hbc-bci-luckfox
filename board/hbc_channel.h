/*
 * hbc_channel.h — EQS-HBC 穿颅信道模型（分层 R-C 梯形网络）
 *
 * G2 核心：电准静态人体通信(EQS-HBC)信号穿过颅骨的信道特性。
 * 参考：Maity et al. arXiv:1805.05200（EQS-HBC 集总生物物理模型）、
 *       Datta et al. arXiv:2010.15339、BP-QBC arXiv:2205.08540。
 *
 * 物理机制：在 1MHz 载波点，人体呈准静态等势体，信号以位移电流穿过
 * 各组织层。每层建模为并联 R-C（R=d/(σA)，C=ε0εr A/d），颅骨因
 * σ≈0.02 S/m（比脑脊液低 100 倍）成为主衰减层。
 *
 * ⚠️ 组织电参数为文献典型量级（近似值），精确值应从 IT'IS 组织数据库
 *    (https://itis.swiss/virtual-population/tissue-properties/database/)
 *    导出后替换。
 */
#ifndef HBC_CHANNEL_H
#define HBC_CHANNEL_H

#define EPS0 8.8541878128e-12   /* 真空介电常数 F/m */

/* 复数（手动实现，避免依赖 complex.h） */
typedef struct { double re, im; } cpx;

/* 组织层：name 名称，d_mm 厚度(mm)，sigma 电导率(S/m)，epsr 相对介电常数 */
/* ⚠️ sigma/epsr 为固定近似值（向后兼容/对照用）；精确计算用 ITIS_LAYERS 的 Cole-Cole 色散 */
typedef struct {
    const char *name;
    double d_mm;
    double sigma;
    double epsr;
} TissueLayer;

/* 4-Cole-Cole 色散参数（IT'IS V4.2 / Gabriel 1996） */
typedef struct {
    double eps_inf;       /* ε_∞ 高频极限 */
    double de[4];         /* Δε1..4 色散强度 */
    double tau[4];        /* τ1..4 弛豫时间 (s) */
    double alpha[4];      /* α1..4 展宽参数 */
    double sigma_i;       /* 静态离子电导 (S/m) */
} ColeColeParams;

/* 信道增益结果 */
typedef struct {
    double mag;        /* |H| 线性增益 */
    double loss_dB;    /* -20log10|H| 路径损耗(dB) */
    double phase_deg;  /* 相位(度) */
} ChannelResult;

/* 复数运算 */
cpx c_add(cpx a, cpx b);
cpx c_div(cpx a, cpx b);
double c_mag(cpx z);

/* 单层并联 R-C 阻抗：Z = R/(1+jωRC) */
cpx layer_impedance(const TissueLayer *L, double freq_hz, double area_m2);

/* 前向总阻抗（各层串联） */
cpx forward_impedance(const TissueLayer *layers, int n, double freq_hz, double area_m2);

/*
 * 信道复数增益 H(f) = Z_be / (Z_fwd + Z_be)
 *   Z_fwd = 各组织层串联阻抗
 *   Z_be  = 人体-大地返回路径电容 C_body_earth（约 150pF 量级）
 * 返回 |H|、损耗(dB)、相位。
 */
ChannelResult channel_gain(const TissueLayer *layers, int n,
                           double freq_hz, double area_m2, double C_body_earth);

/* 频率扫描：fmin~fmax 取 M 点，输出每个频点的损耗(dB) */
void channel_sweep(const TissueLayer *layers, int n,
                   double fmin, double fmax, int M,
                   double area_m2, double C_body_earth,
                   double *freqs, double *loss_dB);

/* ---- Cole-Cole 色散版（精确，IT'IS V4.2 参数）---- */

/* 某频点的复介电常数 → εr'(f) 和 σ_eff(f)（含位移损耗 + 离子电导） */
void cole_cole_epsigma(const ColeColeParams *p, double freq_hz,
                       double *eps_r_out, double *sigma_eff_out);

/* 频变单层并联 R-C 阻抗（用 Cole-Cole，L 只用 d_mm） */
cpx layer_impedance_cc(const TissueLayer *L, const ColeColeParams *p,
                       double freq_hz, double area_m2);

/* 频变前向总阻抗 */
cpx forward_impedance_cc(const TissueLayer *layers, const ColeColeParams *params,
                         int n, double freq_hz, double area_m2);

/* 频变信道增益（用 Cole-Cole 参数表） */
ChannelResult channel_gain_cc(const TissueLayer *layers, const ColeColeParams *params,
                              int n, double freq_hz, double area_m2, double C_body_earth);

/* 频变频率扫描 */
void channel_sweep_cc(const TissueLayer *layers, const ColeColeParams *params, int n,
                      double fmin, double fmax, int M,
                      double area_m2, double C_body_earth,
                      double *freqs, double *loss_dB);

#endif /* HBC_CHANNEL_H */

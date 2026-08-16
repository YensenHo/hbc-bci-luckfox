/*
 * hbc_channel.c — 分层 R-C 体表信道模型实现
 * 见 hbc_channel.h 说明。
 */
#include <math.h>
#include "hbc_channel.h"
#include "cole_cole_params.h"   /* ITIS_LAYERS[5] — IT'IS V4.2 4-Cole-Cole 参数 */

cpx c_add(cpx a, cpx b) { cpx r = { a.re + b.re, a.im + b.im }; return r; }

cpx c_div(cpx a, cpx b) {
    double d = b.re * b.re + b.im * b.im;
    cpx r = { (a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d };
    return r;
}

double c_mag(cpx z) { return sqrt(z.re * z.re + z.im * z.im); }

/* 单层并联 R-C：R = d/(σA)，C = ε0 εr A/d，Z = R/(1+jωRC) */
cpx layer_impedance(const TissueLayer *L, double freq_hz, double area_m2) {
    double d = L->d_mm * 1e-3;                       /* 厚度 -> m */
    double R = d / (L->sigma * area_m2);             /* 电阻 */
    double C = EPS0 * L->epsr * area_m2 / d;         /* 电容 */
    double w = 2.0 * M_PI * freq_hz;
    double wRC = w * R * C;
    double denom = 1.0 + wRC * wRC;
    cpx z = { R / denom, -R * wRC / denom };         /* R(1-jωRC)/(1+ω²R²C²) */
    return z;
}

cpx forward_impedance(const TissueLayer *layers, int n, double freq_hz, double area_m2) {
    cpx z = { 0.0, 0.0 };
    for (int i = 0; i < n; i++) {
        z = c_add(z, layer_impedance(&layers[i], freq_hz, area_m2));
    }
    return z;
}

ChannelResult channel_gain(const TissueLayer *layers, int n,
                           double freq_hz, double area_m2, double C_body_earth) {
    cpx zf = forward_impedance(layers, n, freq_hz, area_m2);
    double w = 2.0 * M_PI * freq_hz;
    cpx zbe = { 0.0, -1.0 / (w * C_body_earth) };   /* 人体-大地返回路径容抗 */
    cpx h = c_div(zbe, c_add(zf, zbe));
    ChannelResult r;
    r.mag = c_mag(h);
    r.loss_dB = -20.0 * log10(r.mag);
    r.phase_deg = atan2(h.im, h.re) * 180.0 / M_PI;
    return r;
}

void channel_sweep(const TissueLayer *layers, int n,
                   double fmin, double fmax, int M,
                   double area_m2, double C_body_earth,
                   double *freqs, double *loss_dB) {
    for (int i = 0; i < M; i++) {
        double f = fmin * pow(fmax / fmin, (double)i / (M - 1)); /* 对数均匀取点 */
        freqs[i] = f;
        loss_dB[i] = channel_gain(layers, n, f, area_m2, C_body_earth).loss_dB;
    }
}

/* ==================== Cole-Cole 色散版（精确） ==================== */

/* 复介电常数：ε(ω)=ε_∞ + Σ Δε_n/(1+(jωτ_n)^(1-α_n)) + σ_i/(jωε0)
 * 返回 εr'=Re(ε) 和 σ_eff=ωε0·ε''（位移损耗 + 离子电导）。 */
void cole_cole_epsigma(const ColeColeParams *p, double freq_hz,
                       double *eps_r_out, double *sigma_eff_out) {
    double w = 2.0 * M_PI * freq_hz;
    double eps_re = p->eps_inf;
    double eps_disp_im = 0.0;    /* Σ Δε·Im(1/(1+z))，即 ε'' 的色散贡献 */
    for (int n = 0; n < 4; n++) {
        if (p->de[n] == 0.0) continue;
        double wt = w * p->tau[n];
        double wta = pow(wt, 1.0 - p->alpha[n]);            /* (ωτ)^(1-α) */
        double ang = (1.0 - p->alpha[n]) * M_PI / 2.0;
        double re_term = wta * cos(ang);                    /* Re((jωτ)^(1-α)) */
        double im_term = wta * sin(ang);                    /* Im((jωτ)^(1-α)) */
        double denom = (1.0 + re_term) * (1.0 + re_term) + im_term * im_term;
        eps_re += p->de[n] * (1.0 + re_term) / denom;
        eps_disp_im += p->de[n] * im_term / denom;
    }
    double eps_im = eps_disp_im + p->sigma_i / (w * EPS0);  /* ε'' */
    if (eps_r_out) *eps_r_out = eps_re;
    if (sigma_eff_out) *sigma_eff_out = w * EPS0 * eps_im;
}

cpx layer_impedance_cc(const TissueLayer *L, const ColeColeParams *p,
                       double freq_hz, double area_m2) {
    double eps_r, sigma_eff;
    cole_cole_epsigma(p, freq_hz, &eps_r, &sigma_eff);
    double d = L->d_mm * 1e-3;
    double R = d / (sigma_eff * area_m2);
    double C = EPS0 * eps_r * area_m2 / d;
    double w = 2.0 * M_PI * freq_hz;
    double wRC = w * R * C;
    double denom = 1.0 + wRC * wRC;
    cpx z = { R / denom, -R * wRC / denom };
    return z;
}

cpx forward_impedance_cc(const TissueLayer *layers, const ColeColeParams *params,
                         int n, double freq_hz, double area_m2) {
    cpx z = { 0.0, 0.0 };
    for (int i = 0; i < n; i++) {
        z = c_add(z, layer_impedance_cc(&layers[i], &params[i], freq_hz, area_m2));
    }
    return z;
}

ChannelResult channel_gain_cc(const TissueLayer *layers, const ColeColeParams *params,
                              int n, double freq_hz, double area_m2, double C_body_earth) {
    cpx zf = forward_impedance_cc(layers, params, n, freq_hz, area_m2);
    double w = 2.0 * M_PI * freq_hz;
    cpx zbe = { 0.0, -1.0 / (w * C_body_earth) };
    cpx h = c_div(zbe, c_add(zf, zbe));
    ChannelResult r;
    r.mag = c_mag(h);
    r.loss_dB = -20.0 * log10(r.mag);
    r.phase_deg = atan2(h.im, h.re) * 180.0 / M_PI;
    return r;
}

void channel_sweep_cc(const TissueLayer *layers, const ColeColeParams *params, int n,
                      double fmin, double fmax, int M,
                      double area_m2, double C_body_earth,
                      double *freqs, double *loss_dB) {
    for (int i = 0; i < M; i++) {
        double f = fmin * pow(fmax / fmin, (double)i / (M - 1));
        freqs[i] = f;
        loss_dB[i] = channel_gain_cc(layers, params, n, f, area_m2, C_body_earth).loss_dB;
    }
}

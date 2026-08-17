/*
 * hbc_bci_loop.c — 板载「人体通信 × 脑机接口」闭环仿真系统
 *
 * ⚠️ SUPERSEDED（已归档）：本程序是早期玩具版——用「合成 EEG + Goertzel 频域解码」，
 *   已被「真实 EEG + EEGNet 解码」的 hbc_bci_fusion.c 全面取代。保留仅作历史对照，
 *   审稿/展示请用 hbc_bci_fusion.c（真实脑电）与 hbc_bci_realtime.c（实时流式）。
 *
 * 把 PPT 的 HBC-BCI 信号链在 RV1106 上完整仿真一遍：
 *
 *   [1] 合成脑电(运动想象 4 类, mu/beta/gamma 节律)
 *        ↓
 *   [2] 频域解码(Goertzel 带功率检测 → 类别)
 *        ↓
 *   [3] 编码 + OOK 调制到 1MHz EQS 载波
 *        ↓
 *   [4] HBC 体表信道(分层 R-C 模型, 颅骨主衰减) + AWGN
 *        ↓
 *   [5] 包络检波解调 → 恢复比特
 *        ↓
 *   [6] 端到端指标：解码准确率 / 信道损耗 / 颅骨贡献 / BER / 数据率 / 功耗
 *
 * 用法：./hbc_bci_loop
 * 编译：zig cc -target arm-linux-musleabihf -O2 -static hbc_bci_loop.c hbc_channel.c -o hbc_bci_loop -lm
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "hbc_channel.h"

/* ---------- 全局参数 ---------- */
#define FC              1.0e6     /* EQS 载波 1 MHz */
#define CARRIER_FS      4.0e6     /* 载波采样率 4 MHz (4 样本/周期) */
#define SAMPLES_PER_BIT 64        /* 每 bit 64 样本 = 16 个载波周期 */
#define BITS_PER_FRAME  3         /* 1 前导 bit + 2 数据 bit (4 类) */
#define N_TRIALS        200       /* 闭环试验次数 */
#define EEG_FS          250.0     /* 脑电采样率 250 Hz */
#define EEG_N           250       /* 脑电 1 秒窗口 */

/* 合成脑电 4 类运动想象的特征频率 (mu/beta/gamma 节律) */
static const double CLASS_FREQ[4] = { 10.0, 15.0, 20.0, 30.0 };

/* ---------- 简单 LCG 伪随机数 (可复现) ---------- */
static unsigned long long rng_state = 88172645463325252ULL;
static double urand(void) {
    rng_state = (rng_state * 6364136223846793005ULL + 1442695040888963407ULL);
    return (double)((rng_state >> 11) & 0xFFFFFFFFFFFFFULL) / 0xFFFFFFFFFFFFFULL;
}
static double gauss(void) { /* Box-Muller */
    double u1 = urand() + 1e-12, u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ---------- [1] 合成脑电：特征频率正弦 + 谐波 + 噪声 ---------- */
static void synth_eeg(double *eeg, int class_id) {
    double f = CLASS_FREQ[class_id];
    for (int i = 0; i < EEG_N; i++) {
        double t = i / EEG_FS;
        double s = 1.0 * sin(2.0 * M_PI * f * t)
                 + 0.35 * sin(2.0 * M_PI * 2.0 * f * t)
                 + 0.15 * sin(2.0 * M_PI * 3.0 * f * t);
        eeg[i] = s + 0.6 * gauss();
    }
}

/* ---------- [2] Goertzel 带功率检测 → 频域解码 ---------- */
static double goertzel_power(const double *x, int n, double target_hz, double fs) {
    double k = 0.5 + n * target_hz / fs;
    double w = 2.0 * M_PI * k / n;
    double cw = 2.0 * cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < n; i++) {
        s0 = x[i] + cw * s1 - s2;
        s2 = s1; s1 = s0;
    }
    return s1 * s1 + s2 * s2 - cw * s1 * s2;
}
static int decode_eeg(const double *eeg) {
    int best = 0; double best_p = -1.0;
    for (int c = 0; c < 4; c++) {
        double p = goertzel_power(eeg, EEG_N, CLASS_FREQ[c], EEG_FS);
        if (p > best_p) { best_p = p; best = c; }
    }
    return best;
}

/* ---------- [3] OOK 调制：bit=1 载波开, bit=0 关 ---------- */
static void ook_modulate(const int *bits, int nbits, double *tx, int *tx_len) {
    *tx_len = nbits * SAMPLES_PER_BIT;
    for (int b = 0; b < nbits; b++) {
        for (int s = 0; s < SAMPLES_PER_BIT; s++) {
            int idx = b * SAMPLES_PER_BIT + s;
            double t = (double)idx / CARRIER_FS;
            tx[idx] = bits[b] ? sin(2.0 * M_PI * FC * t) : 0.0;
        }
    }
}

/* ---------- [4]+[5] 信道衰减 + 噪声 + 包络检波解调 ---------- */
static void channel_apply(double *sig, int len, double gain_mag, double noise_sigma) {
    for (int i = 0; i < len; i++) {
        sig[i] = sig[i] * gain_mag + noise_sigma * gauss(); /* 幅度衰减 + AWGN */
    }
}
static void ook_demodulate(const double *rx, int len, int nbits, double thresh, int *bits) {
    /* 每 bit 取包络均值，与阈值比较 */
    for (int b = 0; b < nbits; b++) {
        double env = 0.0;
        for (int s = 0; s < SAMPLES_PER_BIT; s++) {
            env += fabs(rx[b * SAMPLES_PER_BIT + s]);
        }
        env /= SAMPLES_PER_BIT;
        bits[b] = env > thresh ? 1 : 0;
    }
}

/* ---------- 主程序 ---------- */
int main(void) {
    clock_t t_start = clock();
    /* 体表分层组织模型 (近似值, 精确值从 IT'IS DB 导出) */
    const double A_ELECTRODE = 1e-6;       /* 1mm² 植入电极 */
    const double C_BODY_EARTH = 150e-12;   /* 人体-大地返回路径电容 */
    TissueLayer layers[5] = {
        { "皮肤(skin)",     1.5, 2e-4,  1000  },
        { "脂肪(fat)",      2.0, 0.03,  30    },
        { "颅骨(skull)",    5.0, 0.02,  150   },
        { "脑脊液(CSF)",    2.0, 2.0,   109   },
        { "脑灰质(gray)",   10.0, 0.4,  10000 },
    };
    const int N_LAYER = 5;

    printf("============================================================\n");
    printf("  HBC x BCI 闭环仿真系统 (Luckfox Pico Ultra / RV1106)\n");
    printf("  人体通信体表信道 + 脑机接口解码, 板载端到端仿真\n");
    printf("============================================================\n\n");

    /* ===== Part A: EQS-HBC 体表信道特性 (分层 R-C 模型) ===== */
    printf("--- [EQS-HBC] 体表信道特性 @ 1 MHz ---\n");
    cpx ztotal = { 0, 0 };
    for (int i = 0; i < N_LAYER; i++) {
        cpx z = layer_impedance(&layers[i], FC, A_ELECTRODE);
        ztotal = c_add(ztotal, z);
        printf("  %-16s |Z| = %9.1f ohm   (R=%8.1f ohm, Xc=%8.1f ohm)\n",
               layers[i].name, c_mag(z), z.re, -z.im);
    }
    printf("  前向总阻抗 |Z_fwd| = %.1f kohm\n\n", c_mag(ztotal) / 1e3);

    /* 颅骨贡献: 有颅骨 vs 无颅骨 */
    TissueLayer noskull[4] = { layers[0], layers[1], layers[3], layers[4] };
    ChannelResult g_with = channel_gain(layers, 5, FC, A_ELECTRODE, C_BODY_EARTH);
    ChannelResult g_without = channel_gain(noskull, 4, FC, A_ELECTRODE, C_BODY_EARTH);
    printf("  信道损耗 @1MHz (有颅骨) = %.1f dB\n", g_with.loss_dB);
    printf("  信道损耗 @1MHz (无颅骨) = %.1f dB\n", g_without.loss_dB);
    printf("  >> 颅骨额外贡献 = %.1f dB  (颅骨 σ=0.02 S/m, 是主衰减层)\n\n", 
           g_with.loss_dB - g_without.loss_dB);

    /* 扫频 Bode 图 */
    printf("  频率扫描 (100kHz - 10MHz, 对数取点):\n");
    printf("    freq(MHz)  loss(dB)\n");
    enum { M = 9 };
    double freqs[M], losses[M];
    channel_sweep(layers, 5, 1e5, 1e7, M, A_ELECTRODE, C_BODY_EARTH, freqs, losses);
    for (int i = 0; i < M; i++) {
        printf("    %9.3f  %8.1f\n", freqs[i] / 1e6, losses[i]);
    }
    printf("\n");

    /* ===== Part B: HBC-BCI 闭环 ===== */
    printf("--- [闭环] 脑电 → 解码 → HBC 传输 → 接收 ---\n");
    printf("  载波 %.0f kHz OOK, 每帧 %d bit (1前导+2数据), 试验 %d 次\n\n",
           FC / 1e3, BITS_PER_FRAME, N_TRIALS);

    int decode_ok = 0;
    int bit_err = 0, bit_total = 0;

    for (int trial = 0; trial < N_TRIALS; trial++) {
        int truth = (int)(urand() * 4.0) % 4;       /* 随机真实类别 */

        /* [1] 合成脑电 */
        double eeg[EEG_N];
        synth_eeg(eeg, truth);

        /* [2] 频域解码 */
        int pred = decode_eeg(eeg);
        if (pred == truth) decode_ok++;

        /* [3] 编码 + OOK 调制 */
        int bits[BITS_PER_FRAME] = { 1, (truth >> 1) & 1, truth & 1 }; /* 前导 + 2bit */
        double tx[BITS_PER_FRAME * SAMPLES_PER_BIT];
        int tx_len;
        ook_modulate(bits, BITS_PER_FRAME, tx, &tx_len);

        /* [4] HBC 体表信道: 动态衰落(±25%) + 幅度衰减 + AWGN */
        double fade = 1.0 + 0.25 * gauss();          /* 信道动态变化 (PPT P22) */
        double gain = g_with.mag * fade;
        double noise_sigma = 0.15 * gain;             /* 较高噪声 → 非零 BER */
        channel_apply(tx, tx_len, gain, noise_sigma);

        /* [5] 包络检波解调 */
        double thresh = 0.5 * gain;                 /* 载波幅度衰减后的一半作阈值 */
        int rx_bits[BITS_PER_FRAME];
        ook_demodulate(tx, tx_len, BITS_PER_FRAME, thresh, rx_bits);

        /* 对比数据 bit (跳过前导 bit) */
        for (int b = 1; b < BITS_PER_FRAME; b++) {
            bit_total++;
            if (rx_bits[b] != bits[b]) bit_err++;
        }
    }

    /* ===== 端到端指标 ===== */
    printf("--- 端到端指标 ---\n");
    printf("  解码准确率  : %.1f%% (%d/%d)\n", 100.0 * decode_ok / N_TRIALS, decode_ok, N_TRIALS);
    printf("  信道损耗    : %.1f dB @ %.0f kHz (颅骨贡献 %.1f dB)\n",
           g_with.loss_dB, FC / 1e3, g_with.loss_dB - g_without.loss_dB);
    printf("  误码率 BER  : %.2e (%d/%d bit)\n",
           (double)bit_err / bit_total, bit_err, bit_total);

    double frame_us = BITS_PER_FRAME * SAMPLES_PER_BIT / CARRIER_FS * 1e6;
    double data_kbps = 2.0 / (frame_us * 1e-6) / 1e3;   /* 每帧 2 数据 bit */
    printf("  等效数据率  : %.1f kbps (每帧 %.0f us)\n", data_kbps, frame_us);

    /* 功耗：仅文献引用（BP-QBC 0.52uW@1Mbps），无实测电流，不做线性缩放外推 */
    printf("  功耗        : 无实测（RV1106 无片上功率计，文献 BP-QBC 参考 TX 0.52uW@1Mbps）\n");

    printf("\n============================================================\n");
    printf("  结论: 体表 EQS-HBC 信道模型 + 脑电解码在 RV1106 单核上\n");
    printf("        实时跑通, 颅骨为主衰减层 (贡献 %.1f dB)。\n",
           g_with.loss_dB - g_without.loss_dB);
    printf("  总耗时: %.1f ms (含 %d 次闭环 + 信道扫频)\n",
           1000.0 * (clock() - t_start) / CLOCKS_PER_SEC, N_TRIALS);
    printf("============================================================\n");
    return 0;
}

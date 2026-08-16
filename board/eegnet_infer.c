/*
 * eegnet_infer.c — EEGNet 8ch 二分类前向推理的纯 C 实现（CPU 版）
 *
 * 与 PyTorch EEGNet（model/eegnet.py）逐层对应，BN 已折叠进卷积（见 export_c_weights.py）。
 * 输入 (1,1,8,500)，输出 2 类 logits。无外部依赖，仅需 math.h。
 *
 * 结构：
 *   conv1(1→8, 1×63, pad L31/R31) → depthwise(8→16, 8×1, groups8)
 *   → ELU → AvgPool(1,4) → sep_depthwise(16→16, 1×15, pad L7/R7)
 *   → pointwise(16→16) → ELU → AvgPool(1,8) → FC(240→2)
 *
 * 用法（板载单测）：
 *   zig cc -target arm-linux-musleabihf -O2 -static eegnet_infer.c -o eegnet_infer -lm
 */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "eegnet_weights.h"

#define NCH  EEGNET_N_CH        /* 8 */
#define T    EEGNET_T           /* 500 */
#define F1   EEGNET_F1          /* 8 */
#define F2   EEGNET_F2          /* 16 */
#define TOUT EEGNET_T_OUT       /* 15 = 500//32 */

#define T4  (T / 4)             /* 125 */
#define T32 (T / 32)            /* 15 */

static inline float elu(float x) { return x > 0.0f ? x : expf(x) - 1.0f; }

/*
 * eegnet_infer：输入 x[NCH*T]（通道优先，逐 500 样本），输出 logits[2]。
 * 返回预测类别（0/1）。
 */
int eegnet_infer(const float *x, float *logits) {
    /* ---- Block 1 ---- */
    /* conv1: 1→F1, kernel (1,63), pad 左31 右31（对称，奇数核） */
    static float c1[F1][NCH][T];
    for (int f = 0; f < F1; f++) {
        for (int c = 0; c < NCH; c++) {
            for (int t = 0; t < T; t++) {
                float s = conv1_b[f];
                for (int k = 0; k < 63; k++) {
                    int idx = t + k - 31;              /* 左 pad 31 */
                    if (idx >= 0 && idx < T) {
                        s += conv1_w[(f * 1 * 1 + 0) * 63 + k] * x[c * T + idx];
                    }
                }
                c1[f][c][t] = s;
            }
        }
    }

    /* depthwise: F1→2*F1, kernel (NCH,1), groups=F1（每个输入通道→2 输出，空间 8 通道折叠） */
    static float dw[F2][T];
    for (int o = 0; o < F2; o++) {
        int g = o / 2;                                  /* 输入通道 = 输出//2 */
        for (int t = 0; t < T; t++) {
            float s = depthwise_b[o];
            for (int c = 0; c < NCH; c++) {
                s += depthwise_w[(o * 1 * 1 + 0) * NCH + c] * c1[g][c][t];
            }
            dw[o][t] = s;
        }
    }

    /* ELU + AvgPool(1,4) → (F2, T4) */
    static float p1[F2][T4];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T4; t++) {
            float acc = 0.0f;
            for (int k = 0; k < 4; k++) acc += elu(dw[o][t * 4 + k]);
            p1[o][t] = acc / 4.0f;
        }
    }

    /* ---- Block 2 ---- */
    /* sep_depthwise: F2→F2, kernel (1,15), pad 左7 右7（对称，奇数核） */
    static float sdw[F2][T4];
    for (int c = 0; c < F2; c++) {
        for (int t = 0; t < T4; t++) {
            float s = 0.0f;
            for (int k = 0; k < 15; k++) {
                int idx = t + k - 7;                    /* 左 pad 7 */
                if (idx >= 0 && idx < T4) {
                    s += sep_dw_w[(c * 1 * 1 + 0) * 15 + k] * p1[c][idx];
                }
            }
            sdw[c][t] = s;
        }
    }

    /* pointwise: F2→F2, (1,1) */
    static float pw[F2][T4];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T4; t++) {
            float s = pointwise_b[o];
            for (int c = 0; c < F2; c++) {
                s += pointwise_w[(o * F2 + c) * 1 * 1] * sdw[c][t];
            }
            pw[o][t] = s;
        }
    }

    /* ELU + AvgPool(1,8) → (F2, T32) */
    static float p2[F2][T32];
    for (int o = 0; o < F2; o++) {
        for (int t = 0; t < T32; t++) {
            float acc = 0.0f;
            for (int k = 0; k < 8; k++) acc += elu(pw[o][t * 8 + k]);
            p2[o][t] = acc / 8.0f;
        }
    }

    /* ---- FC: F2*T32 → 2 ---- */
    for (int o = 0; o < 2; o++) {
        float s = fc_b[o];
        for (int c = 0; c < F2; c++) {
            for (int t = 0; t < T32; t++) {
                s += fc_w[o * (F2 * T32) + c * T32 + t] * p2[c][t];
            }
        }
        logits[o] = s;
    }
    return logits[0] >= logits[1] ? 0 : 1;
}

#ifdef EEGNET_INFER_MAIN
/* 单测：随机输入，跑 100 次计时 */
int main(void) {
    static float x[NCH * T];
    for (int i = 0; i < NCH * T; i++) x[i] = (float)(i % 7) / 7.0f - 0.5f;
    float logits[2];
    clock_t t0 = clock();
    for (int n = 0; n < 100; n++) eegnet_infer(x, logits);
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC / 100.0;
    printf("预测类别: %d, logits=[%.3f, %.3f]\n", logits[0] >= logits[1] ? 0 : 1, logits[0], logits[1]);
    printf("单次前向耗时: %.2f ms\n", ms);
    return 0;
}
#endif

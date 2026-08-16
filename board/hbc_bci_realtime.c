/*
 * hbc_bci_realtime.c — 实时流式演示：板载逐窗口「采集→解码→HBC传输→收端解码」
 *
 * 回放真实脑电（BCIC IV 2a 评估集），逐个 2s 窗口实时处理并流式输出结果，
 * 演示「端侧芯片实时完成 HBC×BCI 融合」。核心指标：单窗口处理时间 << 2s 窗口。
 *
 * 用法：
 *   ./hbc_bci_realtime /root/fusion_test.bin [循环次数]
 *   编译：zig cc -target arm-linux-musleabihf -O2 -static \
 *           hbc_bci_realtime.c hbc_channel.c eegnet_infer.c -o hbc_bci_realtime -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "hbc_channel.h"
#include "cole_cole_params.h"

int eegnet_infer(const float *x, float *logits);

#define NCH 8
#define T   500
#define NSAMP (NCH * T)
#define QBITS 8
#define QRANGE 4.0f
#define QLEVELS (1 << QBITS)

static unsigned long long rng_state = 88172645463325252ULL;
static double urand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((rng_state >> 11) & 0xFFFFFFFFFFFFFULL) / 0xFFFFFFFFFFFFFULL;
}
static double gauss(void) {
    double u1 = urand() + 1e-12, u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}
static unsigned char quantize(float x) {
    int q = (int)lrintf((x + QRANGE) / (2.0f * QRANGE) * (QLEVELS - 1));
    if (q < 0) q = 0;
    if (q >= QLEVELS) q = QLEVELS - 1;
    return (unsigned char)q;
}
static float dequantize(unsigned char q) {
    return (q / (float)(QLEVELS - 1)) * 2.0f * QRANGE - QRANGE;
}

static const char *LABEL_CN[2] = { "左手", "右手" };

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: %s <fusion_test.bin> [循环次数]\n", argv[0]); return 1; }
    int loops = (argc > 2) ? atoi(argv[2]) : 1;

    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", argv[1]); return 1; }
    int N;
    if (fread(&N, 4, 1, f) != 1) { fprintf(stderr, "读取 N 失败\n"); return 1; }

    /* 读全部窗口进内存（回放用） */
    float *X = (float *)malloc((size_t)N * NSAMP * sizeof(float));
    int *Y = (int *)malloc((size_t)N * sizeof(int));
    for (int n = 0; n < N; n++) {
        fread(&X[(size_t)n * NSAMP], 4, NSAMP, f);
        fread(&Y[n], 4, 1, f);
    }
    fclose(f);

    /* HBC 信道（IT'IS Cole-Cole + 105pF 返回路径，1MHz） */
    const double A_ELEC = 1e-6, C_BE = 105e-12;
    TissueLayer layers[5] = {
        { "皮肤", 1.5, 0, 0 }, { "脂肪", 2.0, 0, 0 }, { "颅骨", 5.0, 0, 0 },
        { "脑脊液", 2.0, 0, 0 }, { "脑灰质", 10.0, 0, 0 },
    };
    ChannelResult g = channel_gain_cc(layers, ITIS_LAYERS, 5, 1.0e6, A_ELEC, C_BE);
    double H = g.mag, noise_sigma = 0.1 * H, thresh = 0.5 * H;

    printf("=== 实时流式演示：RV1106 端侧 HBC×BCI 融合（%d 窗口 × %d 轮）===\n", N, loops);
    printf("体表信道 %.1f dB @1MHz | 单窗口=2s 脑电\n\n", g.loss_dB);

    int total = 0, direct_ok = 0, hbc_ok = 0, consistent = 0;
    double sum_ms = 0.0;

    for (int loop = 0; loop < loops; loop++) {
        for (int n = 0; n < N; n++) {
            float *x = &X[(size_t)n * NSAMP];
            int y = Y[n];

            clock_t t0 = clock();

            /* 1. 片上解码（直连基线） */
            float logits_d[2];
            int pred_d = eegnet_infer(x, logits_d);

            /* 2. 量化 → OOK → HBC 信道 → 解调 → 反量化 */
            float x_rec[NSAMP];
            for (int s = 0; s < NSAMP; s++) {
                unsigned char q = quantize(x[s]);
                unsigned char q_rec = 0;
                for (int b = 0; b < QBITS; b++) {
                    int bit = (q >> (QBITS - 1 - b)) & 1;
                    double tx = bit ? 1.0 : 0.0;
                    double rx = tx * H + noise_sigma * gauss();
                    int bit_rec = rx > thresh ? 1 : 0;
                    q_rec = (q_rec << 1) | bit_rec;
                }
                x_rec[s] = dequantize(q_rec);
            }

            /* 3. 收端解码 */
            float logits_h[2];
            int pred_h = eegnet_infer(x_rec, logits_h);

            double dt = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;
            sum_ms += dt;
            total++;
            if (pred_d == y) direct_ok++;
            if (pred_h == y) hbc_ok++;
            if (pred_h == pred_d) consistent++;

            /* 流式输出每窗口结果 */
            printf("[%3d] 真值=%s 直连=%s 经HBC=%s %s | %.1fms (实时余量 %.0f%%)\n",
                   n, LABEL_CN[y], LABEL_CN[pred_d], LABEL_CN[pred_h],
                   (pred_h == y) ? "✓" : "✗", dt, 100.0 * (1.0 - dt / 2000.0));
        }
    }

    printf("\n=== 汇总（%d 窗口）===\n", total);
    printf("  直连准确率   : %.1f%% (%d/%d)\n", 100.0 * direct_ok / total, direct_ok, total);
    printf("  经HBC准确率  : %.1f%% (%d/%d)\n", 100.0 * hbc_ok / total, hbc_ok, total);
    printf("  两者一致率   : %.1f%% (%d/%d)\n", 100.0 * consistent / total, consistent, total);
    printf("  平均单窗口   : %.1f ms（2s 窗口，实时余量 %.0f×）\n",
           sum_ms / total, 2000.0 / (sum_ms / total));
    free(X); free(Y);
    return 0;
}

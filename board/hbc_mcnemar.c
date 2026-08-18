/*
 * hbc_mcnemar.c — 全量 1296 试次配对预测 + McNemar 列联表
 *
 * 直连 vs 经HBC 的配对预测（同一试次两种方式），输出 2×2 列联表：
 *   a = 都对，b = 直连对/经HBC错，c = 直连错/经HBC对，d = 都错
 * Python 侧用 b/c 算 McNemar 卡方 → p 值，检验「直连 vs 经HBC 是否显著差异」。
 *
 * 用法：./hbc_mcnemar /root/fusion_test_full.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hbc_channel.h"
#include "cole_cole_params.h"

int eegnet_infer(const float *x, float *logits);

#define NCH 8
#define T 500
#define NSAMP (NCH * T)
#define QBITS 5
#define QRANGE 4.0f
#define QLEVELS (1 << QBITS)
#define N_SEED 5

static unsigned long long rng_state;
static void srand_lcg(unsigned long long s) { rng_state = s; }
static double urand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((rng_state >> 11) & 0xFFFFFFFFFFFFFULL) / 0xFFFFFFFFFFFFFULL;
}
static double gauss(void) {
    double u1 = urand() + 1e-12, u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: %s <fusion_test_full.bin>\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    int N;
    if (fread(&N, 4, 1, f) != 1) return 1;
    float *X = (float *)malloc((size_t)N * NSAMP * sizeof(float));
    int *Y = (int *)malloc((size_t)N * sizeof(int));
    for (int n = 0; n < N; n++) { fread(&X[(size_t)n * NSAMP], 4, NSAMP, f); fread(&Y[n], 4, 1, f); }
    fclose(f);

    const double A_ELEC = 1e-6, C_BE = 105e-12;
    TissueLayer layers[5] = {
        { "皮肤", 1.5, 0, 0 }, { "脂肪", 2.0, 0, 0 }, { "颅骨", 5.0, 0, 0 },
        { "脑脊液", 2.0, 0, 0 }, { "脑灰质", 10.0, 0, 0 },
    };
    ChannelResult g = channel_gain_cc(layers, ITIS_LAYERS, 5, 1.0e6, A_ELEC, C_BE);
    double H = g.mag, sigma = 0.1 * H, thresh = 0.5 * H;

    const unsigned long long seeds[N_SEED] = {
        88172645463325252ULL, 12345678901234567ULL, 98765432109876543ULL,
        11112222333344445ULL, 55556666777788889ULL,
    };

    long a = 0, b = 0, c = 0, d = 0;   /* 都对/直对H错/直错H对/都错 */
    long direct_ok = 0, hbc_ok = 0;

    printf("=== 直连 vs 经HBC 配对预测（%d 试次 × %d 种子）===\n", N, N_SEED);
    printf("噪声 σ/H=0.1，位深 %dbit，信道 %.1f dB @1MHz\n\n", QBITS, g.loss_dB);

    for (int k = 0; k < N_SEED; k++) {
        srand_lcg(seeds[k]);
        for (int n = 0; n < N; n++) {
            float *x = &X[(size_t)n * NSAMP];
            int y = Y[n];

            float ld[2]; int pd = eegnet_infer(x, ld);
            if (pd == y) direct_ok++;

            float x_rec[NSAMP];
            for (int i = 0; i < NSAMP; i++) {
                int q = (int)lrintf((x[i] + QRANGE) / (2.0f * QRANGE) * (QLEVELS - 1));
                if (q < 0) q = 0;
                if (q >= QLEVELS) q = QLEVELS - 1;
                int q_rec = 0;
                for (int j = 0; j < QBITS; j++) {
                    int bit = (q >> (QBITS - 1 - j)) & 1;
                    double rx = (bit ? 1.0 : 0.0) * H + sigma * gauss();
                    q_rec = (q_rec << 1) | (rx > thresh ? 1 : 0);
                }
                x_rec[i] = (q_rec / (float)(QLEVELS - 1)) * 2.0f * QRANGE - QRANGE;
            }
            float lh[2]; int ph = eegnet_infer(x_rec, lh);
            if (ph == y) hbc_ok++;

            int dc = (pd == y), hc = (ph == y);
            if (dc && hc) a++;
            else if (dc && !hc) b++;
            else if (!dc && hc) c++;
            else d++;
        }
    }

    printf("直连准确率  : %.2f%% (%ld/%ld)\n", 100.0 * direct_ok / (N * (long)N_SEED), direct_ok, (long)N * N_SEED);
    printf("经HBC准确率 : %.2f%% (%ld/%ld)\n", 100.0 * hbc_ok / (N * (long)N_SEED), hbc_ok, (long)N * N_SEED);
    printf("\n配对 2×2 列联表（跨 %d 种子汇总）：\n", N_SEED);
    printf("                      经HBC对    经HBC错\n");
    printf("  直连对              %7ld   %7ld\n", a, b);
    printf("  直连错              %7ld   %7ld\n", c, d);
    printf("\n  b(直对H错)=%ld  c(直错H对)=%ld  → McNemar 由 Python 算 p 值\n", b, c);
    free(X); free(Y);
    return 0;
}

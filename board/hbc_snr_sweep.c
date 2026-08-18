/*
 * hbc_snr_sweep.c — 片上 SNR→准确率/BER 瀑布曲线（多种子 Monte Carlo）
 *
 * 把 fusion.c 里固定的 noise_sigma=0.1*H 改成外循环扫描 σ/H ∈ {0.05..2.0}，
 * 每个 SNR 点跑多种子（5 个 LCG 种子）× 全 1296 试次，
 * 输出「SNR(dB) → 经HBC准确率(均值±跨种子范围) + BER」瀑布表，
 * 定位准确率从直连水平掉到随机(50%)的 SNR 拐点。
 *
 * 用法：./hbc_snr_sweep /root/fusion_test_full.bin
 * 编译：zig cc -target arm-linux-musleabihf -O2 -static \
 *         hbc_snr_sweep.c hbc_channel.c eegnet_infer.c -o hbc_snr_sweep -lm
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
#define QBITS 5          /* 位深扫描结论：5 bit 保精度 */
#define QRANGE 4.0f
#define QLEVELS (1 << QBITS)

#define N_SNR 7          /* SNR 扫描点数 */
#define N_SEED 5         /* Monte Carlo 种子数 */

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
    if (!f) { fprintf(stderr, "打不开 %s\n", argv[1]); return 1; }
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
    double H = g.mag, thresh = 0.5 * H;

    /* 直连基线（无信道，只解码） */
    int direct_ok = 0;
    for (int n = 0; n < N; n++) {
        float logits[2];
        if (eegnet_infer(&X[(size_t)n * NSAMP], logits) == Y[n]) direct_ok++;
    }

    printf("=== 片上 SNR→准确率/BER 瀑布曲线（多种子 Monte Carlo）===\n");
    printf("体表信道 %.1f dB @1MHz | %d 试次 | %d 种子 | 位深 %dbit\n",
           g.loss_dB, N, N_SEED, QBITS);
    printf("直连基线准确率: %.2f%% (%d/%d)\n\n", 100.0 * direct_ok / N, direct_ok, N);
    printf(" SNR(dB) | σ/H    | 经HBC准确率(均值) | 跨种子范围 |    BER(均值)\n");
    printf("---------|--------|-------------------|------------|--------------\n");

    const double sigma_ratios[N_SNR] = { 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0 };
    const unsigned long long seeds[N_SEED] = {
        88172645463325252ULL, 12345678901234567ULL, 98765432109876543ULL,
        11112222333344445ULL, 55556666777788889ULL,
    };

    for (int s = 0; s < N_SNR; s++) {
        double sr = sigma_ratios[s];
        double snr_db = 20.0 * log10(1.0 / sr);
        double acc_sum = 0, acc_min = 1e9, acc_max = -1e9, ber_sum = 0;

        for (int k = 0; k < N_SEED; k++) {
            srand_lcg(seeds[k]);
            double sigma = sr * H;
            long bit_err = 0, bit_total = 0;
            int hbc_ok = 0;

            for (int n = 0; n < N; n++) {
                float *x = &X[(size_t)n * NSAMP];
                int y = Y[n];
                float x_rec[NSAMP];
                for (int i = 0; i < NSAMP; i++) {
                    int q = (int)lrintf((x[i] + QRANGE) / (2.0f * QRANGE) * (QLEVELS - 1));
                    if (q < 0) q = 0;
                    if (q >= QLEVELS) q = QLEVELS - 1;
                    int q_rec = 0;
                    for (int b = 0; b < QBITS; b++) {
                        int bit = (q >> (QBITS - 1 - b)) & 1;
                        double rx = (bit ? 1.0 : 0.0) * H + sigma * gauss();
                        int bit_rec = rx > thresh ? 1 : 0;
                        if (bit_rec != bit) bit_err++;
                        bit_total++;
                        q_rec = (q_rec << 1) | bit_rec;
                    }
                    x_rec[i] = (q_rec / (float)(QLEVELS - 1)) * 2.0f * QRANGE - QRANGE;
                }
                float logits[2];
                if (eegnet_infer(x_rec, logits) == y) hbc_ok++;
            }
            double acc = 100.0 * hbc_ok / N;
            acc_sum += acc;
            if (acc < acc_min) acc_min = acc;
            if (acc > acc_max) acc_max = acc;
            ber_sum += (double)bit_err / bit_total;
        }

        printf(" %6.1f  | %.3f  |  %.2f%%           | %.2f~%.2f%%  |  %.2e\n",
               snr_db, sr, acc_sum / N_SEED, acc_min, acc_max, ber_sum / N_SEED);
    }
    printf("\n（SNR=20*log10(1/(σ/H))；准确率掉到 50% 即解码失效）\n");
    free(X); free(Y);
    return 0;
}

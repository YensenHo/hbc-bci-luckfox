/*
 * hbc_bitdepth_scan.c — 位深扫描：QBITS 3→12，找「保精度最小量化位深」
 *
 * 把单点 BER 变成曲线：对每个量化位深，报告
 *   - 量化 MSE（重建脑电 vs 原始）
 *   - 误码率 BER（经 HBC 信道后）
 *   - 直连 vs 经 HBC 解码准确率
 *
 * 用法：./hbc_bitdepth_scan /root/fusion_test.bin
 * 编译：zig cc -target arm-linux-musleabihf -O2 -static \
 *         hbc_bitdepth_scan.c hbc_channel.c eegnet_infer.c -o hbc_bitdepth_scan -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "hbc_channel.h"
#include "cole_cole_params.h"

int eegnet_infer(const float *x, float *logits);

#define NCH 8
#define T 500
#define NSAMP (NCH * T)
#define QRANGE 4.0f

static unsigned long long rng_state = 88172645463325252ULL;
static double urand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((rng_state >> 11) & 0xFFFFFFFFFFFFFULL) / 0xFFFFFFFFFFFFFULL;
}
static double gauss(void) {
    double u1 = urand() + 1e-12, u2 = urand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: %s <fusion_test.bin>\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", argv[1]); return 1; }
    int N;
    if (fread(&N, 4, 1, f) != 1) return 1;
    float *X = (float *)malloc((size_t)N * NSAMP * sizeof(float));
    int *Y = (int *)malloc((size_t)N * sizeof(int));
    for (int n = 0; n < N; n++) {
        fread(&X[(size_t)n * NSAMP], 4, NSAMP, f);
        fread(&Y[n], 4, 1, f);
    }
    fclose(f);

    const double A_ELEC = 1e-6, C_BE = 105e-12;
    TissueLayer layers[5] = {
        { "皮肤", 1.5, 0, 0 }, { "脂肪", 2.0, 0, 0 }, { "颅骨", 5.0, 0, 0 },
        { "脑脊液", 2.0, 0, 0 }, { "脑灰质", 10.0, 0, 0 },
    };
    ChannelResult g = channel_gain_cc(layers, ITIS_LAYERS, 5, 1.0e6, A_ELEC, C_BE);
    double H = g.mag, noise_sigma = 0.1 * H, thresh = 0.5 * H;

    printf("=== 位深扫描（HBC 传输神经信号，QBITS 3→12）===\n");
    printf("体表信道 %.1f dB @1MHz | %d 试次\n", g.loss_dB, N);
    printf("------------------------------------------------------------\n");
    printf(" QBITS | 量化MSE  |   BER      | 直连准确率 | 经HBC准确率 | 下降\n");
    printf("------------------------------------------------------------\n");

    for (int qbits = 3; qbits <= 12; qbits++) {
        int qlevels = 1 << qbits;
        int direct_ok = 0, hbc_ok = 0;
        long bit_err = 0, bit_total = 0;
        double mse_sum = 0.0;

        for (int n = 0; n < N; n++) {
            float *x = &X[(size_t)n * NSAMP];
            int y = Y[n];

            float logits_d[2];
            int pred_d = eegnet_infer(x, logits_d);
            if (pred_d == y) direct_ok++;

            float x_rec[NSAMP];
            for (int s = 0; s < NSAMP; s++) {
                int q = (int)lrintf((x[s] + QRANGE) / (2.0f * QRANGE) * (qlevels - 1));
                if (q < 0) q = 0;
                if (q >= qlevels) q = qlevels - 1;
                int q_rec = 0;
                for (int b = 0; b < qbits; b++) {
                    int bit = (q >> (qbits - 1 - b)) & 1;
                    double tx = bit ? 1.0 : 0.0;
                    double rx = tx * H + noise_sigma * gauss();
                    int bit_rec = rx > thresh ? 1 : 0;
                    if (bit_rec != bit) bit_err++;
                    bit_total++;
                    q_rec = (q_rec << 1) | bit_rec;
                }
                x_rec[s] = (q_rec / (float)(qlevels - 1)) * 2.0f * QRANGE - QRANGE;
                double e = x[s] - x_rec[s];
                mse_sum += e * e;
            }

            float logits_h[2];
            int pred_h = eegnet_infer(x_rec, logits_h);
            if (pred_h == y) hbc_ok++;
        }

        printf("  %2d   | %.6f | %.2e |  %.1f%%    |  %.1f%%    | %.2fpp\n",
               qbits, mse_sum / (N * NSAMP), (double)bit_err / bit_total,
               100.0 * direct_ok / N, 100.0 * hbc_ok / N,
               100.0 * (direct_ok - hbc_ok) / N);
    }
    printf("------------------------------------------------------------\n");
    free(X); free(Y);
    return 0;
}

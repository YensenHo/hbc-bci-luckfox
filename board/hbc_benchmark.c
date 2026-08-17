/*
 * hbc_benchmark.c — 端到端延迟分布（墙钟）+ 资源占用基准
 *
 * 跑 N 窗口完整闭环（解码→量化→OOK→信道→解调→再解码），
 * 用 clock_gettime(CLOCK_MONOTONIC) 记录每个窗口墙钟延迟，报 p50/p99/max 分布；
 * 用 getrusage 报峰值 RSS（内存占用）。
 *
 * 用法：./hbc_benchmark /root/fusion_test.bin [循环次数]
 * 编译：zig cc -target arm-linux-musleabihf -O2 -static \
 *         hbc_benchmark.c hbc_channel.c eegnet_infer.c -o hbc_benchmark -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include "hbc_channel.h"
#include "cole_cole_params.h"

int eegnet_infer(const float *x, float *logits);

#define NCH 8
#define T 500
#define NSAMP (NCH * T)
#define QBITS 5      /* 位深扫描结论：5 bit 即保精度 */
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

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: %s <fusion_test.bin> [循环]\n", argv[0]); return 1; }
    int loops = (argc > 2) ? atoi(argv[2]) : 1;
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
    double H = g.mag, noise_sigma = 0.1 * H, thresh = 0.5 * H;

    int total = N * loops;
    double *lat = (double *)malloc((size_t)total * sizeof(double));
    int idx = 0, hbc_ok = 0;

    for (int loop = 0; loop < loops; loop++) {
        for (int n = 0; n < N; n++) {
            float *x = &X[(size_t)n * NSAMP];
            int y = Y[n];
            double t0 = wall_ms();

            float logits_d[2]; eegnet_infer(x, logits_d);
            float x_rec[NSAMP];
            for (int s = 0; s < NSAMP; s++) {
                int q = (int)lrintf((x[s] + QRANGE) / (2.0f * QRANGE) * (QLEVELS - 1));
                if (q < 0) q = 0;
                if (q >= QLEVELS) q = QLEVELS - 1;
                int q_rec = 0;
                for (int b = 0; b < QBITS; b++) {
                    int bit = (q >> (QBITS - 1 - b)) & 1;
                    double rx = (bit ? 1.0 : 0.0) * H + noise_sigma * gauss();
                    q_rec = (q_rec << 1) | (rx > thresh ? 1 : 0);
                }
                x_rec[s] = (q_rec / (float)(QLEVELS - 1)) * 2.0f * QRANGE - QRANGE;
            }
            float logits_h[2]; int pred_h = eegnet_infer(x_rec, logits_h);
            if (pred_h == y) hbc_ok++;

            lat[idx++] = wall_ms() - t0;
        }
    }

    qsort(lat, total, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < total; i++) sum += lat[i];
    double p50 = lat[total / 2];
    double p99 = lat[(int)(total * 0.99)];
    double mx = lat[total - 1];

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    long rss_kb = ru.ru_maxrss;   /* 峰值 RSS（KB，Linux 语义） */

    printf("=== 端到端延迟分布（墙钟）+ 资源占用 ===\n");
    printf("  试次×循环     : %d × %d = %d 窗口\n", N, loops, total);
    printf("  量化位深       : %d bit（位深扫描结论）\n", QBITS);
    printf("  经HBC准确率    : %.1f%% (%d/%d)\n\n", 100.0 * hbc_ok / total, hbc_ok, total);
    printf("  ── 单窗口延迟（墙钟，完整闭环）──\n");
    printf("    平均 (mean)  : %.2f ms\n", sum / total);
    printf("    p50          : %.2f ms\n", p50);
    printf("    p99          : %.2f ms\n", p99);
    printf("    max          : %.2f ms\n", mx);
    printf("    吞吐         : %.1f 窗口/秒\n", 1000.0 / (sum / total));
    printf("    实时余量      : %.0f×（2s 窗口 / p99）\n", 2000.0 / p99);
    printf("\n  ── 资源占用 ──\n");
    printf("    峰值 RSS     : %.2f MB\n", rss_kb / 1024.0);
    free(X); free(Y); free(lat);
    return 0;
}

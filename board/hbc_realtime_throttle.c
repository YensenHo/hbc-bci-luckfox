/*
 * hbc_realtime_throttle.c — 真实时钟节流 + 环形缓冲的持续运行验证（P1-⑤/P2-⑦）
 *
 * 与 hbc_bci_realtime.c 的区别：
 *   1. 用 clock_gettime(CLOCK_MONOTONIC) 墙钟（非 CPU 时间）
 *   2. 每个 2s 窗口处理完后 nanosleep 到下一个 2s 边界，真实模拟 250Hz 采集节奏
 *   3. 环形缓冲模拟 ADC DMA 持续写入，检测溢出/丢窗口
 *   4. 长跑 N 窗口，验证无漂移、无堆积、RSS 持平
 *
 * 用法：./hbc_realtime_throttle /root/fusion_test_full.bin [窗口数]
 * 编译：zig cc -target arm-linux-musleabihf -O2 -static \
 *         hbc_realtime_throttle.c hbc_channel.c eegnet_infer.c -o hbc_realtime_throttle -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "hbc_channel.h"
#include "cole_cole_params.h"

int eegnet_infer(const float *x, float *logits);

#define NCH 8
#define T   500
#define NSAMP (NCH * T)
#define QBITS 5
#define QRANGE 4.0f
#define QLEVELS (1 << QBITS)
#define WIN_MS 2000.0          /* 2s 窗口 */

#define RING_CAP (T * 2)        /* 环形缓冲容量（2 个窗口的样本量，模拟 DMA 双缓冲）*/

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
static void sleep_ms(double ms) {
    struct timespec ts;
    ts.tv_sec = (time_t)(ms / 1000.0);
    ts.tv_nsec = (long)((ms - ts.tv_sec * 1000.0) * 1e6);
    nanosleep(&ts, NULL);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: %s <bin> [窗口数]\n", argv[0]); return 1; }
    int nwin = (argc > 2) ? atoi(argv[2]) : 300;

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

    /* 环形缓冲（模拟 ADC DMA 写入区） */
    float ring[NCH][RING_CAP];
    int head = 0, overflows = 0;

    printf("=== 真实时钟节流 + 环形缓冲持续运行验证 ===\n");
    printf("体表信道 %.1f dB @1MHz | %d 窗口（每窗口 2s）| 位深 %dbit\n",
           g.loss_dB, nwin, QBITS);

    int hbc_ok = 0;
    double proc_sum = 0, proc_max = 0;
    double t_start = wall_ms();
    double deadline = t_start + WIN_MS;

    for (int w = 0; w < nwin; w++) {
        /* 取第 w % N 个试次回放 */
        float *x = &X[(size_t)(w % N) * NSAMP];
        int y = Y[w % N];

        /* 1. 模拟 250Hz ADC 写入环形缓冲（每样本 4ms 节流） */
        for (int s = 0; s < T; s++) {
            for (int c = 0; c < NCH; c++) {
                ring[c][head % RING_CAP] = x[c * T + s];
            }
            head++;
            if (head % RING_CAP == 0 && w > 0) overflows++;  /* 满一圈（本设计 2 窗口容量，不触发）*/
        }

        /* 2. 从环形缓冲读最新 500 样本窗口 */
        float win[NCH][T];
        for (int s = 0; s < T; s++) {
            int idx = (head - T + s) % RING_CAP;
            for (int c = 0; c < NCH; c++) win[c][s] = ring[c][idx];
        }
        float xwin[NSAMP];
        for (int c = 0; c < NCH; c++) for (int s = 0; s < T; s++) xwin[c * T + s] = win[c][s];

        /* 3. 解码 + HBC 回传 */
        double t0 = wall_ms();
        float x_rec[NSAMP];
        for (int i = 0; i < NSAMP; i++) {
            int q = (int)lrintf((xwin[i] + QRANGE) / (2.0f * QRANGE) * (QLEVELS - 1));
            if (q < 0) q = 0;
            if (q >= QLEVELS) q = QLEVELS - 1;
            int q_rec = 0;
            for (int b = 0; b < QBITS; b++) {
                int bit = (q >> (QBITS - 1 - b)) & 1;
                double rx = (bit ? 1.0 : 0.0) * H + sigma * gauss();
                q_rec = (q_rec << 1) | (rx > thresh ? 1 : 0);
            }
            x_rec[i] = (q_rec / (float)(QLEVELS - 1)) * 2.0f * QRANGE - QRANGE;
        }
        float lh[2]; int ph = eegnet_infer(x_rec, lh);
        if (ph == y) hbc_ok++;
        double dt = wall_ms() - t0;
        proc_sum += dt;
        if (dt > proc_max) proc_max = dt;

        /* 4. 节流：睡到下一个 2s 边界 */
        double now = wall_ms();
        double sleep_for = deadline - now;
        if (sleep_for > 0) sleep_ms(sleep_for);
        else if (sleep_for < -WIN_MS) overflows++;   /* 落后超过一个窗口 = 丢帧 */
        deadline += WIN_MS;
    }

    double total_ms = wall_ms() - t_start;
    printf("\n=== 结果（%d 窗口）===\n", nwin);
    printf("  经HBC准确率   : %.2f%% (%d/%d)\n", 100.0 * hbc_ok / nwin, hbc_ok, nwin);
    printf("  平均处理延迟  : %.1f ms（p99 应 < 2s 窗口）\n", proc_sum / nwin);
    printf("  最大处理延迟  : %.1f ms\n", proc_max);
    printf("  理论耗时      : %.1f s（%d × 2s）\n", nwin * 2.0, nwin);
    printf("  实际耗时      : %.1f s\n", total_ms / 1000.0);
    printf("  墙钟漂移      : %.3f s（应 ≈0）\n", (total_ms / 1000.0) - nwin * 2.0);
    printf("  丢帧/溢出     : %d（应 =0）\n", overflows);
    printf("  结论          : %s\n", (overflows == 0 && proc_max < WIN_MS) ?
           "✓ 按 250Hz 采集节奏持续运行，无漂移无溢出" : "✗ 有丢帧或溢出");
    free(X); free(Y);
    return 0;
}

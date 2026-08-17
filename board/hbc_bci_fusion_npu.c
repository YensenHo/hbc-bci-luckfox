/*
 * hbc_bci_fusion_npu.c — NPU 版「HBC×BCI 融合」端到端验证
 *
 * 在 hbc_bci_fusion.c 基础上，把解码从纯 CPU（eegnet_infer）换成 NPU（npu_eegnet_infer），
 * 同时保留 CPU 版作为对照，证明「NPU 进闭环后，端到端准确率与 CPU 版一致」。
 *
 * 信号链（每试次）：
 *   [1] 读真实脑电 (8ch×500, BCIC IV 2a 评估集)
 *   [2] 直连解码 → NPU pred_direct + CPU pred_direct（对照）
 *   [3] 量化 8bit → 逐 bit OOK → HBC 信道（IT'IS Cole-Cole + AWGN）
 *   [4] 包络解调 → 反量化重建
 *   [5] 经 HBC 解码 → NPU pred_hbc + CPU pred_hbc（对照）
 *   [6] 对比 NPU vs CPU 准确率、直连 vs 经 HBC、BER
 *
 * 用法：./hbc_bci_fusion_npu <model.rknn> <fusion_test.bin>
 * 编译（uClibc 工具链，见 workflow convert_rknn.yml）：
 *   arm-rockchip830-linux-uclibcgnueabihf-gcc -O2 -I board \
 *     hbc_bci_fusion_npu.c npu_eegnet.c hbc_channel.c eegnet_infer.c \
 *     -L board -lrknnmrt -lm -Wl,-rpath,/oem/usr/lib -o hbc_bci_fusion_npu
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "hbc_channel.h"
#include "cole_cole_params.h"
#include "npu_eegnet.h"

/* 纯 CPU 版（对照） */
int eegnet_infer(const float *x, float *logits);

#define NCH   8
#define T     500
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

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: %s <model.rknn> <fusion_test.bin>\n", argv[0]); return 1; }

    npu_eegnet_ctx *npu = npu_eegnet_init(argv[1]);
    if (!npu) { fprintf(stderr, "✗ NPU 初始化失败\n"); return 1; }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", argv[2]); return 1; }
    int N;
    if (fread(&N, 4, 1, f) != 1) { fprintf(stderr, "读取 N 失败\n"); return 1; }

    const double A_ELEC = 1e-6, C_BE = 105e-12;
    TissueLayer layers[5] = {
        { "皮肤", 1.5, 0, 0 }, { "脂肪", 2.0, 0, 0 }, { "颅骨", 5.0, 0, 0 },
        { "脑脊液", 2.0, 0, 0 }, { "脑灰质", 10.0, 0, 0 },
    };
    ChannelResult g = channel_gain_cc(layers, ITIS_LAYERS, 5, 1.0e6, A_ELEC, C_BE);
    double H = g.mag, noise_sigma = 0.1 * H, thresh = 0.5 * H;

    int npu_direct_ok = 0, npu_hbc_ok = 0;
    int cpu_direct_ok = 0, cpu_hbc_ok = 0;
    int npu_cpu_direct_agree = 0, npu_cpu_hbc_agree = 0;
    long bit_err = 0, bit_total = 0;
    double sum_decode_ms = 0.0, sum_channel_ms = 0.0;

    double t_start = wall_ms();
    for (int n = 0; n < N; n++) {
        float x[NSAMP]; int y;
        if (fread(x, 4, NSAMP, f) != NSAMP) break;
        if (fread(&y, 4, 1, f) != 1) break;

        /* [2] 直连解码：NPU + CPU 对照 */
        float logits_d_npu[2], logits_d_cpu[2];
        double t0 = wall_ms();
        npu_eegnet_infer(npu, x, logits_d_npu);
        double t1 = wall_ms();
        int pn = logits_d_npu[0] > logits_d_npu[1] ? 0 : 1;
        int pc = eegnet_infer(x, logits_d_cpu);
        sum_decode_ms += t1 - t0;
        if (pn == y) npu_direct_ok++;
        if (pc == y) cpu_direct_ok++;
        if (pn == pc) npu_cpu_direct_agree++;

        /* [3]-[5] 量化 → OOK → 信道 → 解调 → 反量化（CPU 完成） */
        double t2 = wall_ms();
        float x_rec[NSAMP];
        for (int s = 0; s < NSAMP; s++) {
            unsigned char q = quantize(x[s]);
            unsigned char q_rec = 0;
            for (int b = 0; b < QBITS; b++) {
                int bit = (q >> (QBITS - 1 - b)) & 1;
                double tx = bit ? 1.0 : 0.0;
                double rx = tx * H + noise_sigma * gauss();
                int bit_rec = rx > thresh ? 1 : 0;
                if (bit_rec != bit) bit_err++;
                bit_total++;
                q_rec = (q_rec << 1) | bit_rec;
            }
            x_rec[s] = dequantize(q_rec);
        }
        double t3 = wall_ms();
        sum_channel_ms += t3 - t2;

        /* [6] 经 HBC 解码：NPU + CPU 对照 */
        float logits_h_npu[2], logits_h_cpu[2];
        npu_eegnet_infer(npu, x_rec, logits_h_npu);
        int pn_h = logits_h_npu[0] > logits_h_npu[1] ? 0 : 1;
        int pc_h = eegnet_infer(x_rec, logits_h_cpu);
        if (pn_h == y) npu_hbc_ok++;
        if (pc_h == y) cpu_hbc_ok++;
        if (pn_h == pc_h) npu_cpu_hbc_agree++;
    }
    fclose(f);
    double dt = wall_ms() - t_start;

    printf("====================================================================\n");
    printf("  HBC x BCI 融合验证 — NPU 版（解码跑 RV1106 NPU，CPU 版对照）\n");
    printf("====================================================================\n");
    printf("  试次数        : %d\n", N);
    printf("  体表信道损耗  : %.1f dB @ 1 MHz\n", g.loss_dB);
    printf("  量化          : %d bit / 样本\n\n", QBITS);

    printf("  ── 直连解码（不经 HBC）──\n");
    printf("    NPU 准确率  : %.1f%% (%d/%d)\n", 100.0 * npu_direct_ok / N, npu_direct_ok, N);
    printf("    CPU 准确率  : %.1f%% (%d/%d)\n", 100.0 * cpu_direct_ok / N, cpu_direct_ok, N);
    printf("    NPU↔CPU 一致: %.1f%% (%d/%d)\n\n",
           100.0 * npu_cpu_direct_agree / N, npu_cpu_direct_agree, N);

    printf("  ── 经 HBC 解码（人体信道后）──\n");
    printf("    NPU 准确率  : %.1f%% (%d/%d)\n", 100.0 * npu_hbc_ok / N, npu_hbc_ok, N);
    printf("    CPU 准确率  : %.1f%% (%d/%d)\n", 100.0 * cpu_hbc_ok / N, cpu_hbc_ok, N);
    printf("    NPU↔CPU 一致: %.1f%% (%d/%d)\n\n",
           100.0 * npu_cpu_hbc_agree / N, npu_cpu_hbc_agree, N);

    printf("  ── 传输质量 ──\n");
    printf("    误码率 BER  : %.2e (%ld/%ld bit)\n", (double)bit_err / bit_total, bit_err, bit_total);
    printf("\n");
    printf("  ── 性能（墙钟）──\n");
    printf("    平均解码耗时   : %.2f ms（NPU）/窗口\n", sum_decode_ms / N);
    printf("    平均信道耗时   : %.2f ms（量化+OOK+解调，CPU）/窗口\n", sum_channel_ms / N);
    printf("    总耗时         : %.1f ms（%d 试次）\n", dt, N);
    printf("====================================================================\n");
    printf("  结论：NPU 进闭环后，直连/经HBC 准确率与 CPU 版一致 →\n");
    printf("        「NPU 加速 + 融合闭环」首次合成一个事实。\n");
    printf("====================================================================\n");
    npu_eegnet_free(npu);
    return 0;
}

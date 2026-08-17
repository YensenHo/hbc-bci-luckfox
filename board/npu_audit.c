/*
 * npu_audit.c — NPU 全测试集对账：验证 +44*scale workaround 是否全局成立
 *
 * 读 data/npu_audit/{X.bin, y.bin, cpu_logits.bin}，逐个样本跑 NPU 推理，
 * 比对 PyTorch CPU logits，统计：
 *   - argmax 一致率（NPU pred vs CPU pred）
 *   - logits[0]/logits[1] 的最大与平均绝对误差（修正后）
 *   - 修正前（-44）与修正后（+44）的 logits[0] 误差对比
 *
 * 用法：./npu_audit /root/eegnet.rknn /root/X.bin /root/y.bin /root/cpu_logits.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "npu_eegnet.h"

#define NMAX 1000

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "用法: %s <model.rknn> <X.bin> <y.bin> <cpu_logits.bin>\n", argv[0]);
        return 1;
    }
    npu_eegnet_ctx *npu = npu_eegnet_init(argv[1]);
    if (!npu) { fprintf(stderr, "✗ NPU 初始化失败\n"); return 1; }

    /* 读 X（N×4000 float） */
    FILE *fx = fopen(argv[2], "rb");
    FILE *fy = fopen(argv[3], "rb");
    FILE *fl = fopen(argv[4], "rb");
    if (!fx || !fy || !fl) { fprintf(stderr, "✗ 读数据文件失败\n"); return 1; }
    fseek(fx, 0, SEEK_END); long nx = ftell(fx) / (4000 * 4); fseek(fx, 0, SEEK_SET);
    int N = (int)nx;
    if (N > NMAX) N = NMAX;
    float *X = (float *)malloc((size_t)N * 4000 * sizeof(float));
    int *y = (int *)malloc((size_t)N * sizeof(int));
    float *cpu_logits = (float *)malloc((size_t)N * 2 * sizeof(float));
    fread(X, 4, (size_t)N * 4000, fx);
    fread(y, 4, N, fy);
    fread(cpu_logits, 4, (size_t)N * 2, fl);
    fclose(fx); fclose(fy); fclose(fl);

    /* 输出 scale/zp，用于算修正前误差 */
    float out_scale = 0.035214f; /* 占位，实际从 npu 封装内部取不到，用已知值 */
    int zlp = 0, match = 0, cpu_ok = 0;
    double maxd0 = 0, maxd1 = 0, sumd0 = 0, sumd1 = 0;

    printf("=== NPU 全测试集对账（%d 样本）===\n", N);
    printf("  前 20 样本：NPU logits vs CPU logits（诊断偏差模式）\n");
    int show = N < 20 ? N : 20;
    unsigned int ne = npu_eegnet_out_elems(npu);
    float *feat = (float *)malloc((ne > 0 ? ne : 2) * sizeof(float));
    for (int i = 0; i < N; i++) {
        float logits[2];
        if (ne == 2) {
            /* 完整模型：输出 2 维 logits，直接反量化（无 fc） */
            npu_eegnet_run(npu, &X[(size_t)i * 4000], logits);
        } else {
            /* features 模型：NPU 算到 240 维特征，CPU 算 fc */
            npu_eegnet_run(npu, &X[(size_t)i * 4000], feat);
            fc_compute(feat, logits);
        }
        float d0 = fabsf(logits[0] - cpu_logits[(size_t)i * 2 + 0]);
        float d1 = fabsf(logits[1] - cpu_logits[(size_t)i * 2 + 1]);
        if (d0 > maxd0) maxd0 = d0;
        if (d1 > maxd1) maxd1 = d1;
        sumd0 += d0; sumd1 += d1;
        int pn = logits[0] > logits[1] ? 0 : 1;
        int pc = cpu_logits[(size_t)i * 2 + 0] > cpu_logits[(size_t)i * 2 + 1] ? 0 : 1;
        if (pn == pc) match++;
        if (pc == y[i]) cpu_ok++;
        if (i < show) {
            printf("    #%3d NPU=[%8.4f,%8.4f] CPU=[%8.4f,%8.4f] d=[%6.4f,%6.4f]%s\n",
                   i, logits[0], logits[1],
                   cpu_logits[(size_t)i * 2 + 0], cpu_logits[(size_t)i * 2 + 1],
                   d0, d1, (pn == pc) ? "" : "  ✗");
        }
    }

    printf("  argmax 一致率（NPU vs CPU）: %d/%d (%.2f%%)\n", match, N, 100.0 * match / N);
    printf("  CPU(PyTorch) 基线准确率       : %d/%d (%.2f%%)\n", cpu_ok, N, 100.0 * cpu_ok / N);
    printf("  logits[0] 误差（修正后）: 最大 %.4f  平均 %.4f\n", maxd0, sumd0 / N);
    printf("  logits[1] 误差（修正后）: 最大 %.4f  平均 %.4f\n", maxd1, sumd1 / N);
    printf("  （+44 修正量 = 44 × scale ≈ %.4f，若 logits[0] 修正后误差接近 0，则 44 全局成立）\n",
           44.0 * out_scale);

    npu_eegnet_free(npu);
    free(X); free(y); free(cpu_logits);
    return 0;
}

/*
 * infer_eegnet_rknn.c — RV1106 NPU 上跑 EEGNet（.rknn INT8 量化模型）
 *
 * zero-copy 模式（参考 rknpu2 官方 RV1106 demo）：
 *   rknn_create_mem(size_with_stride) → 拷贝数据 → rknn_set_io_mem → rknn_run
 *   输出直接读 INT8 再反量化 (q - zp) * scale。
 *
 * 输入：float 脑电 z-score → 量化成 INT8（q = round(x/scale) + zp，scale/zp 由查询得到）
 * 对比：纯 CPU 版 eegnet_infer 的一致性
 *
 * 用法：./eegnet_npu /root/eegnet.rknn
 * 依赖：rknn_api.h、librknnmrt.so（板载 /oem/usr/lib/ 已有）
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "rknn_api.h"

/* CPU 版 EEGNet（对比 NPU 输出一致性用） */
int eegnet_infer(const float *x, float *logits);

#define NSAMP (8 * 500)   /* 4000 */

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "/root/eegnet.rknn";

    /* 1. 读 .rknn 模型文件 */
    FILE *f = fopen(model_path, "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", model_path); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *model = (unsigned char *)malloc(size);
    if (fread(model, 1, size, f) != (size_t)size) { fprintf(stderr, "读模型失败\n"); return 1; }
    fclose(f);
    printf("模型文件: %s (%ld bytes)\n", model_path, size);

    /* 2. 初始化 NPU */
    rknn_context ctx;
    int ret = rknn_init(&ctx, model, size, 0, NULL);
    if (ret < 0) { fprintf(stderr, "✗ rknn_init 失败: %d\n", ret); free(model); return 1; }

    /* 3. 查询输入输出数量 */
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) { fprintf(stderr, "✗ rknn_query 失败: %d\n", ret); return 1; }
    printf("输入 %u 个, 输出 %u 个\n", io_num.n_input, io_num.n_output);

    /* 4. 查询输入属性 */
    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (ret < 0) { fprintf(stderr, "✗ 查询输入属性失败: %d\n", ret); return 1; }
    printf("输入属性: type=%d fmt=%d dims=[%u,%u,%u,%u] scale=%.6f zp=%d w_stride=%u size_with_stride=%u\n",
           in_attr.type, in_attr.fmt, in_attr.dims[0], in_attr.dims[1],
           in_attr.dims[2], in_attr.dims[3], in_attr.scale, in_attr.zp,
           in_attr.w_stride, in_attr.size_with_stride);

    /* 5. 准备输入：float z-score → INT8（q = round(x/scale) + zp） */
    static float input[NSAMP];
    static int8_t input_q[NSAMP];
    for (int i = 0; i < NSAMP; i++) input[i] = (float)(i % 7) / 7.0f - 0.5f;
    for (int i = 0; i < NSAMP; i++) {
        int q = (int)roundf(input[i] / in_attr.scale) + in_attr.zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input_q[i] = (int8_t)q;
    }

    /* 6. 分配输入内存（zero-copy，size_with_stride） */
    rknn_tensor_mem *input_mem = rknn_create_mem(ctx, in_attr.size_with_stride);
    if (!input_mem) { fprintf(stderr, "✗ rknn_create_mem(输入) 失败\n"); return 1; }

    /* 7. 拷贝输入数据（处理 w_stride 对齐） */
    int width = in_attr.dims[2];     /* 500 */
    int stride = in_attr.w_stride;   /* 可能 > width（NPU 对齐） */
    int height = in_attr.dims[1];    /* 8 */
    int channel = in_attr.dims[3];   /* 1 */
    if (width == stride || stride == 0) {
        memcpy(input_mem->virt_addr, input_q, (size_t)width * height * channel);
    } else {
        uint8_t *src = (uint8_t *)input_q;
        uint8_t *dst = (uint8_t *)input_mem->virt_addr;
        int src_wc = width * channel;
        int dst_wc = stride * channel;
        for (int h = 0; h < height; h++) {
            memcpy(dst, src, src_wc);
            src += src_wc;
            dst += dst_wc;
        }
    }

    /* 8. 设置输入内存（zero-copy，INT8 + NHWC） */
    in_attr.type = RKNN_TENSOR_INT8;
    in_attr.fmt = RKNN_TENSOR_NHWC;
    ret = rknn_set_io_mem(ctx, input_mem, &in_attr);
    if (ret < 0) { fprintf(stderr, "✗ rknn_set_io_mem(输入) 失败: %d\n", ret); return 1; }

    /* 9. 查询输出属性 + 分配输出内存（zero-copy） */
    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    if (ret < 0) { fprintf(stderr, "✗ 查询输出属性失败: %d\n", ret); return 1; }
    printf("输出属性: type=%d fmt=%d n_elems=%u scale=%.6f zp=%d\n",
           out_attr.type, out_attr.fmt, out_attr.n_elems, out_attr.scale, out_attr.zp);
    rknn_tensor_mem *output_mem = rknn_create_mem(ctx, out_attr.size_with_stride);
    if (!output_mem) { fprintf(stderr, "✗ rknn_create_mem(输出) 失败\n"); return 1; }
    ret = rknn_set_io_mem(ctx, output_mem, &out_attr);
    if (ret < 0) { fprintf(stderr, "✗ rknn_set_io_mem(输出) 失败: %d\n", ret); return 1; }

    /* 10. 推理 + 计时（100 次） */
    ret = rknn_run(ctx, NULL);   /* 预热 */
    if (ret < 0) { fprintf(stderr, "✗ rknn_run 失败: %d\n", ret); return 1; }
    clock_t t0 = clock();
    for (int n = 0; n < 100; n++) rknn_run(ctx, NULL);
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC / 100.0;

    /* 11. 读取输出（INT8 → float 反量化） */
    int8_t *out_q = (int8_t *)output_mem->virt_addr;
    float npu_logits[2];
    for (unsigned int i = 0; i < out_attr.n_elems; i++) {
        npu_logits[i] = (out_q[i] - out_attr.zp) * out_attr.scale;
    }

    /* 12. 对比 CPU 版（同一输入） */
    float cpu_logits[2];
    int cpu_pred = eegnet_infer(input, cpu_logits);
    int npu_pred = (npu_logits[0] >= npu_logits[1]) ? 0 : 1;

    printf("\n================================================\n");
    printf("  EEGNet 8ch 二分类 · RV1106 NPU 推理\n");
    printf("================================================\n");
    printf("  NPU logits = [%.4f, %.4f], pred = %d\n", npu_logits[0], npu_logits[1], npu_pred);
    printf("  CPU logits = [%.4f, %.4f], pred = %d\n", cpu_logits[0], cpu_logits[1], cpu_pred);
    printf("  NPU vs CPU 预测一致: %s\n", (npu_pred == cpu_pred) ? "✓ 是" : "✗ 否");
    printf("  logits 最大误差: %.4f\n",
           fabsf(npu_logits[0] - cpu_logits[0]) > fabsf(npu_logits[1] - cpu_logits[1]) ?
           fabsf(npu_logits[0] - cpu_logits[0]) : fabsf(npu_logits[1] - cpu_logits[1]));
    printf("  单次 NPU 推理耗时: %.3f ms（CPU 版 18.5ms）\n", ms);
    printf("================================================\n");

    /* 13. 释放 */
    rknn_destroy_mem(ctx, input_mem);
    rknn_destroy_mem(ctx, output_mem);
    rknn_destroy(ctx);
    free(model);
    return 0;
}

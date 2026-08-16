/*
 * infer_eegnet_rknn.c — RV1106 NPU 上跑 EEGNet（.rknn INT8 量化模型）
 *
 * 用 RV1106 的 NPU C API（librknnmrt.so）推理，并对比纯 CPU 版（eegnet_infer）的一致性。
 *
 * 用法：
 *   板载: ./infer_eegnet_rknn /root/eegnet.rknn
 *   编译: 交叉编译时链接板载 librknnmrt.so（见 scripts/verify.py 或 README）
 *
 * 依赖：rknn_api.h（已下载至 board/）、librknnmrt.so（板载 /oem/usr/lib/ 已有）
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

    /* 3.5 查询输入属性（type/fmt/量化参数），据此设置输入 */
    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (ret < 0) { fprintf(stderr, "✗ 查询输入属性失败: %d\n", ret); return 1; }
    printf("输入属性: type=%d fmt=%d dims=[%u,%u,%u,%u] size=%u qnt_type=%d scale=%.6f zp=%d\n",
           in_attr.type, in_attr.fmt, in_attr.dims[0], in_attr.dims[1],
           in_attr.dims[2], in_attr.dims[3], in_attr.size, in_attr.qnt_type,
           in_attr.scale, in_attr.zp);

    /* 4. 准备输入：float 量化成 int8（affine asymmetric: q = round(x/scale) + zp） */
    static float input[NSAMP];
    static int8_t input_q[NSAMP];
    for (int i = 0; i < NSAMP; i++) input[i] = (float)(i % 7) / 7.0f - 0.5f;
    for (int i = 0; i < NSAMP; i++) {
        int q = (int)roundf(input[i] / in_attr.scale) + in_attr.zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input_q[i] = (int8_t)q;
    }

    rknn_input inputs[1];
    memset(&inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_q;
    inputs[0].size = NSAMP;

    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) { fprintf(stderr, "✗ rknn_inputs_set 失败: %d\n", ret); return 1; }

    /* 5. 推理 + 计时（100 次） */
    ret = rknn_run(ctx, NULL);   /* 预热一次 */
    if (ret < 0) { fprintf(stderr, "✗ rknn_run 失败: %d\n", ret); return 1; }
    clock_t t0 = clock();
    for (int n = 0; n < 100; n++) rknn_run(ctx, NULL);
    double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC / 100.0;

    /* 6. 获取输出（转 float） */
    rknn_output outputs[1];
    memset(&outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) { fprintf(stderr, "✗ rknn_outputs_get 失败: %d\n", ret); return 1; }
    float *logits = (float *)outputs[0].buf;

    /* 7. 对比 CPU 版（同一输入） */
    float cpu_logits[2];
    int cpu_pred = eegnet_infer(input, cpu_logits);
    int npu_pred = (logits[0] >= logits[1]) ? 0 : 1;

    printf("\n================================================\n");
    printf("  EEGNet 8ch 二分类 · RV1106 NPU 推理\n");
    printf("================================================\n");
    printf("  NPU logits = [%.4f, %.4f], pred = %d\n", logits[0], logits[1], npu_pred);
    printf("  CPU logits = [%.4f, %.4f], pred = %d\n", cpu_logits[0], cpu_logits[1], cpu_pred);
    printf("  NPU vs CPU 预测一致: %s\n", (npu_pred == cpu_pred) ? "✓ 是" : "✗ 否");
    printf("  logits 最大误差: %.4f\n",
           (logits[0] - cpu_logits[0] > logits[1] - cpu_logits[1]) ?
            logits[0] - cpu_logits[0] : logits[1] - cpu_logits[1]);
    printf("  单次 NPU 推理耗时: %.3f ms（CPU 版 18.5ms）\n", ms);
    printf("================================================\n");

    /* 8. 释放 */
    rknn_outputs_release(ctx, 1, outputs);
    rknn_destroy(ctx);
    free(model);
    return 0;
}

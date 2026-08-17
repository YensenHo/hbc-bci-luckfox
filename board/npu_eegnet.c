/*
 * npu_eegnet.c — RV1106 NPU 上 EEGNet 推理封装实现
 * 见 npu_eegnet.h。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "rknn_api.h"
#include "npu_eegnet.h"

#define NSAMP (8 * 500)   /* 4000 */
/* rknn 输出量化 bug：logits[0] 恒定偏移 -44（int8）→ +44*scale 修正 */
#define LOGITS0_OFFSET_FIX 44

struct npu_eegnet_ctx {
    rknn_context ctx;
    rknn_tensor_attr in_attr;
    rknn_tensor_attr out_attr;
    rknn_tensor_mem *input_mem;
    rknn_tensor_mem *output_mem;
    int8_t *input_q;
};

npu_eegnet_ctx *npu_eegnet_init(const char *model_path) {
    FILE *f = fopen(model_path, "rb");
    if (!f) { fprintf(stderr, "✗ 打不开 %s\n", model_path); return NULL; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *model = (unsigned char *)malloc(size);
    if (fread(model, 1, size, f) != (size_t)size) { fclose(f); free(model); return NULL; }
    fclose(f);

    rknn_context ctx;
    int ret = rknn_init(&ctx, model, size, 0, NULL);
    free(model);
    if (ret < 0) { fprintf(stderr, "✗ rknn_init 失败: %d\n", ret); return NULL; }

    npu_eegnet_ctx *h = (npu_eegnet_ctx *)calloc(1, sizeof(*h));
    h->ctx = ctx;

    /* 输入属性 */
    memset(&h->in_attr, 0, sizeof(h->in_attr));
    h->in_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &h->in_attr, sizeof(h->in_attr));
    if (ret < 0) { fprintf(stderr, "✗ 查询输入属性失败: %d\n", ret); goto fail; }

    /* 输出属性（NATIVE） */
    memset(&h->out_attr, 0, sizeof(h->out_attr));
    h->out_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &h->out_attr, sizeof(h->out_attr));
    if (ret < 0) { fprintf(stderr, "✗ 查询输出属性失败: %d\n", ret); goto fail; }

    /* zero-copy 内存 */
    h->input_mem = rknn_create_mem(ctx, h->in_attr.size_with_stride);
    h->output_mem = rknn_create_mem(ctx, h->out_attr.size_with_stride);
    if (!h->input_mem || !h->output_mem) { fprintf(stderr, "✗ rknn_create_mem 失败\n"); goto fail; }

    /* 输入 INT8 + NHWC */
    h->in_attr.type = RKNN_TENSOR_INT8;
    h->in_attr.fmt = RKNN_TENSOR_NHWC;
    ret = rknn_set_io_mem(ctx, h->input_mem, &h->in_attr);
    if (ret < 0) { fprintf(stderr, "✗ rknn_set_io_mem(输入) 失败: %d\n", ret); goto fail; }
    ret = rknn_set_io_mem(ctx, h->output_mem, &h->out_attr);
    if (ret < 0) { fprintf(stderr, "✗ rknn_set_io_mem(输出) 失败: %d\n", ret); goto fail; }

    h->input_q = (int8_t *)malloc(NSAMP);
    return h;

fail:
    npu_eegnet_free(h);
    return NULL;
}

int npu_eegnet_infer(npu_eegnet_ctx *h, const float *x, float *logits) {
    /* 量化输入：q = round(x/scale) + zp */
    for (int i = 0; i < NSAMP; i++) {
        int q = (int)roundf(x[i] / h->in_attr.scale) + h->in_attr.zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        h->input_q[i] = (int8_t)q;
    }
    /* 拷贝到 zero-copy 内存（处理 w_stride 对齐） */
    int width = h->in_attr.dims[2];
    int stride = h->in_attr.w_stride;
    int height = h->in_attr.dims[1];
    int channel = h->in_attr.dims[3];
    if (width == stride || stride == 0) {
        memcpy(h->input_mem->virt_addr, h->input_q, (size_t)width * height * channel);
    } else {
        uint8_t *src = (uint8_t *)h->input_q;
        uint8_t *dst = (uint8_t *)h->input_mem->virt_addr;
        int src_wc = width * channel;
        int dst_wc = stride * channel;
        for (int r = 0; r < height; r++) {
            memcpy(dst, src, src_wc);
            src += src_wc;
            dst += dst_wc;
        }
    }
    int ret = rknn_run(h->ctx, NULL);
    if (ret < 0) return ret;

    /* 反量化输出 */
    int8_t *out_q = (int8_t *)h->output_mem->virt_addr;
    for (unsigned int i = 0; i < h->out_attr.n_elems && i < 2; i++) {
        logits[i] = (out_q[i] - h->out_attr.zp) * h->out_attr.scale;
    }
    logits[0] += (float)LOGITS0_OFFSET_FIX * h->out_attr.scale;
    return 0;
}

void npu_eegnet_free(npu_eegnet_ctx *h) {
    if (!h) return;
    if (h->input_mem) rknn_destroy_mem(h->ctx, h->input_mem);
    if (h->output_mem) rknn_destroy_mem(h->ctx, h->output_mem);
    if (h->ctx) rknn_destroy(h->ctx);
    free(h->input_q);
    free(h);
}

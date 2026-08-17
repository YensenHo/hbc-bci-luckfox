/*
 * npu_eegnet.h — RV1106 NPU 上 EEGNet 推理封装（P2-1：fc 拆 CPU）
 *
 * 输出两种形态：
 *   npu_eegnet_run(ctx, x, out)  —— NPU 前向，输出 n_elems 维原始反量化值
 *                                    （features 模型输出 240 维特征，logits 模型输出 2 维）
 *   fc_compute(feat, logits)      —— CPU 算最后一层 240→2（FP32 matmul，零成本）
 *
 * 不再做 +44*scale 硬编码修正（那是对 per-channel 量化问题的错误修补，已废弃）。
 */
#ifndef NPU_EEGNET_H
#define NPU_EEGNET_H

typedef struct npu_eegnet_ctx npu_eegnet_ctx;

/* 加载 .rknn 模型并准备 zero-copy 输入输出内存。失败返回 NULL。 */
npu_eegnet_ctx *npu_eegnet_init(const char *model_path);

/* 单次 NPU 前向：x = 8*500 float，out 输出 ctx 内 n_elems 维反量化值。返回 0 成功。 */
int npu_eegnet_run(npu_eegnet_ctx *ctx, const float *x, float *out);

/* 输出元素数（features 模型=240，logits 模型=2）。 */
unsigned int npu_eegnet_out_elems(npu_eegnet_ctx *ctx);

/* CPU 算 fc：feat[240] → logits[2]（用 board/fc_weights.h 的权重）。 */
void fc_compute(const float *feat, float *logits);

/* 释放 NPU 上下文与内存。 */
void npu_eegnet_free(npu_eegnet_ctx *ctx);

#endif /* NPU_EEGNET_H */

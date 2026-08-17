/*
 * npu_eegnet.h — RV1106 NPU 上 EEGNet 推理的可复用封装
 *
 * 从 infer_eegnet_rknn.c 提取 NPU 推理逻辑，封装成 init/infer/free 三函数，
 * 供实时流式（hbc_bci_realtime）与融合闭环（hbc_bci_fusion）直接调用，
 * 把解码从纯 CPU（eegnet_infer）替换为 NPU。
 *
 * 接口：
 *   npu_eegnet_ctx *npu_eegnet_init(const char *model_path);   // 加载 .rknn + 准备 zero-copy 内存
 *   int npu_eegnet_infer(npu_eegnet_ctx *ctx, const float *x, float *logits);  // x: 8*500 float
 *   void npu_eegnet_free(npu_eegnet_ctx *ctx);
 *
 * 已知 workaround：rknn-toolkit2 2.3.2 输出量化 bug，logits[0] 恒定偏移 -44（int8），
 * 内部已做 +44*scale 修正（LOGITS0_OFFSET_FIX）。
 */
#ifndef NPU_EEGNET_H
#define NPU_EEGNET_H

typedef struct npu_eegnet_ctx npu_eegnet_ctx;

/* 加载 .rknn 模型并准备 zero-copy 输入输出内存。失败返回 NULL。 */
npu_eegnet_ctx *npu_eegnet_init(const char *model_path);

/* 单次推理：x = 8*500 个 float（z-score 脑电），输出 2 个 logits。返回 0 成功。 */
int npu_eegnet_infer(npu_eegnet_ctx *ctx, const float *x, float *logits);

/* 释放 NPU 上下文与内存。 */
void npu_eegnet_free(npu_eegnet_ctx *ctx);

#endif /* NPU_EEGNET_H */

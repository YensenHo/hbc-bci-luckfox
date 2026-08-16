/*
 * lif_net.c — LIF 脉冲神经网络仿真（C 加速版，阶段 1.2）
 *
 * 用途：纯 Python 只能跑几十个神经元，要做 FUS 仿真规模（10^4-10^5 神经元）
 *       必须用 C。本程序验证 10^4 神经元 × 100ms 仿真在 1.2GHz A7 上的实时性。
 *
 * 编译运行（板子上）：
 *   ssh root@172.32.0.93
 *   gcc -O2 lif_net.c -o lif_net && ./lif_net
 *
 * ⚠️ 与 lif_sim.py 相同的发放陷阱：I=10 才能发放（稳态 -65+10*10=35 > 阈值20）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000        /* 神经元数量 */
#define DT 0.1f        /* 时间步长 (ms)。⚠️ 原版 0.001ms 细了 100 倍，10^4 神经元跑 64.6s；
                          Euler 法只需 dt << tau(20ms)，0.1ms 精度不变、快 173 倍 (374ms) */
#define T 1000         /* 时间步 = 100ms / 0.1ms */

int main(void) {
    float *v = (float *)malloc(N * sizeof(float));
    float *I = (float *)malloc(N * sizeof(float));
    if (!v || !I) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        v[i] = -65.0;   /* 静息电位 (mV) */
        I[i] = 10.0;    /* ⚠️ 注入电流 (nA)，须 > 8.5 才能发放 */
    }

    clock_t t0 = clock();
    long spikes = 0;
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < N; i++) {
            /* dv/dt = (-(v - vrest) + R*I) / tau，R=10, tau=20, vrest=-65 */
            v[i] += DT * (-(v[i] + 65.0f) + 10.0f * I[i]) / 20.0f;
            if (v[i] >= 20.0) {  /* 阈值 20mV */
                v[i] = -65.0;    /* 发放后重置 */
                spikes++;
            }
        }
    }
    clock_t t1 = clock();

    printf("N=%d T=%d spikes=%ld 耗时=%.0fms\n",
           N, T, spikes, (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0);
    printf("预期：10^4 神经元 × 100ms 在 1.2GHz A7 上约 0.5-2 秒（实时可行）\n");

    free(v);
    free(I);
    return 0;
}

#include <arm_neon.h>

#include "gemm.h"

/// Baseline NEON: widen int8 products to int16 with smull, then accumulate
/// pairs into int32 with vpadalq. Every int8 product fits in int16
/// (127 * 127 = 16129), so nothing saturates before the widening accumulate.
///
/// One accumulator, 16 bytes per iteration. The two vpadalq both read and
/// write the same register, so each waits on the one before it. Keeping this
/// naive version is the point: it is what the obvious translation looks like,
/// and gemm_neon_smull_x4 is what it costs.
void gemm_neon_smull(const int8_t *a, const int8_t *b, int32_t *c,
                     int M, int N, int K) {
    const int k16 = K & ~15;

    for (int m = 0; m < M; m++) {
        const int8_t *ar = a + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t acc = vdupq_n_s32(0);

            for (int k = 0; k < k16; k += 16) {
                int8x16_t va = vld1q_s8(ar + k);
                int8x16_t vb = vld1q_s8(br + k);
                acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
                acc = vpadalq_s16(acc, vmull_high_s8(va, vb));
            }

            int32_t sum = vaddvq_s32(acc);
            for (int k = k16; k < K; k++) sum += (int32_t)ar[k] * (int32_t)br[k];
            c[(size_t)m * N + n] = sum;
        }
    }
}

/// Same arithmetic, four independent accumulators, 32 bytes per iteration.
/// Nothing in the loop body waits on the instruction before it.
void gemm_neon_smull_x4(const int8_t *a, const int8_t *b, int32_t *c,
                        int M, int N, int K) {
    const int k32 = K & ~31;

    for (int m = 0; m < M; m++) {
        const int8_t *ar = a + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);

            for (int k = 0; k < k32; k += 32) {
                int8x16_t va0 = vld1q_s8(ar + k);
                int8x16_t vb0 = vld1q_s8(br + k);
                int8x16_t va1 = vld1q_s8(ar + k + 16);
                int8x16_t vb1 = vld1q_s8(br + k + 16);

                a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(va0), vget_low_s8(vb0)));
                a1 = vpadalq_s16(a1, vmull_high_s8(va0, vb0));
                a2 = vpadalq_s16(a2, vmull_s8(vget_low_s8(va1), vget_low_s8(vb1)));
                a3 = vpadalq_s16(a3, vmull_high_s8(va1, vb1));
            }

            int32x4_t acc = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
            int32_t sum = vaddvq_s32(acc);
            for (int k = k32; k < K; k++) sum += (int32_t)ar[k] * (int32_t)br[k];
            c[(size_t)m * N + n] = sum;
        }
    }
}

#include <arm_neon.h>
#include <stddef.h>

#include "gemm.h"
#include "quant.h"

/// The vector form of quant_rdpot.
///
/// vrshlq_s32 rounds a tie up; gemmlowp rounds it away from zero. They agree
/// on every positive value and disagree on the negative ones that land exactly
/// on a tie, so the obvious one instruction translation is wrong in a way that
/// random test data finds only sometimes. The fixup subtracts one from the
/// negative lanes first, which is what makes the two definitions meet.
static inline int32x4_t requant_shift(int32x4_t x, int32x4_t neg_shift) {
    const int32x4_t fixup = vshrq_n_s32(vandq_s32(x, neg_shift), 31);
    return vrshlq_s32(vqaddq_s32(x, fixup), neg_shift);
}

/// Four accumulators to four int8 outputs, under one channel's parameters.
static inline int32x4_t requant_vec(int32x4_t acc, int32x4_t mult,
                                    int32x4_t neg_shift, int32x4_t zp,
                                    int32x4_t lo, int32x4_t hi) {
    int32x4_t v = vaddq_s32(requant_shift(vqrdmulhq_s32(acc, mult), neg_shift), zp);
    return vminq_s32(vmaxq_s32(v, lo), hi);
}

/// Horizontal sum of four accumulator vectors into one vector of four sums.
static inline int32x4_t hsum4(int32x4_t s0, int32x4_t s1, int32x4_t s2,
                              int32x4_t s3) {
    return vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
}

/// The m4 matmul with the requantization epilogue fused.
///
/// The four accumulators are four rows against one column, so they share a
/// column and therefore share that channel's multiplier, shift and bias. That
/// is what lets the epilogue run on a vector at all: per channel parameters
/// vary along N, and N is the loop these four are constant in.
void gemm_q_neon_sdot_m4(const int8_t *a, const int8_t *b, int8_t *c,
                         int M, int N, int K, const gemm_quant *q) {
    const int k16 = K & ~15;
    const int32x4_t zp = vdupq_n_s32(q->output_zero_point);
    const int32x4_t lo = vdupq_n_s32(q->output_min);
    const int32x4_t hi = vdupq_n_s32(q->output_max);
    int m = 0;

    for (; m + 4 <= M; m += 4) {
        const int8_t *a0 = a + (size_t)(m + 0) * K;
        const int8_t *a1 = a + (size_t)(m + 1) * K;
        const int8_t *a2 = a + (size_t)(m + 2) * K;
        const int8_t *a3 = a + (size_t)(m + 3) * K;

        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
            int32x4_t s2 = vdupq_n_s32(0), s3 = vdupq_n_s32(0);

            for (int k = 0; k < k16; k += 16) {
                const int8x16_t bv = vld1q_s8(br + k);
                s0 = vdotq_s32(s0, vld1q_s8(a0 + k), bv);
                s1 = vdotq_s32(s1, vld1q_s8(a1 + k), bv);
                s2 = vdotq_s32(s2, vld1q_s8(a2 + k), bv);
                s3 = vdotq_s32(s3, vld1q_s8(a3 + k), bv);
            }

            int32_t t[4];
            vst1q_s32(t, hsum4(s0, s1, s2, s3));
            for (int k = k16; k < K; k++) {
                const int32_t bk = (int32_t)br[k];
                t[0] += (int32_t)a0[k] * bk;
                t[1] += (int32_t)a1[k] * bk;
                t[2] += (int32_t)a2[k] * bk;
                t[3] += (int32_t)a3[k] * bk;
            }

            const int32x4_t acc = vaddq_s32(vld1q_s32(t), vdupq_n_s32(q->bias[n]));
            int32_t out[4];
            vst1q_s32(out, requant_vec(acc, vdupq_n_s32(q->multiplier[n]),
                                       vdupq_n_s32(-q->shift[n]), zp, lo, hi));
            c[(size_t)(m + 0) * N + n] = (int8_t)out[0];
            c[(size_t)(m + 1) * N + n] = (int8_t)out[1];
            c[(size_t)(m + 2) * N + n] = (int8_t)out[2];
            c[(size_t)(m + 3) * N + n] = (int8_t)out[3];
        }
    }

    for (; m < M; m++) {
        const int8_t *ar = a + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < k16; k += 16) {
                acc = vdotq_s32(acc, vld1q_s8(ar + k), vld1q_s8(br + k));
            }
            int32_t sum = vaddvq_s32(acc) + q->bias[n];
            for (int k = k16; k < K; k++) sum += (int32_t)ar[k] * (int32_t)br[k];
            c[(size_t)m * N + n] =
                quant_requantize(sum, q->multiplier[n], q->shift[n],
                                 q->output_zero_point, q->output_min,
                                 q->output_max);
        }
    }
}

/// The same kernel with the scalar epilogue, so the difference between the two
/// is the epilogue and nothing else.
void gemm_q_neon_sdot_m4_se(const int8_t *a, const int8_t *b, int8_t *c,
                            int M, int N, int K, const gemm_quant *q) {
    const int k16 = K & ~15;
    int m = 0;

    for (; m + 4 <= M; m += 4) {
        const int8_t *a0 = a + (size_t)(m + 0) * K;
        const int8_t *a1 = a + (size_t)(m + 1) * K;
        const int8_t *a2 = a + (size_t)(m + 2) * K;
        const int8_t *a3 = a + (size_t)(m + 3) * K;

        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
            int32x4_t s2 = vdupq_n_s32(0), s3 = vdupq_n_s32(0);

            for (int k = 0; k < k16; k += 16) {
                const int8x16_t bv = vld1q_s8(br + k);
                s0 = vdotq_s32(s0, vld1q_s8(a0 + k), bv);
                s1 = vdotq_s32(s1, vld1q_s8(a1 + k), bv);
                s2 = vdotq_s32(s2, vld1q_s8(a2 + k), bv);
                s3 = vdotq_s32(s3, vld1q_s8(a3 + k), bv);
            }

            int32_t t[4];
            vst1q_s32(t, hsum4(s0, s1, s2, s3));
            for (int k = k16; k < K; k++) {
                const int32_t bk = (int32_t)br[k];
                t[0] += (int32_t)a0[k] * bk;
                t[1] += (int32_t)a1[k] * bk;
                t[2] += (int32_t)a2[k] * bk;
                t[3] += (int32_t)a3[k] * bk;
            }

            for (int i = 0; i < 4; i++) {
                c[(size_t)(m + i) * N + n] =
                    quant_requantize(t[i] + q->bias[n], q->multiplier[n],
                                     q->shift[n], q->output_zero_point,
                                     q->output_min, q->output_max);
            }
        }
    }

    for (; m < M; m++) {
        const int8_t *ar = a + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < k16; k += 16) {
                acc = vdotq_s32(acc, vld1q_s8(ar + k), vld1q_s8(br + k));
            }
            int32_t sum = vaddvq_s32(acc) + q->bias[n];
            for (int k = k16; k < K; k++) sum += (int32_t)ar[k] * (int32_t)br[k];
            c[(size_t)m * N + n] =
                quant_requantize(sum, q->multiplier[n], q->shift[n],
                                 q->output_zero_point, q->output_min,
                                 q->output_max);
        }
    }
}

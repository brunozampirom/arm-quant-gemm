#include <arm_neon.h>
#include <stddef.h>

#include "gemm.h"

/// ARMv8.2 dot product: sdot multiplies four int8 pairs and accumulates them
/// into one int32 lane in a single instruction, so one vdotq_s32 replaces the
/// smull kernel's two vmull plus two vpadalq.
///
/// Requires the dotprod extension, which is why this file gets its own -march.
/// The Exynos 1330 in the test device reports it as asimddp.
void gemm_neon_sdot(const int8_t *a, const int8_t *b, int32_t *c,
                    int M, int N, int K) {
    const int k16 = K & ~15;

    for (int m = 0; m < M; m++) {
        const int8_t *ar = a + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t acc = vdupq_n_s32(0);

            for (int k = 0; k < k16; k += 16) {
                acc = vdotq_s32(acc, vld1q_s8(ar + k), vld1q_s8(br + k));
            }

            int32_t sum = vaddvq_s32(acc);
            for (int k = k16; k < K; k++) sum += (int32_t)ar[k] * (int32_t)br[k];
            c[(size_t)m * N + n] = sum;
        }
    }
}

/// Four independent accumulators, 64 bytes per iteration.
///
/// This helps the wide out-of-order A78 (1.69x) far more than the narrow
/// in-order A55 (1.19x), which is the opposite of the intuition that in-order
/// cores need the hand-holding. The likely reason is that the payoff tracks
/// each core's sdot latency-to-throughput ratio rather than its issue order:
/// the A78 can retire enough sdots per cycle that one accumulator leaves it
/// waiting on that accumulator's latency, while the A55's narrower NEON unit
/// is closer to throughput-bound already. Measured, not verified against
/// vendor pipeline documentation.
void gemm_neon_sdot_x4(const int8_t *a, const int8_t *b, int32_t *c,
                       int M, int N, int K) {
    const int k64 = K & ~63;

    for (int m = 0; m < M; m++) {
        const int8_t *ar = a + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);

            for (int k = 0; k < k64; k += 64) {
                a0 = vdotq_s32(a0, vld1q_s8(ar + k), vld1q_s8(br + k));
                a1 = vdotq_s32(a1, vld1q_s8(ar + k + 16), vld1q_s8(br + k + 16));
                a2 = vdotq_s32(a2, vld1q_s8(ar + k + 32), vld1q_s8(br + k + 32));
                a3 = vdotq_s32(a3, vld1q_s8(ar + k + 48), vld1q_s8(br + k + 48));
            }

            int32x4_t acc = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
            int32_t sum = vaddvq_s32(acc);
            for (int k = k64; k < K; k++) sum += (int32_t)ar[k] * (int32_t)br[k];
            c[(size_t)m * N + n] = sum;
        }
    }
}

/// Four rows of A against one vector of B, so each B load feeds four sdots
/// instead of one.
///
/// The x4 kernel above issues two loads per sdot and reuses nothing, which
/// caps it at 1.5 sdot per cycle on a core that issues three vector loads and
/// two sdots per cycle. Blocking M by four drops that to 1.25 loads per sdot,
/// which puts the load ceiling above the arithmetic one and hands the limit
/// back to the dot product unit.
///
/// The four accumulators are four different output cells rather than four
/// partial sums of one, so they are independent for the same reason.
void gemm_neon_sdot_m4(const int8_t *a, const int8_t *b, int32_t *c,
                       int M, int N, int K) {
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

            int32_t t0 = vaddvq_s32(s0), t1 = vaddvq_s32(s1);
            int32_t t2 = vaddvq_s32(s2), t3 = vaddvq_s32(s3);
            for (int k = k16; k < K; k++) {
                const int32_t bk = (int32_t)br[k];
                t0 += (int32_t)a0[k] * bk;
                t1 += (int32_t)a1[k] * bk;
                t2 += (int32_t)a2[k] * bk;
                t3 += (int32_t)a3[k] * bk;
            }
            c[(size_t)(m + 0) * N + n] = t0;
            c[(size_t)(m + 1) * N + n] = t1;
            c[(size_t)(m + 2) * N + n] = t2;
            c[(size_t)(m + 3) * N + n] = t3;
        }
    }

    // Rows past the last full block of four, on the plain one-row path.
    for (; m < M; m++) {
        const int8_t *ar = a + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < k16; k += 16) {
                acc = vdotq_s32(acc, vld1q_s8(ar + k), vld1q_s8(br + k));
            }
            int32_t sum = vaddvq_s32(acc);
            for (int k = k16; k < K; k++) sum += (int32_t)ar[k] * (int32_t)br[k];
            c[(size_t)m * N + n] = sum;
        }
    }
}

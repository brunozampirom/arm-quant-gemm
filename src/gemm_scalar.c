#include <stddef.h>

#include "gemm.h"

// Compiled twice: once with vectorization off as gemm_scalar, once with it on
// as gemm_auto. Same C, two compiler settings, so the difference between them
// is exactly what the auto-vectorizer bought.
#ifndef GEMM_NAME
#define GEMM_NAME gemm_scalar
#endif

/// The reference every other kernel is checked against.
void GEMM_NAME(const int8_t *a, const int8_t *b, int32_t *c,
                 int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            const int8_t *ar = a + (size_t)m * K;
            const int8_t *br = b + (size_t)n * K;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)ar[k] * (int32_t)br[k];
            }
            c[(size_t)m * N + n] = acc;
        }
    }
}

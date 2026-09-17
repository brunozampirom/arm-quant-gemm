#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

/// C[m*N + n] = sum over k of A[m*K + k] * B[n*K + k]
///
/// B is stored transposed, as N rows of K, so every kernel walks both operands
/// contiguously. Comparing kernels that disagree about memory layout would
/// measure the layout, not the arithmetic.
typedef void (*gemm_fn)(const int8_t *a, const int8_t *b, int32_t *c,
                        int M, int N, int K);

void gemm_scalar(const int8_t *a, const int8_t *b, int32_t *c,
                 int M, int N, int K);

/// Same source as gemm_scalar, built with the auto-vectorizer enabled.
void gemm_auto(const int8_t *a, const int8_t *b, int32_t *c,
               int M, int N, int K);

void gemm_neon_smull(const int8_t *a, const int8_t *b, int32_t *c,
                     int M, int N, int K);

void gemm_neon_sdot(const int8_t *a, const int8_t *b, int32_t *c,
                    int M, int N, int K);

/// Same arithmetic as the two above, with four independent accumulators so no
/// iteration waits on the one before it.
void gemm_neon_smull_x4(const int8_t *a, const int8_t *b, int32_t *c,
                        int M, int N, int K);

void gemm_neon_sdot_x4(const int8_t *a, const int8_t *b, int32_t *c,
                       int M, int N, int K);

#endif

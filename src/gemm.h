#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

#include "quant.h"

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

/// Blocks M by four so each B load feeds four sdots, testing whether the x4
/// kernel is limited by operand delivery rather than by the dot product unit.
void gemm_neon_sdot_m4(const int8_t *a, const int8_t *b, int32_t *c,
                       int M, int N, int K);

/// Quantized kernels: the same matmul, but the int32 accumulator is
/// requantized back to int8 before it leaves the kernel. quant.h says what
/// that involves and why it belongs inside rather than in a pass afterwards.
void gemm_q_scalar(const int8_t *a, const int8_t *b, int8_t *c,
                   int M, int N, int K, const gemm_quant *q);

void gemm_q_neon_sdot_m4(const int8_t *a, const int8_t *b, int8_t *c,
                         int M, int N, int K, const gemm_quant *q);

/// Same kernel, scalar epilogue, so the pair isolates what the epilogue costs.
void gemm_q_neon_sdot_m4_se(const int8_t *a, const int8_t *b, int8_t *c,
                            int M, int N, int K, const gemm_quant *q);

#endif

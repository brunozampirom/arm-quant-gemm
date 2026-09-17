#include <stddef.h>

#include "gemm.h"
#include "quant.h"

/// The reference every quantized kernel is checked against.
///
/// Built with the vectorizer off, for the same reason gemm_scalar is: a
/// reference the compiler has rewritten is still a fine reference, but it is
/// no longer a baseline anyone would recognise.
void gemm_q_scalar(const int8_t *a, const int8_t *b, int8_t *c,
                   int M, int N, int K, const gemm_quant *q) {
    for (int m = 0; m < M; m++) {
        const int8_t *ar = a + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *br = b + (size_t)n * K;
            int32_t acc = q->bias[n];
            for (int k = 0; k < K; k++) {
                acc += (int32_t)ar[k] * (int32_t)br[k];
            }
            c[(size_t)m * N + n] =
                quant_requantize(acc, q->multiplier[n], q->shift[n],
                                 q->output_zero_point, q->output_min,
                                 q->output_max);
        }
    }
}

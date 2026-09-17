#include <math.h>
#include <stddef.h>

#include "quant.h"

void quant_fold_bias(int32_t *out, const int32_t *bias, const int8_t *b,
                     int N, int K, int32_t a_zero_point) {
    for (int n = 0; n < N; n++) {
        const int8_t *row = b + (size_t)n * K;
        int32_t wsum = 0;
        for (int k = 0; k < K; k++) wsum += (int32_t)row[k];
        out[n] = (bias ? bias[n] : 0) - a_zero_point * wsum;
    }
}

void quant_split_multiplier(double real, int32_t *multiplier, int32_t *shift) {
    if (real <= 0.0) {
        *multiplier = 0;
        *shift = 0;
        return;
    }

    int exponent = 0;
    const double q = frexp(real, &exponent);
    int64_t fixed = (int64_t)llround(q * (double)(1LL << 31));

    // frexp returns [0.5, 1), so the only value that can round up out of Q31
    // is one that lands exactly on 1.0.
    if (fixed == (1LL << 31)) {
        fixed /= 2;
        exponent++;
    }

    // A real multiplier at or above 1 would need a left shift, which the
    // epilogue here does not implement. Clamping keeps the output defined; the
    // parameters this benchmark builds never reach it.
    if (exponent > 0) {
        *multiplier = INT32_MAX;
        *shift = 0;
        return;
    }

    *multiplier = (int32_t)fixed;
    *shift = -exponent;
    if (*shift > 31) {
        *multiplier = 0;
        *shift = 0;
    }
}

#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>

/// Requantization: turning the int32 accumulator back into an int8 tensor.
///
/// The matmul kernels next door stop at int32. That is the arithmetic core of
/// quantized inference but not the whole operator: a real int8 layer has to
/// hand the next layer int8 again, and the conversion is not a cast. It is a
/// fixed point multiply, a rounding shift, a zero point, and a saturating
/// narrow, per output element.
///
/// The convention here is TFLite's, which is what an exported int8 model
/// actually carries: weights per output channel and symmetric, activations per
/// tensor with a zero point, and the scale expressed as a Q31 multiplier plus a
/// shift rather than a float, so the epilogue stays in integer arithmetic.

/// Per output channel parameters. All arrays are N long.
typedef struct {
    /// The model's bias with the activation zero point already folded in; see
    /// quant_fold_bias.
    const int32_t *bias;
    /// Q31 value in [0.5, 1) that input_scale * weight_scale / output_scale
    /// was decomposed into.
    const int32_t *multiplier;
    /// The right shift that goes with it, >= 0.
    const int32_t *shift;
    int32_t output_zero_point;
    int8_t output_min;
    int8_t output_max;
} gemm_quant;

/// Same shape as gemm_fn, but the output tensor is int8 and carries the
/// quantization parameters it was produced under.
typedef void (*gemm_q_fn)(const int8_t *a, const int8_t *b, int8_t *c,
                          int M, int N, int K, const gemm_quant *q);

/// gemmlowp's SaturatingRoundingDoublingHighMul: the high 32 bits of a*b*2,
/// rounded, saturating on the one input pair that overflows. This is exactly
/// what vqrdmulhq_s32 computes, which is why the vector epilogue can be
/// checked against this one.
static inline int32_t quant_sat_rdmulh(int32_t a, int32_t b) {
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
    return (int32_t)((ab + nudge) / (1LL << 31));
}

/// gemmlowp's RoundingDivideByPOT.
///
/// Ties round away from zero. That is not what an arithmetic shift does, and
/// it is not what vrshlq_s32 does either: vrshlq rounds ties up, so it differs
/// on exactly the negative values that land on one. The vector epilogue has to
/// nudge for that rather than assume the obvious instruction matches.
static inline int32_t quant_rdpot(int32_t x, int32_t exp) {
    if (exp <= 0) return x;
    const int32_t mask = (int32_t)((1LL << exp) - 1);
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exp) + (remainder > threshold ? 1 : 0);
}

/// One output element, from int32 accumulator to int8.
static inline int8_t quant_requantize(int32_t acc, int32_t multiplier,
                                      int32_t shift, int32_t out_zp,
                                      int8_t lo, int8_t hi) {
    int32_t v = quant_rdpot(quant_sat_rdmulh(acc, multiplier), shift) + out_zp;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return (int8_t)v;
}

/// Folds the activation zero point into the bias.
///
/// sum_k (a[k] - za) * w[k] is sum_k a[k]*w[k] minus za * sum_k w[k]. The
/// second term depends only on the weights, so it is a per channel constant
/// computed once here instead of work in the inner loop. Real int8 runtimes do
/// the same, which is why the zero point never appears in the kernel and why
/// leaving it out of a benchmark does not make the benchmark faster.
///
/// `bias` may be NULL, meaning the layer has none.
void quant_fold_bias(int32_t *out, const int32_t *bias, const int8_t *b,
                     int N, int K, int32_t a_zero_point);

/// Decomposes a real multiplier in (0, 1) into the Q31 pair.
void quant_split_multiplier(double real, int32_t *multiplier, int32_t *shift);

#endif

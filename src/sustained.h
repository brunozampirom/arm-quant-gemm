#ifndef SUSTAINED_H
#define SUSTAINED_H

#include <stdint.h>

#include "gemm.h"

int run_sustained_mt(gemm_fn fn, const int8_t *a, const int8_t *b,
                     int M, int N, int K,
                     const int *cpus, int n_cpus, int seconds);

#endif

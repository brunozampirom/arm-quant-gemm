#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemm.h"
#include "measure.h"
#include "sustained.h"

typedef struct {
    gemm_fn fn;
    const int8_t *a, *b;
    int32_t *out;
    int M, N, K;
    int cpu;
    atomic_long iters;
    atomic_int stop;
    pthread_t th;
} Worker;

static void *worker_main(void *arg) {
    Worker *w = (Worker *)arg;
    if (w->cpu >= 0) pin_to_cpu(w->cpu);
    while (!atomic_load_explicit(&w->stop, memory_order_relaxed)) {
        w->fn(w->a, w->b, w->out, w->M, w->N, w->K);
        atomic_fetch_add_explicit(&w->iters, 1, memory_order_relaxed);
    }
    return NULL;
}

/// Loads `n_cpus` cores at once for `seconds` and reports aggregate throughput
/// per second. One core is not enough to heat a phone; throttling only shows
/// up when the whole cluster is busy, which is also what real inference does.
int run_sustained_mt(gemm_fn fn, const int8_t *a, const int8_t *b,
                     int M, int N, int K,
                     const int *cpus, int n_cpus, int seconds) {
    Worker *ws = calloc((size_t)n_cpus, sizeof(Worker));
    if (!ws) return -1;

    size_t nc = (size_t)M * N;
    for (int i = 0; i < n_cpus; i++) {
        ws[i].fn = fn;
        ws[i].a = a;
        ws[i].b = b;
        // Each worker writes its own output so the cores do not fight over
        // the same cache lines, which would measure coherence, not compute.
        ws[i].out = malloc(nc * sizeof(int32_t));
        if (!ws[i].out) return -1;
        ws[i].M = M; ws[i].N = N; ws[i].K = K;
        ws[i].cpu = cpus[i];
        atomic_init(&ws[i].iters, 0);
        atomic_init(&ws[i].stop, 0);
    }

    int freq_lo = freq_open(cpus[0]);
    int freq_hi = freq_open(cpus[n_cpus - 1]);
    double ops_per_iter = 2.0 * (double)M * (double)N * (double)K;

    printf("second,gops_total,freq_first_khz,freq_last_khz\n");
    fflush(stdout);

    for (int i = 0; i < n_cpus; i++) {
        if (pthread_create(&ws[i].th, NULL, worker_main, &ws[i]) != 0) return -1;
    }

    long last_total = 0;
    double bucket = now_wall();
    for (int s = 0; s < seconds; s++) {
        // Sleep rather than spin. Spinning here would put a ninth runnable
        // thread on an eight core device and steal time from the workers,
        // which is exactly the thing this run is supposed to measure. The
        // bucket length is measured afterwards, so sleep jitter costs nothing.
        struct timespec iv = {1, 0};
        while (nanosleep(&iv, &iv) == -1 && errno == EINTR) { }

        long total = 0;
        for (int i = 0; i < n_cpus; i++) {
            total += atomic_load_explicit(&ws[i].iters, memory_order_relaxed);
        }
        double elapsed = now_wall() - bucket;
        double gops = ops_per_iter * (double)(total - last_total) / elapsed * 1e-9;
        printf("%d,%.2f,%ld,%ld\n", s, gops, freq_read(freq_lo),
               freq_read(freq_hi));
        fflush(stdout);
        last_total = total;
        bucket = now_wall();
    }

    for (int i = 0; i < n_cpus; i++) {
        atomic_store_explicit(&ws[i].stop, 1, memory_order_relaxed);
    }
    for (int i = 0; i < n_cpus; i++) pthread_join(ws[i].th, NULL);
    for (int i = 0; i < n_cpus; i++) free(ws[i].out);
    free(ws);
    return 0;
}

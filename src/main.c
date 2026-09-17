#define _GNU_SOURCE

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/auxv.h>

#include "gemm.h"
#include "measure.h"
#include "sustained.h"

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif

/// Exactly one of fn and qfn is set. The quantized kernels write int8 under
/// the parameters in Quant, the others write int32 and stop there.
typedef struct {
    const char *name;
    gemm_fn fn;
    gemm_q_fn qfn;
    int needs_dotprod;
} Kernel;

static const Kernel kernels[] = {
    {"scalar", gemm_scalar, NULL, 0},
    {"auto_vec", gemm_auto, NULL, 0},
    {"neon_smull", gemm_neon_smull, NULL, 0},
    {"neon_smull_x4", gemm_neon_smull_x4, NULL, 0},
    {"neon_sdot", gemm_neon_sdot, NULL, 1},
    {"neon_sdot_x4", gemm_neon_sdot_x4, NULL, 1},
    {"neon_sdot_m4", gemm_neon_sdot_m4, NULL, 1},
    {"q_scalar", NULL, gemm_q_scalar, 0},
    {"q_sdot_m4_se", NULL, gemm_q_neon_sdot_m4_se, 1},
    {"q_sdot_m4", NULL, gemm_q_neon_sdot_m4, 1},
};
static const int n_kernels = (int)(sizeof(kernels) / sizeof(kernels[0]));

typedef struct {
    int M, N, K, reps, cpu, seconds;
    const char *only;
    int sustained;
    int csv;
    int check_only;
    const char *cpus;
} Opts;

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [options]\n"
            "  --cpu N       pin to this cpu (required for trustworthy numbers)\n"
            "  -M -N -K n    matrix dimensions (default 64 64 1024)\n"
            "  --reps n      timed repetitions per kernel (default 50)\n"
            "  --only name   run one kernel\n"
            "  --sustained s run one kernel for s seconds and report throughput\n"
            "                over time, which is where thermal throttling shows\n"
            "  --cpus a,b,c  load these cpus at once during --sustained\n"
            "  --csv         machine-readable output\n"
            "  --check       verify every kernel against the scalar reference\n"
            "                and exit; no timing, so it runs under emulation\n",
            argv0);
}

static void fill(int8_t *p, size_t n, uint64_t seed) {
    for (size_t i = 0; i < n; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        p[i] = (int8_t)((seed >> 33) & 0xFF);
    }
}

typedef struct {
    int8_t *a, *b;
    int32_t *ref, *out;
    /// int8 outputs, for the quantized kernels.
    int8_t *qref, *qout;
    int32_t *bias, *mult, *shift;
    gemm_quant q;
} Buffers;

/// Builds quantization parameters the way an exported model would carry them:
/// one multiplier and shift per output channel, an activation zero point
/// folded into the bias, and an output zero point.
///
/// The scales are derived from the accumulators this data actually produces,
/// targeting a little past full scale so a few outputs clamp. Parameters that
/// never reach the clamp would leave that path untested, and parameters that
/// clamp everything would make every kernel agree on a constant.
static void build_quant(Buffers *bufs, const Opts *o) {
    const int32_t a_zero_point = 7;

    for (int n = 0; n < o->N; n++) bufs->bias[n] = (n % 17) - 8;
    quant_fold_bias(bufs->bias, bufs->bias, bufs->b, o->N, o->K, a_zero_point);

    for (int n = 0; n < o->N; n++) {
        int32_t peak = 1;
        for (int m = 0; m < o->M; m++) {
            int32_t v = bufs->ref[(size_t)m * o->N + n] + bufs->bias[n];
            if (v < 0) v = -v;
            if (v > peak) peak = v;
        }
        quant_split_multiplier(140.0 / (double)peak, &bufs->mult[n],
                               &bufs->shift[n]);
    }

    bufs->q.bias = bufs->bias;
    bufs->q.multiplier = bufs->mult;
    bufs->q.shift = bufs->shift;
    bufs->q.output_zero_point = -3;
    bufs->q.output_min = -128;
    bufs->q.output_max = 127;
}

static int alloc_buffers(Buffers *bufs, const Opts *o) {
    size_t na = (size_t)o->M * o->K, nb = (size_t)o->N * o->K;
    size_t nc = (size_t)o->M * o->N;
    bufs->a = malloc(na);
    bufs->b = malloc(nb);
    bufs->ref = malloc(nc * sizeof(int32_t));
    bufs->out = malloc(nc * sizeof(int32_t));
    bufs->qref = malloc(nc);
    bufs->qout = malloc(nc);
    bufs->bias = malloc((size_t)o->N * sizeof(int32_t));
    bufs->mult = malloc((size_t)o->N * sizeof(int32_t));
    bufs->shift = malloc((size_t)o->N * sizeof(int32_t));
    if (!bufs->a || !bufs->b || !bufs->ref || !bufs->out || !bufs->qref ||
        !bufs->qout || !bufs->bias || !bufs->mult || !bufs->shift) {
        return -1;
    }
    fill(bufs->a, na, 1);
    fill(bufs->b, nb, 2);
    gemm_scalar(bufs->a, bufs->b, bufs->ref, o->M, o->N, o->K);
    build_quant(bufs, o);
    gemm_q_scalar(bufs->a, bufs->b, bufs->qref, o->M, o->N, o->K, &bufs->q);
    return 0;
}

static void free_buffers(Buffers *b) {
    free(b->a); free(b->b); free(b->ref); free(b->out);
    free(b->qref); free(b->qout);
    free(b->bias); free(b->mult); free(b->shift);
}

static void invoke(const Kernel *kn, Buffers *b, const Opts *o) {
    if (kn->qfn) kn->qfn(b->a, b->b, b->qout, o->M, o->N, o->K, &b->q);
    else kn->fn(b->a, b->b, b->out, o->M, o->N, o->K);
}

static int correct(const Kernel *kn, Buffers *b, const Opts *o) {
    size_t nc = (size_t)o->M * o->N;
    if (kn->qfn) {
        memset(b->qout, 0, nc);
        kn->qfn(b->a, b->b, b->qout, o->M, o->N, o->K, &b->q);
        return memcmp(b->qref, b->qout, nc) == 0;
    }
    memset(b->out, 0, nc * sizeof(int32_t));
    kn->fn(b->a, b->b, b->out, o->M, o->N, o->K);
    return memcmp(b->ref, b->out, nc * sizeof(int32_t)) == 0;
}

static double gops_for(const Opts *o, double seconds) {
    double ops = 2.0 * (double)o->M * (double)o->N * (double)o->K;
    return seconds > 0 ? ops / seconds * 1e-9 : 0.0;
}

static void run_kernel(const Kernel *kn, Buffers *b, const Opts *o,
                       int freq_fd, long peak) {
    if (!correct(kn, b, o)) {
        printf("%-14s DISAGREES with the scalar reference\n", kn->name);
        return;
    }

    for (int r = 0; r < 5; r++) invoke(kn, b, o);

    Rep *reps = malloc((size_t)o->reps * sizeof(Rep));
    if (!reps) return;

    for (int r = 0; r < o->reps; r++) {
        reps[r].freq_before = freq_read(freq_fd);
        double w0 = now_wall(), c0 = now_cpu();
        invoke(kn, b, o);
        reps[r].wall = now_wall() - w0;
        reps[r].cpu = now_cpu() - c0;
        reps[r].freq_after = freq_read(freq_fd);
    }

    Summary s = summarise(reps, o->reps, peak);
    free(reps);

    if (s.ok == 0) {
        printf("%-14s no usable reps out of %d "
               "(preempted %d, dvfs %d, downclocked %d)\n",
               kn->name, s.total, s.preempted, s.dvfs, s.downclocked);
        return;
    }

    if (o->csv) {
        printf("%s,%d,%d,%.6f,%.6f,%.6f,%.2f,%d,%d,%ld\n",
               kn->name, s.ok, s.total, s.p10 * 1e3, s.median * 1e3,
               s.p90 * 1e3, gops_for(o, s.median), s.preempted + s.dvfs,
               s.downclocked, s.freq_khz);
        return;
    }

    printf("%-14s median %7.3f ms  p10 %7.3f  p90 %7.3f  %6.2f GOP/s  "
           "spread %4.1f%%  kept %d/%d",
           kn->name, s.median * 1e3, s.p10 * 1e3, s.p90 * 1e3,
           gops_for(o, s.median), s.spread_pct, s.ok, s.total);
    if (s.preempted || s.dvfs || s.downclocked) {
        printf("  (preempt %d, dvfs %d, slow %d)",
               s.preempted, s.dvfs, s.downclocked);
    }
    printf("\n");
}

/// Repeats one kernel for a fixed wall-clock span and reports throughput per
/// second. A benchmark that only reports a burst cannot see throttling, which
/// is the thing that actually decides what a phone delivers.
static void run_sustained(const Kernel *kn, Buffers *b, const Opts *o,
                          int freq_fd) {
    printf("second,gops,freq_khz\n");

    double start = now_wall();
    double bucket_start = start;
    int iters = 0;
    int second = 0;

    while (now_wall() - start < (double)o->seconds) {
        invoke(kn, b, o);
        iters++;

        double elapsed = now_wall() - bucket_start;
        if (elapsed >= 1.0) {
            double ops = 2.0 * (double)o->M * (double)o->N * (double)o->K * iters;
            printf("%d,%.2f,%ld\n", second, ops / elapsed * 1e-9,
                   freq_read(freq_fd));
            fflush(stdout);
            second++;
            iters = 0;
            bucket_start = now_wall();
        }
    }
}

int main(int argc, char **argv) {
    Opts o = {64, 64, 1024, 50, -1, 0, NULL, 0, 0, 0, NULL};

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_next = i + 1 < argc;
        if (!strcmp(a, "--cpu") && has_next) o.cpu = atoi(argv[++i]);
        else if (!strcmp(a, "-M") && has_next) o.M = atoi(argv[++i]);
        else if (!strcmp(a, "-N") && has_next) o.N = atoi(argv[++i]);
        else if (!strcmp(a, "-K") && has_next) o.K = atoi(argv[++i]);
        else if (!strcmp(a, "--reps") && has_next) o.reps = atoi(argv[++i]);
        else if (!strcmp(a, "--only") && has_next) o.only = argv[++i];
        else if (!strcmp(a, "--sustained") && has_next) {
            o.sustained = 1;
            o.seconds = atoi(argv[++i]);
        } else if (!strcmp(a, "--csv")) o.csv = 1;
        else if (!strcmp(a, "--check")) o.check_only = 1;
        else if (!strcmp(a, "--cpus") && has_next) o.cpus = argv[++i];
        else { usage(argv[0]); return 2; }
    }

    if (o.M <= 0 || o.N <= 0 || o.K <= 0 || o.reps <= 0) {
        usage(argv[0]);
        return 2;
    }

    unsigned long hwcap = getauxval(AT_HWCAP);
    int have_dotprod = (hwcap & HWCAP_ASIMDDP) != 0;

    if (o.cpu >= 0 && pin_to_cpu(o.cpu) != 0) {
        fprintf(stderr, "error: could not pin to cpu%d\n", o.cpu);
        return 1;
    }

    int freq_fd = o.cpu >= 0 ? freq_open(o.cpu) : -1;
    long peak = o.cpu >= 0 ? freq_max(o.cpu) : 0;

    Buffers bufs;
    if (alloc_buffers(&bufs, &o) != 0) {
        fprintf(stderr, "error: out of memory\n");
        return 1;
    }

    if (!o.csv) {
        printf("M=%d N=%d K=%d reps=%d cpu=%d dotprod=%s peak=%ld kHz\n",
               o.M, o.N, o.K, o.reps, o.cpu, have_dotprod ? "yes" : "no", peak);
        if (o.cpu < 0 && !o.cpus) {
            printf("warning: not pinned; the scheduler may move this between "
                   "core types mid-run\n");
        }
    }

    if (o.check_only) {
        int bad = 0;
        for (int i = 0; i < n_kernels; i++) {
            const Kernel *kn = &kernels[i];
            if (kn->needs_dotprod && !have_dotprod) {
                printf("%-14s skipped, this cpu has no dotprod\n", kn->name);
                continue;
            }
            int ok = correct(kn, &bufs, &o);
            printf("%-14s %s\n", kn->name,
                   ok ? "matches the reference" : "WRONG");
            if (!ok) bad = 1;
        }
        free_buffers(&bufs);
        return bad;
    }

    if (o.sustained && o.cpus) {
        int cpus[32], n = 0;
        char *spec = strdup(o.cpus);
        char *save = NULL;
        for (char *t = strtok_r(spec, ",", &save); t && n < 32;
             t = strtok_r(NULL, ",", &save)) {
            cpus[n++] = atoi(t);
        }
        const Kernel *kn = NULL;
        for (int i = 0; i < n_kernels; i++) {
            const char *want = o.only ? o.only : "neon_sdot_m4";
            if (!strcmp(want, kernels[i].name)) kn = &kernels[i];
        }
        // The multi-core path shares one output buffer between threads, which
        // is fine for a throughput measurement and wrong for a quantized
        // kernel's int8 output only in the same harmless way. What it cannot
        // do is call a null pointer, so say so instead.
        if (!kn || !kn->fn) {
            fprintf(stderr, "error: --cpus needs an int32 output kernel\n");
            free(spec);
            free_buffers(&bufs);
            return 2;
        }
        int rc = run_sustained_mt(kn->fn, bufs.a, bufs.b, o.M, o.N, o.K,
                                  cpus, n, o.seconds);
        free(spec);
        free_buffers(&bufs);
        return rc == 0 ? 0 : 1;
    }

    for (int i = 0; i < n_kernels; i++) {
        const Kernel *kn = &kernels[i];
        if (o.only && strcmp(o.only, kn->name) != 0) continue;
        if (kn->needs_dotprod && !have_dotprod) {
            if (!o.csv) printf("%-14s skipped, this cpu has no dotprod\n", kn->name);
            continue;
        }
        if (o.sustained) run_sustained(kn, &bufs, &o, freq_fd);
        else run_kernel(kn, &bufs, &o, freq_fd, peak);
    }

    if (freq_fd >= 0) close(freq_fd);
    free_buffers(&bufs);
    return 0;
}

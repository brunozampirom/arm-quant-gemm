#define _GNU_SOURCE

#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "measure.h"

/// A rep whose wall time exceeds its CPU time by more than this was descheduled
/// while it ran, so its duration says more about the scheduler than the kernel.
static const double PREEMPT_RATIO = 1.02;

double now_wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double now_cpu(void) {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

/// cpufreq policy directories are named after the first CPU in the cluster, so
/// the policy for a given CPU is the highest-numbered one at or below it.
static int policy_path(int cpu, const char *leaf, char *out, size_t n) {
    for (int p = cpu; p >= 0; p--) {
        snprintf(out, n, "/sys/devices/system/cpu/cpufreq/policy%d/%s", p, leaf);
        if (access(out, R_OK) == 0) return 0;
    }
    return -1;
}

int freq_open(int cpu) {
    char path[160];
    if (policy_path(cpu, "scaling_cur_freq", path, sizeof(path)) != 0) return -1;
    return open(path, O_RDONLY);
}

long freq_read(int fd) {
    if (fd < 0) return 0;
    char buf[32];
    if (pread(fd, buf, sizeof(buf) - 1, 0) <= 0) return 0;
    buf[sizeof(buf) - 1] = 0;
    return strtol(buf, NULL, 10);
}

long freq_max(int cpu) {
    char path[160];
    if (policy_path(cpu, "cpuinfo_max_freq", path, sizeof(path)) != 0) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    long v = 0;
    if (fscanf(f, "%ld", &v) != 1) v = 0;
    fclose(f);
    return v;
}

static int cmp_double(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

static double pct(const double *sorted, int n, double q) {
    if (n <= 0) return 0.0;
    double idx = q * (n - 1);
    int lo = (int)idx;
    int hi = lo + 1 < n ? lo + 1 : lo;
    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

Summary summarise(Rep *reps, int n, long peak_khz) {
    Summary s;
    memset(&s, 0, sizeof(s));
    s.total = n;

    double *keep = malloc((size_t)n * sizeof(double));
    if (!keep) return s;

    for (int i = 0; i < n; i++) {
        Rep *r = &reps[i];

        if (r->cpu > 0 && r->wall / r->cpu > PREEMPT_RATIO) {
            s.preempted++;
            continue;
        }
        if (r->freq_before && r->freq_after && r->freq_before != r->freq_after) {
            s.dvfs++;
            continue;
        }
        if (peak_khz && r->freq_before && r->freq_before < peak_khz) {
            s.downclocked++;
            continue;
        }

        keep[s.ok] = r->wall;
        if (!s.freq_khz) s.freq_khz = r->freq_before;
        s.ok++;
    }

    if (s.ok > 0) {
        qsort(keep, (size_t)s.ok, sizeof(double), cmp_double);
        s.min = keep[0];
        s.p10 = pct(keep, s.ok, 0.10);
        s.median = pct(keep, s.ok, 0.50);
        s.p90 = pct(keep, s.ok, 0.90);
        s.spread_pct = s.p10 > 0 ? (s.p90 / s.p10 - 1.0) * 100.0 : 0.0;
    }

    free(keep);
    return s;
}

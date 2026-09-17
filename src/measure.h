#ifndef MEASURE_H
#define MEASURE_H

#include <stdint.h>

/// One timed repetition, with the evidence needed to decide whether to trust it.
typedef struct {
    double wall;        ///< seconds, CLOCK_MONOTONIC
    double cpu;         ///< seconds, CLOCK_THREAD_CPUTIME_ID
    long freq_before;   ///< kHz, 0 if unreadable
    long freq_after;
} Rep;

typedef enum {
    REP_OK = 0,
    REP_PREEMPTED,   ///< wall time ran ahead of cpu time
    REP_DVFS,        ///< the core changed frequency mid-rep
    REP_DOWNCLOCKED, ///< ran below the cluster's peak frequency
} RepVerdict;

typedef struct {
    int total, ok, preempted, dvfs, downclocked;
    double p10, median, p90, min;
    double spread_pct;  ///< p90 over p10, as a percentage above 1
    long freq_khz;      ///< frequency the accepted reps ran at
} Summary;

double now_wall(void);
double now_cpu(void);

/// Opens the cpufreq policy that governs `cpu`. Returns -1 when unreadable,
/// which is normal on locked-down devices; callers degrade rather than fail.
int freq_open(int cpu);
long freq_read(int fd);
long freq_max(int cpu);

int pin_to_cpu(int cpu);

/// Classifies reps and summarises only the ones worth keeping.
Summary summarise(Rep *reps, int n, long peak_khz);

#endif

# arm-quant-gemm

Int8 matrix multiply on a phone, six ways, measured properly.

Two things it found. The first is that hand-written NEON can be slower than
plain C:

```
Cortex-A55 (in-order little core)
  auto_vec        5.62 GOP/s     plain C, compiler auto-vectorized
  neon_smull      4.78 GOP/s     hand-written NEON intrinsics
```

The intrinsics are not wrong. They use one accumulator, so every iteration waits
on the one before it. The compiler used eight and did not wait.

The second is that a burst measurement overstates what the phone sustains, by
29% with all eight cores loaded:

```
8 cores, 300s under load
  best single second   193.4 GOP/s
  steady state         149.8 GOP/s     reached after 81s
```

## Results

Galaxy A14 5G, Exynos 1330, Android 15, six Cortex-A55 plus two Cortex-A78.
M=64, N=64, K=1024, int8 accumulating into int32. Median of accepted
repetitions, device unplugged from USB. Produced by `run.sh`, raw output in
[results/](results/).

| kernel | A55 GOP/s | A78 GOP/s | what it is |
|---|---|---|---|
| `scalar` | 0.79 | 4.01 | C, vectorizer off |
| `auto_vec` | 5.62 | 20.95 | same C, vectorizer on |
| `neon_smull` | 4.78 | 29.63 | intrinsics, one accumulator |
| `neon_smull_x4` | 5.76 | 32.51 | intrinsics, four accumulators |
| `neon_sdot` | 8.03 | 43.93 | ARMv8.2 dot product, one accumulator |
| `neon_sdot_x4` | **9.31** | **77.45** | dot product, four accumulators |

## What the numbers say

**Against the right baseline the win is 3.7x, not 19x.** `neon_sdot_x4` is 19.3x
faster than scalar C on the A78. That figure is close to meaningless, because
nobody ships scalar C: the compiler vectorizes it for you. Against what you
actually get for free it is 3.7x. A benchmark quoting the first number is
comparing against code no one would have written.

This is why `scalar` and `auto_vec` are one source file compiled twice, once
with `-fno-vectorize`. It was not the original plan. The first `scalar` kernel
turned out to emit 24 SIMD instructions at `-O3`, so an honest baseline had to
be built deliberately rather than assumed.

**Four accumulators helps the big core more than the little one**, which is the
opposite of the intuition that in-order cores need the hand-holding:

| | one accumulator | four | gain |
|---|---|---|---|
| A55 `sdot` | 8.03 | 9.31 | 1.16x |
| A78 `sdot` | 43.93 | 77.45 | 1.76x |

Arm's optimization guides say why, and the reason is not issue order. Both
cores forward an accumulator between consecutive dot products at a latency of
one cycle. What differs is how many dot products each can start per cycle:

| | `SDOT` throughput, Q-form | accumulate latency | one chain saturates it? |
|---|---|---|---|
| A55 | 1 per cycle | 1 cycle | **yes** |
| A78 | 2 per cycle | 1 cycle | no, it fills half |

One accumulator issues at most one dot product per cycle, because the next one
waits a cycle for the forwarded accumulator. On the A55 that already matches
what the core can start, so a second chain has nowhere to go and the gain is
1.16x. On the A78 it leaves half the machine idle, and independent chains
collect the rest: 1.76x measured against 2x available.

The A55 number carries a footnote that is easy to miss. Its table lists a
throughput of 2, but note 1 says the Q-form, which is the 128-bit form these
kernels use, "can only be dual issued as instruction 0 and execution throughput
is 1". Reading the headline number would have predicted a gain that is not
there.

Sources: [Cortex-A78 Software Optimization
Guide](https://developer.arm.com/documentation/102160/latest/) r1p2 issue 5.0,
`ASIMD dot product` on page 34 and the accumulator forwarding note on page 19;
[Cortex-A55 Software Optimization
Guide](https://documentation-service.arm.com/static/6385cb83ff39817c1136abd8)
r2p0 issue 4.0, page 35.

**`sdot` earns its keep.** Over the four-accumulator `smull` kernel it is 1.62x
on the A55 and 2.38x on the A78. It is detected at runtime through `AT_HWCAP`,
and skipped rather than crashing on a core without it.

## How much of the core is that

Being 3.7x faster than the compiler says nothing about how much of the silicon
is left unused. The dot product throughput above gives a ceiling to measure
against.

A `SDOT Vd.4S, Vn.16B, Vm.16B` performs 4 lanes of 4 int8 multiply-accumulates,
so 16 MACs, counted here as 32 operations to match the `2 * M * N * K` the
benchmark reports. Peak is then throughput times 32 times the clock, and the
clock is not the nominal one: every repetition that ran below the cluster's
peak is discarded before the median, so the surviving repetitions ran at the
frequency in the results header.

| | peak | `neon_sdot` | `neon_sdot_x4` |
|---|---|---|---|
| A55 at 2002 MHz | 64.1 GOP/s | 8.03, **12.5%** | 9.31, **14.5%** |
| A78 at 2400 MHz | 153.6 GOP/s | 43.93, **28.6%** | 77.45, **50.4%** |

The A78 number is the interesting one. Four accumulators land it at almost
exactly half of peak, which is what one dot product per cycle looks like on a
core that can start two.

So the accumulators did their job and something else became the limit.
**Hypothesis, not measured:** operand delivery. Each `SDOT` reads 32 bytes, so
two per cycle wants 64 bytes per cycle of loads, and these kernels stream both
operands with no reuse between them. That is the same gap the Scope section
below describes as missing blocking, now with a number attached to it rather
than a caveat.

The A55 at 14.5% is a much wider gap and this ceiling does not explain it. An
in-order core cannot reorder around a load that misses, and nothing here was
written to help it.

This ceiling applies only to the two `sdot` kernels. Applying it to the `smull`
kernels would be wrong, since they issue a different instruction with different
throughput.

## Sustained load is a different question

A burst of 50 repetitions on one core says 77.45 GOP/s, and that is true. It is
also not what the phone delivers. Each row below is 300 seconds of continuous
work, unplugged, summarised by `analyze.sh`:

| load | behaviour | sustained | best single second | thermal status |
|---|---|---|---|---|
| 1 core | settles at 2400 MHz in 1s | 76.5 | 76.5 (1.00x) | 0, never throttled |
| 2 cores | never settles | 113.4 late, drifting -12.6% | 143.4 (1.26x) | 1 |
| 8 cores | settles in 81s at 1920/1536 MHz | 149.8 | 193.4 (1.29x) | 2 |

The three rows are three different behaviours, not three points on one curve.

**One core runs flat forever.** It holds 2400 MHz for the whole five minutes and
Android never reports throttling. Anything that fits on one core is not a
thermal problem on this device.

**Two cores never reach a steady state.** The governor hunts between 1824 and
2112 MHz for the entire run, and throughput drifts down 12.6% from the first
hundred seconds to the last without ever settling. Quoting a single number for
this case would be inventing one.

**Eight cores settle, and settle low.** After 81 seconds the clocks pin at 1920
MHz on the A55s and 1536 MHz on the A78s, which is 64% of the A78 peak, and
throughput sits at 149.8 GOP/s with a p5 to p95 band of 144.6 to 150.3. From
there it does not move.

Picking the window changes the answer here, which is why `analyze.sh` looks for
the DVFS operating point the run actually converges to rather than comparing the
first seconds against the last. An earlier draft of this README did compare the
ends, and reported a 25.9% drop for the eight core case that was an artefact of
catching the tail during a one second dip.

Those dips are real and are left in the raw data. Sustained mode reports
throughput per second with no filtering, so a second where Android decided to do
something else shows up as a 56 GOP/s sample. The burst mode filters described
below do not apply here on purpose: this measurement is meant to include the
phone being a phone.

The throttling level comes from Android's `thermalservice` rather than from a
temperature reading. The thermal zones under `/sys/class/thermal` are root-only
on a retail phone, and battery temperature lags the SoC badly: it moved 2.9 C
across all of this while the A78 clock was being cut by more than a third.

## How it is measured

A phone is a hostile place to benchmark. Three things corrupt a result, and each
is detected rather than hoped away.

**The scheduler moves you.** Six A55s and two A78s. An unpinned run migrates
between them mid-measurement and reports the mixture. Every run pins to one CPU,
and a run without `--cpu` says so in its output.

**The governor changes frequency underneath you.** This device runs
`energy_aware`, and the governor cannot be changed without root. So every
repetition records `scaling_cur_freq` before and after itself, outside the timed
window. A repetition whose frequency moved, or that ran below the cluster's
peak, is discarded and counted.

**Other processes steal your core.** Every repetition records both
`CLOCK_MONOTONIC` and `CLOCK_THREAD_CPUTIME_ID`. When wall time runs more than
2% ahead of CPU time the thread was descheduled, and the repetition is thrown
out.

The output says how many survived and why the rest did not:

```
neon_sdot_x4   median   0.108 ms  p10   0.108  p90   0.108   77.45 GOP/s  spread  0.2%  kept 49/50  (preempt 1, dvfs 0, slow 0)
scalar         median   2.094 ms  p10   2.081  p90   2.101    4.01 GOP/s  spread  1.0%  kept 26/50  (preempt 0, dvfs 11, slow 13)
```

That second line is the point. Half the scalar repetitions ran at the wrong
clock, and nothing except the filter would have said so.

None of this is cosmetic. Before the filters existed the same measurements
spread 135% to 2102% between fastest and slowest repetition, and even reporting
the minimum understated the A78 dot product kernel by 27%, because entire series
had run below peak frequency with no sign of it in the output. After the filters
the spread is 0.2% to 3.0%.

Results are medians of surviving repetitions with p10 and p90, never means. On a
phone the tail is thermal and scheduling noise, and a mean folds it into the
answer.

Run to run, the numbers here move by up to 4%. Two independent full runs of
`run.sh` put A78 `neon_sdot_x4` at 74.77 and 77.45 GOP/s. Treat the ratios
between kernels as the result, not the third digit.

## Reproducing

Needs the Android NDK and a device on `adb`. For the sustained runs the device
must be off USB power, because charging heats it:

```sh
adb tcpip 5555 && adb connect <device-ip>:5555   # then unplug
sh run.sh
```

`run.sh` builds, pushes, checks correctness, measures both core types, runs the
three sustained scenarios with thermal sampling, and writes everything to
`results/`. It warns if the device is still plugged in.

To build alone:

```sh
sh build.sh                            # $ANDROID_NDK_HOME or $ANDROID_HOME/ndk
CC=aarch64-linux-gnu-gcc sh build.sh   # any aarch64 cross compiler
```

Only `gemm_neon_sdot.c` is compiled with `+dotprod`. Building everything with it
would let the compiler reach for `sdot` inside the baseline kernels, and the
comparison would mean nothing.

## Correctness

Every kernel is checked against the scalar reference before it is timed, and one
that disagrees is reported instead of measured. CI cross-compiles for aarch64
and runs those checks under qemu, including K=253 so the tail paths past the 16,
32 and 64 byte loop bodies are exercised.

CI deliberately publishes no timings. A timing under emulation is a timing of
the emulator.

## Scope

**One device.** Every number is from one Exynos 1330. Cortex-A55 and Cortex-A78
are common enough that the shape should carry, but the ratios are not a claim
about ARM in general.

**No i8mm.** This SoC reports `asimddp` but not `i8mm`, so the 8-bit matrix
multiply path is absent rather than untested.

**Nothing is tiled or blocked.** These kernels stream both operands and stay in
cache at these sizes, so they measure arithmetic rather than the memory
hierarchy. A real inference kernel needs blocking, and that would be a different
comparison.

**B is stored transposed** as N rows of K, so every kernel walks both operands
contiguously. Comparing kernels that disagree about layout would measure the
layout.

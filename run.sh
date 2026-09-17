#!/bin/sh
# Runs the whole measurement suite on a connected device and writes results/.
#
# Everything in the README comes out of this script. If a number there cannot
# be reproduced by running this, it does not belong there.
set -eu

# awk decides whether a field is a number using the locale's decimal separator.
# Under a comma locale "193.4" stops looking numeric, comparisons fall back to
# string order, and the maximum of 193.4 and 99.0 comes out as 99.0. The whole
# summary then reads plausibly and is wrong, so pin the numeric locale.
export LC_ALL=C

# Git Bash rewrites /data/... into a Windows path before adb ever sees it.
export MSYS_NO_PATHCONV=1

DEV=/data/local/tmp/gemmbench
OUT=${OUT:-results}
SUSTAIN_S=${SUSTAIN_S:-300}
LITTLE=${LITTLE:-0}
BIG=${BIG:-7}
BIG_CLUSTER=${BIG_CLUSTER:-6,7}
ALL_CPUS=${ALL_CPUS:-0,1,2,3,4,5,6,7}

adb get-state > /dev/null 2>&1 || { echo "no device; run adb connect first" >&2; exit 1; }

mkdir -p "$OUT"

sh build.sh
adb push build/gemmbench "$DEV" > /dev/null
adb shell chmod 755 "$DEV"

echo "== device =="
adb shell getprop ro.product.model | tr -d '\r'
adb shell getprop ro.soc.model | tr -d '\r'

powered=$(adb shell dumpsys battery | tr -d '\r' | awk '/USB powered:/{print $3}')
if [ "$powered" = "true" ]; then
    echo
    echo "WARNING: the device is on USB power. Charging heats it, which shifts"
    echo "the sustained numbers. Switch adb to wifi and unplug:"
    echo "  adb tcpip 5555 && adb connect <device-ip>:5555"
    echo
fi

echo "== correctness =="
adb shell "$DEV" --check | tr -d '\r'
adb shell "$DEV" --check -M 7 -N 5 -K 253 | tr -d '\r' | tail -6

echo
echo "== burst, little core (cpu$LITTLE) =="
adb shell "$DEV" --cpu "$LITTLE" | tr -d '\r' | tee "$OUT/burst_little.txt"

echo
echo "== burst, big core (cpu$BIG) =="
adb shell "$DEV" --cpu "$BIG" | tr -d '\r' | tee "$OUT/burst_big.txt"

# Samples the thermal state alongside a sustained run. Android exposes a
# throttling level through thermalservice even when the thermal zones
# themselves are root-only, which they are on a retail phone.
sample_thermal() {
    pid=$1
    out=$2
    echo "t_s,battery_c,thermal_status" > "$out"
    t=0
    while kill -0 "$pid" 2>/dev/null; do
        c=$(adb shell dumpsys battery 2>/dev/null | tr -d '\r' | awk '/^  temperature:/{print $2}')
        s=$(adb shell dumpsys thermalservice 2>/dev/null | tr -d '\r' | awk '/Thermal Status:/{print $3}')
        echo "$t,$c,$s" >> "$out"
        t=$((t + 5))
        sleep 5
    done
}

sustained_run() {
    label=$1
    cpus=$2
    echo
    echo "== sustained ${SUSTAIN_S}s on cpus $cpus =="
    adb shell "$DEV" --only neon_sdot_x4 --sustained "$SUSTAIN_S" --cpus "$cpus" \
        | tr -d '\r' > "$OUT/sustained_$label.csv" &
    sample_thermal $! "$OUT/thermal_$label.csv"
    wait
    tail -1 "$OUT/sustained_$label.csv"
}

sustained_run single "$BIG"
sustained_run big "$BIG_CLUSTER"
sustained_run all "$ALL_CPUS"

echo
echo "== sustained summary =="
sh analyze.sh "$OUT"/sustained_*.csv

echo "results written to $OUT/"

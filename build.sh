#!/bin/sh
# Builds the benchmark for arm64-v8a with the Android NDK.
#
# Only the sdot files get +dotprod. Building everything with it would let the
# compiler use sdot inside the "baseline" kernels too, and the comparison would
# be meaningless. gemm_quant.c is the quantized reference and gets the same
# -fno-vectorize treatment as gemm_scalar.c, for the same reason.
set -eu

API=${API:-31}
NDK=${ANDROID_NDK_HOME:-}

# CC can point at any aarch64 cross compiler. CI uses aarch64-linux-gnu-gcc so
# it can check correctness under qemu without downloading the NDK; the numbers
# in the README only ever come from the real device.
if [ -n "${CC:-}" ]; then
    SKIP_NDK=1
else
    SKIP_NDK=0
fi

if [ "$SKIP_NDK" = "0" ] && [ -z "$NDK" ]; then
    NDK=$(ls -d "${ANDROID_HOME:?set ANDROID_HOME or ANDROID_NDK_HOME}"/ndk/* 2>/dev/null | sort -V | tail -1)
fi

case "$(uname -s)" in
    Linux*)  HOST=linux-x86_64; EXT= ;;
    Darwin*) HOST=darwin-x86_64; EXT= ;;
    *)       HOST=windows-x86_64; EXT=.cmd ;;
esac

if [ "$SKIP_NDK" = "0" ]; then
    CC="$NDK/toolchains/llvm/prebuilt/$HOST/bin/aarch64-linux-android$API-clang$EXT"
    # -f, not -x: Git Bash on Windows does not mark .cmd files executable.
    [ -f "$CC" ] || { echo "no clang at $CC" >&2; exit 1; }
fi

OUT=build
mkdir -p "$OUT"

COMMON="-O3 -Wall -Wextra -Isrc -pthread"
# Turning the vectorizer off is spelled differently by each compiler, and
# getting it wrong silently gives you a "scalar" baseline full of SIMD.
if "$CC" --version 2>&1 | grep -qi clang; then
    NOVEC="-fno-vectorize -fno-slp-vectorize"
else
    NOVEC="-fno-tree-vectorize -fno-tree-slp-vectorize"
fi
BASE="-march=armv8.2-a"
DOT="-march=armv8.2-a+dotprod"

"$CC" $COMMON $BASE $NOVEC -c src/gemm_scalar.c -o "$OUT/gemm_scalar.o"
"$CC" $COMMON $BASE -DGEMM_NAME=gemm_auto -c src/gemm_scalar.c -o "$OUT/gemm_auto.o"
"$CC" $COMMON $BASE -c src/gemm_neon_smull.c -o "$OUT/gemm_neon_smull.o"
"$CC" $COMMON $DOT  -c src/gemm_neon_sdot.c  -o "$OUT/gemm_neon_sdot.o"
"$CC" $COMMON $BASE $NOVEC -c src/gemm_quant.c -o "$OUT/gemm_quant.o"
"$CC" $COMMON $DOT  -c src/gemm_quant_neon.c -o "$OUT/gemm_quant_neon.o"
"$CC" $COMMON $BASE -c src/quant.c            -o "$OUT/quant.o"
"$CC" $COMMON $BASE -c src/measure.c          -o "$OUT/measure.o"
"$CC" $COMMON $BASE -c src/sustained.c        -o "$OUT/sustained.o"
"$CC" $COMMON $BASE -c src/main.c             -o "$OUT/main.o"

"$CC" -static -pthread -o "$OUT/gemmbench" \
    "$OUT/main.o" "$OUT/measure.o" "$OUT/sustained.o" "$OUT/quant.o" \
    "$OUT/gemm_scalar.o" "$OUT/gemm_auto.o" \
    "$OUT/gemm_neon_smull.o" "$OUT/gemm_neon_sdot.o" \
    "$OUT/gemm_quant.o" "$OUT/gemm_quant_neon.o" -lm

echo "built $OUT/gemmbench"

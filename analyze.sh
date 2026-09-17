#!/bin/sh
# Summarises a sustained run.
#
# Comparing the first few seconds against the last few, which is the obvious
# thing to do, mostly measures the window you picked. Under load a phone spends
# its first minute with the governor hunting, and may or may not then settle
# onto one DVFS operating point.
#
# So: find the frequency pair the run spends most of its second half at. If the
# cores really sit there, report the throughput while they do. If they never
# do, say so and report the drift instead of inventing a steady number.
set -eu

# awk decides whether a field is a number using the locale's decimal separator.
# Under a comma locale "193.4" stops looking numeric, comparisons fall back to
# string order, and the maximum of 193.4 and 99.0 comes out as 99.0. The whole
# summary then reads plausibly and is wrong, so pin the numeric locale.
export LC_ALL=C

# Below this share of samples at the modal operating point, the run is treated
# as never having settled.
SETTLED_SHARE=${SETTLED_SHARE:-50}

for f in "$@"; do
    awk -F, -v file="$f" -v need="$SETTLED_SHARE" '
    function median(arr, len,   i, j, t) {
        for (i = 1; i < len; i++)
            for (j = i + 1; j <= len; j++)
                if (arr[j] < arr[i]) { t = arr[i]; arr[i] = arr[j]; arr[j] = t }
        return arr[int((len + 1) / 2)]
    }
    /^[0-9]+,/ { n++; g[n] = $2; k[n] = $3 "/" $4 }
    END {
        if (n == 0) { printf "%s: no samples\n", file; exit }

        for (i = int(n / 2); i <= n; i++) count[k[i]]++
        best = ""; bestc = 0
        for (key in count) if (count[key] > bestc) { bestc = count[key]; best = key }

        m = 0
        for (i = 1; i <= n; i++) if (k[i] == best) steady[++m] = g[i]
        share = 100 * m / n

        settle = -1; run = 0
        for (i = 1; i <= n; i++) {
            if (k[i] == best) { run++; if (run >= 10 && settle < 0) settle = i - run + 1 }
            else run = 0
        }

        peak = 0
        for (i = 1; i <= n; i++) if (g[i] > peak) peak = g[i]

        third = int(n / 3)
        for (i = 1; i <= third; i++) head[i] = g[i]
        for (i = 1; i <= third; i++) tail[i] = g[n - third + i]
        h = median(head, third); t = median(tail, third)

        printf "%s\n", file
        printf "  samples            %d\n", n
        printf "  modal point        %s kHz (little/big), %.0f%% of samples\n", best, share

        if (share >= need && settle > 0) {
            med = median(steady, m)
            printf "  verdict            settled after %ds\n", settle
            printf "  steady median      %.1f GOP/s\n", med
            printf "  best single second %.1f  (%.2fx the steady median)\n", peak, peak / med
        } else {
            printf "  verdict            never settled; the governor kept hunting\n"
            printf "  median first %-5d %.1f GOP/s\n", third, h
            printf "  median last %-6d %.1f GOP/s\n", third, t
            printf "  drift              %+.1f%%\n", (t / h - 1) * 100
            printf "  best single second %.1f  (%.2fx the later median)\n", peak, peak / t
        }
        printf "\n"
    }' "$f"
done

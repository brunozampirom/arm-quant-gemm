#!/bin/sh
# Draws the sustained runs as one SVG, from the same CSVs analyze.sh reads.
#
# Generated rather than drawn, so the picture cannot drift from the data. CI
# regenerates it and fails if the committed file differs.
#
# usage: sh plot.sh results/sustained_single.csv results/sustained_big.csv \
#                  results/sustained_all.csv > docs/sustained.svg
set -eu
export LC_ALL=C

awk '
BEGIN {
    W = 900; H = 430
    L = 64; R = 28; T = 52; B = 52           # margins; the legend sits inside the plot
    PW = W - L - R; PH = H - T - B
    ymax = 200                                # GOP/s, fixed so reruns stay comparable
    xmax = 300                                # seconds
    ns = 0
}
FNR == 1 { next }                             # the "M=64 N=64 ..." banner
FNR == 2 { ns++; name[ns] = FILENAME; sub(/.*sustained_/, "", name[ns]); sub(/\.csv$/, "", name[ns]); next }
/^[0-9]+,/ {
    split($0, f, ",")
    n[ns]++
    x[ns, n[ns]] = f[1] + 0
    y[ns, n[ns]] = f[2] + 0
    if (y[ns, n[ns]] > peak[ns]) peak[ns] = y[ns, n[ns]]
}
function px(v) { return L + v / xmax * PW }
function py(v) { return T + PH - (v / ymax * PH) }
END {
    colour[1] = "#2f7d32"; colour[2] = "#b26a00"; colour[3] = "#9c2f2f"
    label[1] = "1 core";   label[2] = "2 cores";  label[3] = "8 cores"

    printf "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 %d %d\" width=\"%d\" height=\"%d\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" font-size=\"12\">\n", W, H, W, H
    print "<rect width=\"100%\" height=\"100%\" fill=\"none\"/>"
    printf "<text x=\"%d\" y=\"20\" font-size=\"13\" fill=\"#666\">int8 GEMM throughput under sustained load, 300s, unplugged</text>\n", L

    # grid and y labels
    for (g = 0; g <= ymax; g += 50) {
        printf "<line x1=\"%d\" y1=\"%.1f\" x2=\"%d\" y2=\"%.1f\" stroke=\"#ddd\" stroke-width=\"1\"/>\n", L, py(g), L + PW, py(g)
        printf "<text x=\"%d\" y=\"%.1f\" text-anchor=\"end\" fill=\"#888\">%d</text>\n", L - 8, py(g) + 4, g
    }
    printf "<text x=\"14\" y=\"%.1f\" fill=\"#666\" transform=\"rotate(-90 14 %.1f)\" text-anchor=\"middle\">GOP/s</text>\n", T + PH/2, T + PH/2

    # x labels
    for (g = 0; g <= xmax; g += 60) {
        printf "<text x=\"%.1f\" y=\"%d\" text-anchor=\"middle\" fill=\"#888\">%d</text>\n", px(g), T + PH + 20, g
    }
    printf "<text x=\"%.1f\" y=\"%d\" text-anchor=\"middle\" fill=\"#666\">seconds</text>\n", L + PW/2, H - 12

    for (s = 1; s <= ns; s++) {
        printf "<polyline fill=\"none\" stroke=\"%s\" stroke-width=\"1.4\" stroke-linejoin=\"round\" points=\"", colour[s]
        for (i = 1; i <= n[s]; i++) printf "%.1f,%.1f ", px(x[s, i]), py(y[s, i])
        print "\"/>"
        # legend, laid out horizontally inside the plot so no right margin has
        # to guess at the width of a text run
        lx = L + 12 + (s - 1) * 118
        ly = T + 16
        printf "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"3\"/>\n", lx, ly, lx + 22, ly, colour[s]
        printf "<text x=\"%d\" y=\"%d\" fill=\"#333\">%s</text>\n", lx + 30, ly + 4, label[s]
    }
    print "</svg>"
}' "$@"

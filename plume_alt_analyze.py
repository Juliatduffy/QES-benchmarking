#!/usr/bin/env python3
"""Accuracy analysis for plume_alt_compare.sh (numpy only).

Reads the raw concentration files <work>/<size>_<version>_t<trial>_c.f32 written by qesPlumeGPU -o
(20 x 49 x 69 collection boxes, x fastest) and compares every version with the original:

  c rel RMSE   RMS difference of the time-averaged concentration, divided by the peak; variant trial t vs
               original trial t. The yardstick is the same number for two originals (run-to-run noise).
  plume shape  for each downwind slab x = 37.5 ... 97.5 m: mass, centre (y, z) and width (sigma_y, sigma_z)
               of the concentration; reported as the mean over slabs of the deviation from the mean of the
               original's trials. The original's own per-trial deviations show the noise level; a variant whose
               mean deviation is several times that noise has a systematic (bias) effect, not just noise.

usage: plume_alt_analyze.py <work dir> <size> <version> [<version> ...]   (first version = the reference)
"""
import os
import sys
import numpy as np

NX, NY, NZ = 20, 49, 69
XC = 0.0 + (np.arange(NX) + 0.5) * 5.0
YC = 1.0 + (np.arange(NY) + 0.5) * 2.0
ZC = 1.0 + (np.arange(NZ) + 0.5) * 2.0
SLABS = range(7, NX)


def load(work, size, name):
    runs = []
    t = 1
    while True:
        f = os.path.join(work, "%s_%s_t%d_c.f32" % (size, name, t))
        if not os.path.exists(f):
            break
        a = np.fromfile(f, dtype=np.float32).astype(np.float64)
        if a.size == NX * NY * NZ and np.all(np.isfinite(a)):
            runs.append(a.reshape(NZ, NY, NX))
        t += 1
    return runs


def shape_stats(c):
    """per slab: mass, ybar, zbar, sigma_y, sigma_z"""
    out = []
    for i in SLABS:
        s = c[:, :, i]  # (z, y)
        m = s.sum()
        if m <= 0:
            out.append([0.0, np.nan, np.nan, np.nan, np.nan])
            continue
        py = s.sum(axis=0) / m
        pz = s.sum(axis=1) / m
        yb = (py * YC).sum()
        zb = (pz * ZC).sum()
        sy = np.sqrt((py * (YC - yb) ** 2).sum())
        sz = np.sqrt((pz * (ZC - zb) ** 2).sum())
        out.append([m, yb, zb, sy, sz])
    return np.array(out)


def dev(st, ref):
    """mean over slabs: mass ratio-1, d ybar (m), d zbar (m), sigma_y ratio-1, sigma_z ratio-1"""
    return np.array([
        np.nanmean(st[:, 0] / ref[:, 0] - 1.0),
        np.nanmean(st[:, 1] - ref[:, 1]),
        np.nanmean(st[:, 2] - ref[:, 2]),
        np.nanmean(st[:, 3] / ref[:, 3] - 1.0),
        np.nanmean(st[:, 4] / ref[:, 4] - 1.0),
    ])


def rel_rmse(a, b):
    p = max(np.max(a), 1e-30)
    return np.sqrt(np.mean((a - b) ** 2)) / p


def main():
    work, size, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    ref_name = names[0]
    orig = load(work, size, ref_name)
    if len(orig) < 2:
        print("  (accuracy: need >= 2 runs of %s with output, found %d)" % (ref_name, len(orig)))
        return
    ref_stats = np.mean([shape_stats(c) for c in orig], axis=0)
    noise = [rel_rmse(orig[a], orig[b]) for a in range(len(orig)) for b in range(a + 1, len(orig))]
    print("  accuracy of the time-averaged concentration (%d trials per version; reference = mean of the %d %s runs):"
          % (len(orig), len(orig), ref_name))
    print("    c rel RMSE: variant trial t vs %s trial t; for %s: every pair of its own trials (= run-to-run noise)" % (ref_name, ref_name))
    print("    plume shape: mean over the 13 downwind slabs (x = 37.5..97.5 m) of the deviation from the reference;")
    print("    per-trial values in [ ]; for %s they are the noise (they average to 0 by construction)" % ref_name)
    print("    %-10s %10s   %-24s %-24s %-24s %-24s" % ("version", "c relRMSE", "mass (%)", "z centre (m)", "sigma_y (%)", "sigma_z (%)"))
    devs = {}
    for name in names:
        runs = orig if name == ref_name else load(work, size, name)
        if not runs:
            print("    %-10s (no output files)" % name)
            continue
        if name == ref_name:
            e = float(np.mean(noise))
        else:
            e = float(np.mean([rel_rmse(orig[t % len(orig)], runs[t]) for t in range(len(runs))]))
        d = np.array([dev(shape_stats(c), ref_stats) for c in runs])  # trials x 5

        def fmt(col, scale, nd):
            vals = d[:, col] * scale
            return "%+.*f [%s]" % (nd, vals.mean(), " ".join("%+.*f" % (nd, v) for v in vals))

        print("    %-10s %10.2e   %-24s %-24s %-24s %-24s" % (
            name, e, fmt(0, 100, 3), fmt(2, 1, 3), fmt(3, 100, 3), fmt(4, 100, 3)))
        devs[name] = d

    # Bias test. Versions whose interpolated values are bit-identical to the original (the -V check) are
    # the same physics, so all their runs together estimate the run-to-run spread far better than the
    # original's own few trials. z = (version mean - pooled mean) / (pooled std / sqrt(trials)).
    exact = [n for n in os.environ.get("EXACT", "original,ff-ctl,f32,nonut,aos32").split(",") if n in devs]
    pool = np.concatenate([devs[n] for n in exact], axis=0)
    if pool.shape[0] >= 4:
        m0, s0 = pool.mean(axis=0), pool.std(axis=0, ddof=1)
        print("  bias test: difference from the pooled mean of the %d runs of the bit-identical versions (%s),"
              % (pool.shape[0], ", ".join(exact)))
        print("  in units of the standard error of a %d-trial mean (|z| > 4: a systematic effect, not noise)" % len(orig))
        print("    %-10s %-22s %-22s %-22s %-22s" % ("version", "mass (%)", "z centre (m)", "sigma_y (%)", "sigma_z (%)"))
        for name in names:
            if name not in devs or name in exact:
                continue
            d = devs[name]
            cells = []
            for col, scale in ((0, 100), (2, 1), (3, 100), (4, 100)):
                diff = (d[:, col].mean() - m0[col]) * scale
                z = (d[:, col].mean() - m0[col]) / (s0[col] / np.sqrt(d.shape[0])) if s0[col] > 0 else float("nan")
                cells.append("%+.3f (z %+5.1f)%s" % (diff, z, " *" if abs(z) > 4 else "  "))
            print("    %-10s %-22s %-22s %-22s %-22s" % (name, cells[0], cells[1], cells[2], cells[3]))


if __name__ == "__main__":
    main()

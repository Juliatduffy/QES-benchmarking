#!/usr/bin/env python3
"""Summarise the Nsight Compute CSVs written by plume_alt_compare.sh (python standard library only).

usage: plume_alt_ncu.py <csv> [<csv> ...]      file names: ncu_<version>.csv
Each CSV holds the 3 interpolation kernels of one late time step (new particles, winds, turbulence),
from `ncu --page raw --csv --print-units base`.
"""
import csv
import os
import sys

COLS = [
    ("gpu__time_duration.sum", "time ms", 1e-6),
    ("dram__bytes_read.sum", "DRAM rd MB", 1e-6),
    ("lts__t_sectors_srcunit_tex_op_read.sum", "L2 sect rd M", 1e-6),
    ("lts__t_sector_op_read_hit_rate.pct", "L2 hit %", 1.0),
    ("l1tex__t_sector_hit_rate.pct", "L1 hit %", 1.0),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM %pk", 1.0),
    ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "L2 %pk", 1.0),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM %pk", 1.0),
    ("smsp__inst_executed.sum", "inst M", 1e-6),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "occup %", 1.0),
    ("launch__registers_per_thread", "regs", 1.0),
]


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return float("nan")


def to_base(unit):
    """factor from the printed unit to the base unit (ns, byte, sector, inst); 1 if unknown or already base"""
    u = (unit or "").strip()
    times = {"ns": 1, "nsecond": 1, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9}
    if u in times:
        return times[u]
    if len(u) > 1 and u[0] in "KMG" and u[1].islower():
        return {"K": 1e3, "M": 1e6, "G": 1e9}[u[0]]
    return 1.0


def main():
    print("    %-10s %-6s" % ("version", "kernel") + "".join("%13s" % c[1] for c in COLS))
    for path in sys.argv[1:]:
        name = os.path.basename(path)[len("ncu_"):-len(".csv")]
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f) if r]
        hdr_i = next((i for i, r in enumerate(rows) if "ID" in r and any(c[0] in r for c in COLS)), None)
        if hdr_i is None:
            print("    %-10s (no ncu data; see %s.err)" % (name, path[:-4]))
            continue
        hdr = rows[hdr_i]
        units = rows[hdr_i + 1] if hdr_i + 1 < len(rows) and not rows[hdr_i + 1][hdr.index("ID")].strip().isdigit() else [""] * len(hdr)
        data = [r for r in rows[hdr_i + 1:] if len(r) == len(hdr) and r[hdr.index("ID")].strip().isdigit()]
        kcol = hdr.index("Kernel Name") if "Kernel Name" in hdr else hdr.index("Function Name")
        labels = ["new", "winds", "turb"]
        for n, r in enumerate(data):
            kn = r[kcol]
            lab = "turb" if "turb" in kn else "winds" if "winds" in kn else "new" if "new" in kn else labels[n % 3]
            vals = []
            for m, _, scale in COLS:
                j = hdr.index(m) if m in hdr else -1
                vals.append(num(r[j]) * to_base(units[j] if j < len(units) else "") * scale if j >= 0 else float("nan"))
            print("    %-10s %-6s" % (name, lab) + "".join("%13.3f" % v for v in vals))


if __name__ == "__main__":
    main()

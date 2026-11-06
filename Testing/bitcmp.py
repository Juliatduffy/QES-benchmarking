#!/usr/bin/env python3
# Bit-for-bit compare of two netCDF files (same idea as `nccmp -d -S -f -q`, but
# seconds instead of tens of minutes on SLC's 3.4 GB files).
# Usage:
#   python3 bitcmp.py a.nc b.nc            # every variable in a.nc
#   python3 bitcmp.py a.nc b.nc u,v,w      # only u, v and w
# Exit code 0 = every value has exactly the same bits, 1 = something differs.

# Written by Claude code

import sys
import numpy as np
import netCDF4

file_a = netCDF4.Dataset(sys.argv[1])
file_b = netCDF4.Dataset(sys.argv[2])
if len(sys.argv) > 3:
    names = sys.argv[3].split(",")
else:
    names = list(file_a.variables)

all_same = True
for name in names:
    if name not in file_b.variables:
        print(f"{name}: missing from {sys.argv[2]}")
        all_same = False
        continue
    var_a = file_a.variables[name]
    var_b = file_b.variables[name]
    var_a.set_auto_mask(False)          # compare the raw stored values
    var_b.set_auto_mask(False)
    a = np.ascontiguousarray(var_a[:])
    b = np.ascontiguousarray(var_b[:])
    if a.dtype == object:
        # variable-length strings (ex: timestamps) can't be viewed as bytes: compare the text
        same = a.shape == b.shape and all(x == y for x, y in zip(a.ravel(), b.ravel()))
    else:
        # compare the bytes, not the values: +0 vs -0 or two NaNs count as they are stored
        same = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))
    if not same:
        print(f"{name}: DIFFERENT")
        if a.shape == b.shape and a.dtype.kind == "f":
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print(f"   max abs diff {np.nanmax(diff):.3g}, values differing {(a != b).sum()} of {a.size}")
        all_same = False

sys.exit(0 if all_same else 1)

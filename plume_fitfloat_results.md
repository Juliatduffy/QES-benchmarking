# FitFloat in QES-Plume: results (2026-10-27, RTX 4070 SUPER, WSL2)

Branch `plume-fitfloat`. The GPU particle model (`QES-FitFloat/src/plume/CUDA`, driven by
`qesPlumeGPU`) interpolates 14 wind/turbulence fields at random particle positions; the fields
are never written during a run. `plume_compare_fitfloat.sh` stores them as FitFloat bit-packed
arrays (`-c <variant>`) and compares with the original float arrays. Uniform-flow case of
`tests/unitTests/test_CUDA_plume.cpp`, 1500 steps of 1 s, 10 000 particles released per step,
3 alternating timed trials. Logs: `Testing/logs/2026-10-27_19-50-28_*` (small) and
`2026-10-27_20-08-40_*` (large).

## Large domain: 402 x 202 x 141 cells, 648 MB of fields (far beyond the 48 MB L2), 1.85 M active particles

| variant | bits (s+e+m) | packed MB | loop avg (s) | speedup | interpolation speedup | conc. rel RMSE vs original | particle count ratio |
|---|---|---|---|---|---|---|---|
| original | float | 648 | 19.67 | 1.00x | 1.00x | (noise: 7.6e-6) | (noise: 0.99997) |
| ff-ctl | 1+8+23 | 648 | 21.03 | 0.94x | 0.93x | 7.8e-6 | 0.99996 |
| ff-wt24 | 1+8+15 | 486 | 19.01 | 1.03x | 1.03x | 7.7e-6 | 0.99995 |
| ff-wt20 | 1+8+11 | 405 | 17.15 | 1.15x | 1.18x | 7.3e-6 | 0.99994 |
| ff-wt16 | 1+8+7 | 324 | 15.06 | **1.31x** | **1.39x** | 9.8e-6 | 0.99983 |
| ff-wt16e5 | 1+5+10 | 324 | 15.02 | **1.31x** | **1.40x** | 7.9e-6 | 1.00023 |
| ff-wt12 | 1+8+3 | 243 | 12.48 | 1.58x | 1.79x | 4.8e-5 (6x noise) | 0.99602 |
| ff-w16 | winds 16, turb 32 | 578 | 20.69 | 0.95x | 0.95x | 7.3e-6 | 0.99997 |
| ff-t16 | winds 32, turb 16 | 394 | 15.27 | 1.29x | 1.37x | 9.4e-6 | 0.99984 |

Packing the fields on the CPU (OpenMP, 8 threads, incl. the GPU decode check) takes 0.41-0.48 s once;
the decode check reported 0 mismatches in every run. r2 against the Gaussian plume model is 0.9955
for everything except ff-wt12 (0.9969).

## Small domain: 102 x 102 x 141 cells, 83 MB of fields, 0.42 M active particles

No speedup: every variant 0.90-0.98x (interpolation 0.81-0.93x), ff-ctl 0.96x. Half of the fields
sit in the L2 cache, so the decode cost shows and the byte savings do not. Errors as on the large
domain: noise 6.9e-6, all variants down to 16 bits within 7-8.5e-6, ff-wt12 4.5e-5.

## Reading

- **Speedup appears exactly where predicted**: read-only fields, random access, larger than the
  cache. Interpolation is 80% of the loop, and 16-bit fields make it 1.4x faster (loop 1.3x).
  The 32-bit control costs 6% (decoding), 24 bits is a wash, the gain grows with every bit saved.
- **The accuracy cost is below the model's own run-to-run noise down to 16 bits.** The GPU model
  is not deterministic (particle compaction uses atomics, so particles draw different random
  numbers), and two identical original runs differ by 7e-6 of the peak concentration; ff-wt16
  and ff-wt16e5 sit at that level. 12 bits (3 mantissa bits) is measurable: 6x the noise and
  0.4% of the particles lost.
- **It is the turbulence fields that matter** (11 of the 14; ff-t16 gives the full 1.3x). In the
  uniform case the winds are constant, so ff-w16 shows only their decode cost; a real wind field
  would make ff-w16 informative.
- Fewer exponent bits are free here: 1+5+10 (fp16-like range) packs every value without
  overflow or flush and has the smaller error of the two 16-bit layouts.
- Not tested: a real QES case (buildings, terrain, inhomogeneous winds), where the wind fields
  vary and the access pattern is less regular; the production qesPlume runs on the CPU, so this
  needs the GPU particle model wired into it first.

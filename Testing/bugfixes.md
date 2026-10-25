# Bugs found in QES while setting up GPU correctness checks

Three bugs are fixed (1, 2, 3), one is documented but deliberately left alone (4). All changes
live in `QES-GPU/` only. `QES-Public/` (the timing baseline) and `QES-CPU/` are untouched.
The fixed original solver is selectable with `-s 3 -c bugfix`; `gpu_compare.sh` compares every
GPU optimization against it, because the unfixed original does not reproduce its own output.

How we got here: the `fuse` GPU solver produced the same residuals as the original but
`nccmp` reported the wind fields as DIFFERENT. Running the *unmodified* original twice on
the OKC case also gave DIFFERENT, so the original was not reproducible and could not be used
as a correctness reference. Each bug below was found while tracking that down.

## Summary

| # | File (QES-GPU) | Bug | Effect on output | How we know |
|---|---|---|---|---|
| 1 | `src/winds/CutBuilding.cpp:145` | reads `skip_id[number]` one past the end of the vector | building geometry (`building_volume_frac`, `icellflag`, coefficients, `u0`) changed from run to run in GPU builds | AddressSanitizer stack trace; 5/5 run pairs differed before, 4/4 identical after |
| 2 | `src/winds/Solver_GPU_GlobalMemory.cu`, `applyNeumannBCGlobal` | thread range `nx*ny` instead of `(nx-1)*(ny-1)` | 768 threads overwrite `k=1` cells that other blocks are reading in the same launch; also wiped SOR updates every iteration | arithmetic of the index range; fused error measurement showed a 0.59 residual that dropped to 1e-5 after the fix |
| 3 | `src/fire/CUDA/PotentialGlobal.cu`, `PotGlob` | `gMaxIdx == pot_G`, one past the end of the `G`/`G'` lookup tables, whenever `x1 > 1 - 0.5/pot_G` (cells nearly straight above the fire) | fire-induced winds depended on whatever GPU memory sat next to the table, i.e. on the allocation pattern | instrumented kernel counts 36-72 out-of-range threads per fire step; clamping the index made `fuse` and `bugfix` bit-identical on coupled fire |
| 4 | `src/winds/Solver_GPU_GlobalMemory.cu`, `finalVelocityGlobal` | building thread zeroes a shared face while the neighbor thread writes its velocity to it | a data race by the CUDA memory model; **no effect on output was observed**, so it is **not fixed** (see below) | code reading only; 8 in-situ runs and 10 standalone runs never showed it; removing the fix again left every output bit-identical |

Evidence was gathered with AddressSanitizer on an unmodified CPU-only build, repeated runs
compared with `nccmp`, a standalone CUDA program containing the original boundary kernel, and
an instrumented copy of `PotGlob`; each section below says what was run and what it showed.

Two more things were found but **not** fixed, since they are outside the GPU work:

- The original CPU red-black solver (`cpu-original -s 1`) gives different `u,v,w` on OKC from run to run with identical inputs. Likely the race the QES-CPU `bugfix` variant addresses.
- QES measures time with `std::chrono::high_resolution_clock`, which is the wall clock on GCC. On WSL2 the clock is occasionally re-synced mid-run, which produces negative elapsed times. `helpers.sh` detects a negative time and repeats the trial. The real fix would be `std::chrono::steady_clock` in the QES timers.

## 1. `CutBuilding.cpp`: out-of-bounds read of `skip_id`

```cpp
unsigned int number = 0;
for (auto id = 0u; id < PolyB->polygonVertices.size() - 1; id++) {
  if (skip_id.size() != 0) {
    if (id == skip_id[number]) {          // number can equal skip_id.size()
      if (number < skip_id.size()) {
        number += 1;
      }
      continue;
    }
  }
```

`skip_id` lists polygon edges that were already handled. After the last one is consumed,
`number == skip_id.size()` and `skip_id[number]` reads the 4 bytes after the vector. If that
garbage happens to equal a later edge index, that edge is skipped and its cells are never cut,
so `building_volume_frac` stays 1 instead of the correct fraction (observed: `1 <> 0.374`).
From there `icellflag`, the solver coefficients and `u0` change, and so does everything
downstream.

**Why it only showed in GPU builds:** the CUDA runtime starts helper threads that allocate on
the same heap, so the bytes after that tiny vector could be live, changing memory. In the
CPU-only build they were always the same stale bytes, so the CPU build looked deterministic.

**Evidence**
- CPU-only build: `building_volume_frac`, `icellflag`, `u0` identical across 5 runs.
- GPU build, `-s 3`: all of them differed in 5 of 5 run pairs; `-s 1` in the same build also
  showed `building_volume_frac` differences.
- `MALLOC_PERTURB_` (fills freed heap memory with a pattern) did not make it reproducible,
  which pointed to live memory rather than stale heap contents.
- AddressSanitizer: `heap-buffer-overflow ... READ of size 4 ... CutBuilding::setCutCellFlags
  CutBuilding.cpp:145 ... 0 bytes to the right of 4-byte region` allocated at `CutBuilding.cpp:284`.
  It was the only out-of-bounds access in the whole OKC setup.
- After the fix: 4 of 4 runs identical on OKC at 14 threads.

**Fix:** check `number < skip_id.size()` *before* indexing.

## 2. `applyNeumannBCGlobal`: boundary kernel runs into the next layer

```cpp
if (id < nx * ny) {                                      // should be (nx-1)*(ny-1)
  d_lambda[id] = d_lambda[id + 1 * (nx - 1) * (ny - 1)]; // copy layer k=1 into layer k=0
}
```

One layer of cells has `(nx-1)*(ny-1)` entries (160,000 on the fire grid), but the check
allows `nx*ny` (160,801), so 768 extra threads run. They write cells `160000..160767`, which
are the first cells of layer `k=1`, while threads in block 0 read exactly those cells to fill
layer `k=0`. Blocks run in no fixed order, so each of those cells ends up holding either the
`k=1` value or the `k=2` value. The extra threads also overwrite 768 `k=1` cells on every
iteration, undoing the SOR update there.

**Evidence**
- The index arithmetic above.
- The fused error measurement (which records the change made by the SOR kernel) reported a
  residual of 0.59 after 500 iterations, i.e. cells whose update was being erased each
  iteration. With the range fixed the residual is ~1e-5, like the original's.

**Fix:** `if (id < (nx - 1) * (ny - 1))`.

## 3. `PotGlob`: lookup-table index one past the end

```cpp
float x1 = (1 + (1 / sqrtf((h_k / z_k) * (h_k / z_k) + 1))) / 2.0;
int gMinIdx = floor(pot_G * (x1 - .5) / .5);
int gMaxIdx = ceil(pot_G * (x1 - .5) / .5);
float g_x      = 0.5 * (d_G[gMinIdx] + d_G[gMaxIdx]);
float gprime_x = 0.5 * (d_Gprime[gMinIdx] + d_Gprime[gMaxIdx]);
```

`gMaxIdx` reaches `pot_G`, one element past the end of `d_G` and `d_Gprime` (size `pot_G`),
whenever `x1 > 1 - 0.5/pot_G`; with `pot_G = 400` that is any `x1 > 0.99875`, i.e. every target
cell nearly straight above the fire (`h_k` small compared with `z_k`). The read returns whatever
lies next to the table in GPU memory. An instrumented build that counts the out-of-range indices
finds 36 to 72 such threads in every fire step of `fire_short.xml`, with `x1` values of
0.99917 and 0.99959. Which allocation that is depends on the
sequence of `cudaMalloc`/`cudaFree` calls made before it, and the `fuse` solver frees `d_error`
every solve while the original leaks it, so the two solvers saw different garbage.

**Evidence (by elimination)**
- With fire-induced winds *on*, `fuse` and `bugfix` differed from the second wind solve
  onward; with one fire step only (one coupled solve) they were bit-identical, and the `-fmad=false`
  build still differed, which ruled out the SOR arithmetic.
- Allocating `d_lambda_old` in `fuse` again (unused, then also zero-filled) changed nothing.
- Removing the `cudaFree(d_error)` from `fuse`, so its allocation pattern matched the original,
  made the two bit-identical. A 4-byte free changing physics means a read-before-write somewhere.
- `PotGlob` is the only fire kernel launched during a step and all its buffers are copied from
  host, so the read had to be out of bounds; the `G` lookup is the only index that can reach
  `pot_G`, and the instrumented kernel confirms it does, every step.
- Clamping `gMinIdx`/`gMaxIdx` to `pot_G - 1`, with the `cudaFree` kept, made `fuse` and `bugfix`
  bit-identical on coupled `short` and `tiny`.

Note that the `atomicAdd`s in `PotGlob` are **not** a problem: the kernel is launched once per
burning cell with one thread per target cell, so no two threads add to the same address in one
launch, and launches are ordered. An earlier guess blamed them; it was wrong.

**Fix:** clamp both indices to `pot_G - 1`. The `d_u_r`/`d_u_z` lookups in the same kernel use
the same pattern and were not checked.

## 4. `finalVelocityGlobal`: zeroing building faces races with the velocity update

Each thread owns one cell and does two things in the same kernel: update the velocity on its
own faces, and, if its cell is a building/terrain cell, set all six surrounding faces to zero.
The face between a building cell and its air neighbor is written by both threads:

```
   building cell (thread A)  |  air cell (thread B)
                  A writes 0 -->|<-- B writes u + correction
```

Whichever write lands last wins, so the wall face is sometimes 0 and sometimes the full
velocity.

**Evidence, and its limits**
- Code reading: two writers to one address in one launch with no ordering between them. By the
  CUDA memory model that is a data race, and the fix is correct by construction.
- It was **not** observed to change output. A standalone CUDA test of the same write pattern on
  43,000 shared faces, run 10 times, always had the zero write land last. A QES build with bugs 1,
  2 and 3 fixed and bug 4 left in gave 8 identical OKC runs.
  The reason is that adjacent threads sit in the same warp and execute in lockstep, so the
  later statement (the zero) reliably runs after the earlier one (the velocity write). The race
  can only be lost at warp or block boundaries, and on this GPU it never was.
- The 6.7 m/s face differences seen between two runs of the unmodified original are explained by
  bug 1 (a cell flagged as building in one run and as air in the other), not by this race.

So: real race, but no receipt that it ever affected a result.

**Not fixed.** A fix (splitting the zeroing into a second kernel launched after the velocity
update) was tried and then removed again to keep the changes to QES minimal. With it removed,
`bugfix` still reproduces itself on OKC (3 of 3 runs) and its output is bit-identical to the
output it produced with the fix in place; `fuse` still matches `bugfix` on OKC and on fire.
If the race ever does show up (a `DIFFERENT` between two `bugfix` runs with no other change),
that split is the fix to apply.

## Current state of reproducibility

| Build / mode | Reproducible run to run |
|---|---|
| `gpu-optimized -c bugfix`, `fuse`, `compressed`: OKC wind | yes (4 of 4 at 14 threads; 3 of 3 after removing fix 3) |
| same, coupled fire `short` and `long` | yes |
| `gpu-original` (QES-Public, all four bugs present) | no |
| `cpu-original -s 1` on OKC | no (CPU solver race, not addressed) |

`gpu_compare.sh` therefore times against `gpu-original` (the real baseline) and checks
correctness against `-c bugfix`. `IDENTICAL` in a log means bit-for-bit.

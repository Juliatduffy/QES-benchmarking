# How the timing scripts work

## Quick start (always run from `~/QES`)
- `sbatch build.sh`: compile both QES-Public (the unchanged original code) and QES-CPU (my version with the optimizations). Run it after any code change.
- `sbatch run.sh`: run the unchanged QES-Public code once at 1, 16 and 64 threads and save its output. This is the reference answer every optimization is checked against.
- `sbatch compare.sh bugfix`: time QES-Public against my `bugfix` version at 1, 16 and 64 threads, then check that `bugfix` gives exactly the same wind field.

## Input
- **Test case:** Oklahoma City (OKC), `QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml`.
  - It's the README's official sanity test: 122 × 125 × 78 cells with real downtown buildings.
  - The XML describes the grid, the wind sensor, and the building shapefile.
- **The two programs:**
  - `build-public/qesWinds/qesWinds`: QES-Public, the original, never modified (the baseline)
  - `build-cpu/qesWinds/qesWinds`: QES-CPU, my copy, containing all my optimized solver versions

## `build.sh`: compile both programs
- Loads the compiler and libraries QES needs (the gcc 11.2, netcdf, gdal and boost modules from `QES-Public/CHPC/loadmodules_QES.sh`).
- Runs `cmake` and then `make` twice: once for `QES-Public` into `build-public/`, once for `QES-CPU` into `build-cpu/`.
- Both use the same flags, so the comparison is fair:
  - `ENABLE_OPENMP=ON`: threads, and the red/black solver
  - `HAS_CUDA_SUPPORT=OFF`: no GPU code
  - `NETCDF_CXX_DIR=…`: where the NetCDF C++ library is
- Only the `qesWinds` program is built, which takes about 1 minute.
- The full build output goes to `Testing/logs/build.out`, overwritten each time. Look there if a build fails.

## `run.sh`: get the reference output from QES-Public once
- Runs the unchanged QES-Public program on OKC three times, at 1, 16 and 64 threads.
- Each run saves the wind field it computed as `Testing/runs/public/okc_<N>threads/okc_windsWk.nc`. That file holds the raw `u`, `v`, `w` velocities straight from the solver.
- These three files are the "correct answers" (the golden files) that `compare.sh` checks every optimization against.
- QES-Public never changes, so you only need to run this once. Rerun it only if QES-Public gets rebuilt differently, for example with another compiler.
- Its output goes to `Testing/logs/run.out`.

## `compare.sh <optimization>`: timing
The name you give (e.g. `bugfix`) chooses which of my solver versions runs. For each of 1, 16 and 64 threads:
1. **Warm-up (not timed):**
   - Each program runs once, so files are cached and the node is warmed up before timing starts.
   - My version's warm-up also saves its wind field to `Testing/runs/cpu/<optimization>/okc_<N>threads/`, for the correctness check.
2. **Timed trials:**
   - Ten trials. Each trial runs QES-Public once, then my version once.
   - The two alternate, so both see the same node conditions. Otherwise, a slow start to the job would make whichever version runs first look slower.
3. **The time:**
   - QES measures its own solver time, which leaves out reading the input and writing files.
   - It prints the time on the line right after `Residual after 500 itertations: …`.
   - `collect_results` runs the program once and uses `awk` to pull out two numbers: that time and the residual.
4. **Report:**
   - Min, Max and Avg time for each version, with the math done by `bc`.
   - Speedup = QES-Public's average ÷ my version's average. Above 1.00 means mine is faster.
   - Whether the two residuals match: a quick sanity check that both did the same math.

## `compare.sh`: correctness
- After all the timing, `nccmp` compares my version's saved wind field with QES-Public's golden file for the same thread count.
- It checks every `u`, `v` and `w` value (about 1.2 million each), bit for bit.
- `IDENTICAL` means my version computed exactly the same answer as the original.
- `DIFFERENT` prints a table: how many values differ (Count), and the smallest and largest difference (Min, Max).
- The residual alone isn't enough: two different answers can print the same residual. `nccmp` is the real test.

## `compare.sh`: output
- The summary (times, speedups, correctness) goes to `Testing/logs/<date>_<time>_<optimization>.log`. Each run gets a new file, so you keep a history.
- `Testing/logs/compare.out` only holds the module list and the name of the summary file, overwritten each run.

## Switching between optimizations
- QES-CPU has an extra command-line flag: `-c <name>` (long form `--windOptimization <name>`). QES-Public doesn't have it.
- `QES-CPU/src/winds/SolverFactory.cpp` reads the name and creates the matching solver class:
  - no `-c`: the original solver, the same code as QES-Public
  - `-c bugfix`: `Solver_CPU_RB_bugfix`, which fixes the `max_error` race condition
  - unknown name: QES stops with `Unknown windOptimization '<name>'` and lists the valid names
- `compare.sh` passes its argument straight to `-c`, so `sbatch compare.sh bugfix` times QES-CPU running `-c bugfix`.

## Adding a new optimization
1. Copy the previous version's two files, e.g. `QES-CPU/src/winds/Solver_CPU_RB_bugfix.h` and `.cpp`, to new names. Rename the class inside, and change the `Running Red/Black CPU Solver (...)` message so logs show which version ran.
2. Make one change in the copy: the optimization itself.
3. Register it:
   - add the new `.cpp` to `QES-CPU/src/winds/CMakeLists.txt`
   - `#include` the new `.h` in `SolverFactory.h`
   - add an `else if (windOptimization == "<name>")` in `SolverFactory.cpp`
4. Rebuild with `sbatch build.sh`, then test with `sbatch compare.sh <name>`. Check the summary for the speedup and for `IDENTICAL`.

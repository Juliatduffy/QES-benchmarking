# How the timing scripts work

## Quick start (always run from `~/QES`)
- `sbatch build.sh`: compile both QES-Public (the unchanged original code) and QES-CPU (the copy with the optimizations). Run it after any code change.
- `sbatch run.sh`: run the unchanged QES-Public code on both test cases and save its output. This is the reference answer every optimization is checked against.
- `sbatch wind_compare.sh blocks`: time QES-Public against the `blocks` wind solver in QES-CPU on both test cases, then check that `blocks` gives exactly the same wind field.
- `sbatch fire_compare.sh -f blocks`: time the fire model, QES-Public against QES-CPU's parallel fire code plus the `blocks` wind solver, then check that every fire output is the same.
- **Safest way after a code change:** build, and only start the comparison if the build worked:
  ```bash
  B=$(sbatch --parsable build.sh | cut -d';' -f1)
  sbatch --dependency=afterok:$B wind_compare.sh blocks
  ```
  Otherwise, if the build fails, the comparison quietly times the old program.

## Inputs
- **Wind test cases** (both from QES-Public, never modified):
  - **OKC (Oklahoma City)**, `QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml`.
    - It's the README's official sanity test: 122 × 125 × 78 cells (1.2 M) with real downtown buildings.
    - Small enough to fit in the CPU's cache, so it shows the cost of instructions and of threads waiting for each other.
  - **SLC (Salt Lake City)**, `QES-Public/tests/runTestsFiles/winds_SaltLakeCity.xml`.
    - 700 × 700 × 100 cells (49 M), about 40× bigger.
    - Too big for the cache, so it shows the cost of moving data to and from memory.
  - Some optimizations help one case and hurt the other, so every wind comparison runs both.
- **Fire test case:** shortened copies of `QES-Public/data/InputFiles/fire.xml` (400 × 400 × 20 cells, 3 ignition points), in `/scratch/general/vast/u1417379/qes-fire/inputs/`.
  - **`fire_short.xml`:** 10 s of fire, about 7 fire steps. Used for timing.
  - **`fire_tiny.xml`:** 3 s, 2 steps. Used for a quick correctness check.
- **The programs:**
  - `build-public/qesWinds/qesWinds` and `build-public/qesFire/qesFire`: QES-Public, the original, never modified (the baseline)
  - `build-cpu/qesWinds/qesWinds` and `build-cpu/qesFire/qesFire`: QES-CPU, the working copy, containing all the optimized versions

## `build.sh`: compile both programs
- Loads the compiler and libraries QES needs (the gcc 11.2, netcdf, gdal and boost modules from `QES-Public/CHPC/loadmodules_QES.sh`).
- Runs `cmake` and then `make` twice: once for `QES-Public` into `build-public/`, once for `QES-CPU` into `build-cpu/`.
- Both use the same flags, so the comparison is fair:
  - `ENABLE_OPENMP=ON`: threads, and the red/black solver
  - `HAS_CUDA_SUPPORT=OFF`: no GPU code
  - `NETCDF_CXX_DIR=…`: where the NetCDF C++ library is
- It builds the `qesWinds` and `qesFire` programs, which takes a few minutes.
- The full build output goes to `Testing/logs/build.out`, overwritten each time.
  - Look there if a build fails.
  - A good build shows `Built target qesWinds` and `Built target qesFire` twice each.

## `run.sh`: get the reference output from QES-Public once
- Runs the unchanged QES-Public program:
  - on OKC at 1, 16 and 64 threads
  - on SLC at 16 and 64 threads. One thread would take about 3 minutes per run.
- Each run saves the wind field it computed as `Testing/runs/public/<case>_<N>threads/<case>_windsWk.nc`. That file holds the raw `u`, `v`, `w` velocities straight from the solver.
- These five files are the "correct answers" (the golden files) that `wind_compare.sh` checks every optimization against.
- QES-Public never changes, so you only need to run this once.
  - Rerun it only if QES-Public gets rebuilt differently, for example with another compiler. It overwrites the old files.
  - The SLC files are big (4.6 GB per thread count).
- Uses the long QOS (`coe-students-grn`, 30-minute limit). Its output goes to `Testing/logs/run.out`.

## `wind_compare.sh <optimization>`: timing
The name you give (e.g. `blocks`) chooses which QES-CPU solver version runs. The script does OKC first (1, 16 and 64 threads, 10 trials), then SLC (16 and 64 threads, 3 trials, because each SLC run is slow). For each thread count:
1. **Warm-up (not timed):**
   - Each program runs once, so files are cached and the node is warmed up before timing starts.
   - The optimized version's warm-up also saves its wind field to `Testing/runs/cpu/<optimization>/<case>_<N>threads/`, for the correctness check.
2. **Timed trials:**
   - Each trial runs QES-Public once, then the optimized version once.
   - The two alternate, so both see the same node conditions. Otherwise, a slow start to the job would make whichever version runs first look slower.
3. **The time:**
   - QES measures its own solver time, which leaves out reading the input and writing files.
   - It prints the time on the line right after `Residual after 500 itertations: …`.
   - `collect_results` runs the program once and uses `awk` to pull out two numbers: that time and the residual.
4. **Report:**
   - Min, Max and Avg time for each version, with the math done by `bc`.
   - Speedup = QES-Public's average ÷ the optimized version's average. Above 1.00 means the optimized version is faster.
   - Whether the two residuals match: a quick sanity check that both did the same math.

## `wind_compare.sh`: correctness
- After each case's timing, `nccmp` compares the optimized version's saved wind field with QES-Public's golden file for the same case and thread count.
- It checks every `u`, `v` and `w` value bit for bit: about 1.2 million each for OKC, 49 million for SLC.
- `IDENTICAL` means the optimized version computed exactly the same answer as the original.
- `DIFFERENT` prints a table: how many values differ (Count), and the smallest and largest difference (Min, Max).
- The residual alone isn't enough: two different answers can print the same residual. `nccmp` is the real test.
- The optimized version's SLC output files are deleted right after checking (4.6 GB each); the OKC ones are kept.
- Loading `nccmp` swaps one of QES's libraries, so the script reloads QES's modules before starting the next case.

## `wind_compare.sh`: output and run time
- The summary (times, speedups, correctness, for both cases) goes to `Testing/logs/<date>_<time>_<optimization>.log`. Each run gets a new file, so you keep a history.
- `Testing/logs/wind_compare.out` only holds the module list and the name of the summary file, overwritten each run.
- **Run time:** about 10 minutes, so it uses the long QOS (`coe-students-grn`, 30-minute limit). That queue has lower priority, so the job may wait a bit before starting.
- **Quick OKC-only check** while debugging: use the short queue.
  ```bash
  sbatch --qos=coe-class-shortbatch-grn -t 0:10:00 wind_compare.sh <optimization>
  ```
  The job is stopped at the 10-minute limit partway through SLC, but the OKC part of the log is complete.

## `fire_compare.sh [-f] <optimization> [short|tiny]`: fire timing and correctness
- **Arguments:**
  - **`-f`** (optional): use QES-CPU's parallel fire code (`QES-CPU/src/fire/PotentialParallel.cpp`) instead of the original serial one.
  - **`<optimization>`:** the wind solver version to use, the same names as `wind_compare.sh`.
  - **`short`** (default) or **`tiny`:** which input to use.
- **Examples:**
  - `sbatch fire_compare.sh -f blocks`: parallel fire code + `blocks`, timing run
  - `sbatch fire_compare.sh -f blocks tiny`: the same, quick correctness check
  - `sbatch fire_compare.sh blocks`: original fire code + `blocks`. This shows that a faster wind solver alone barely helps (Amdahl's law), because the fire code takes ~96% of the time.
- **What a fire run does:** each fire time step computes the fire-induced winds (the "potential"), runs a full wind solve, then moves the fire front.
- **Trials:**
  - At 16 and 64 threads, QES-Public's qesFire and QES-CPU's run alternately: 3 trials for `short`, 1 for `tiny`.
  - There's no warm-up run, because each run takes minutes.
  - One QES-Public `fire_short` run takes about 5 minutes.
- **The times** (from QES's own `elapsed time` lines, summed over all fire steps, without setup or file writing):
  - **fire loop:** the whole fire time loop. The speedup is computed from this.
  - **potential:** the time in the fire-induced-winds code
  - **wind solver:** the time in the wind solver
- **Correctness:** `nccmp` compares **every** variable in the two fire output files (winds, burn state, fire front, …). Then both files are deleted (~350 MB per run).
- **Where it runs:** QES's fire code opens `../data/FireFiles/HeatPot.nc` relative to the folder it starts in. So the script starts qesFire from `/scratch/general/vast/u1417379/qes-fire/run/`, where `../data` is a link to QES-Public's data folder. You still submit from `~/QES`.
- **Output:**
  - Summary: `Testing/logs/<date>_<time>_fire_<flags>_<case>.log`.
  - Long QOS, 1.5-hour limit. `Testing/logs/fire_compare.out` holds the module list.

## Switching between optimizations
- **Wind solver:** QES-CPU's `qesWinds` and `qesFire` have an extra flag, `-c <name>` (long form `--windOptimization <name>`). QES-Public doesn't have it.
  - `QES-CPU/src/winds/SolverFactory.cpp` reads the name and creates the matching solver class.
  - **No `-c`:** the original solver, the same code as QES-Public.
  - **`-c bugfix`, `checkerboard`, `fuse`, `collapse`, `oneregion`, `blocks`, `combined`:** the optimized versions. Each one is `Solver_CPU_RB_<name>`.
  - **Unknown name:** QES stops with `Unknown windOptimization '<name>'` and lists the valid names.
- **Fire code:** QES-CPU's `qesFire` also has `-f parallel` (long form `--firePotential parallel`), which calls `Fire::potentialParallel` instead of the original. An unknown name is an error.
- **How the scripts use them:**
  - `wind_compare.sh` passes its argument straight to `-c`.
  - `fire_compare.sh` passes `-f parallel` when you give `-f`, plus `-c <optimization>`.

## Adding a new optimization
1. Copy the previous version's two files, e.g. `QES-CPU/src/winds/Solver_CPU_RB_bugfix.h` and `.cpp`, to new names. Rename the class inside, and change the `Running Red/Black CPU Solver (...)` message so logs show which version ran.
2. Make one change in the copy: the optimization itself.
3. Register it:
   - add the new `.cpp` to `QES-CPU/src/winds/CMakeLists.txt`
   - `#include` the new `.h` in `SolverFactory.h`
   - add an `else if (windOptimization == "<name>")` in `SolverFactory.cpp`, and add the name to the error message's list
4. Build and test with the chained command in Quick start. Check the summary for the speedups and for `IDENTICAL` at every thread count, in both cases.

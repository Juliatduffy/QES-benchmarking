# Plan: `gpu_profile.sh`, a raw GPU profile of QES-Public

**You write it, following the steps below; Claude helps at every step and reviews each one.**
Written 2026-10-18. Revised the same day:
- The log shows raw numbers only, with no verdict.
- The fire inputs are the full `fire.xml` and FFII.
- The log is kept short and plain: about 20 lines per input.

## What the script does
**Run it like this:**
```bash
sbatch gpu_profile.sh chpc      # on CHPC (A100 node; queued)
bash   gpu_profile.sh pc        # on the RTX 4070 Super PC, inside WSL2
```

**What it profiles:** **unmodified QES-Public** on the GPU (`-s 3` = the GPU solver QES actually uses), for four inputs:

| Name | Input file (read from QES-Public, never edited) | What it is |
|---|---|---|
| `okc` | `tests/runTestsFiles/winds_OklahomaCity.xml` | wind only; 1.2 M cells, cut-cell buildings |
| `slc` | `tests/runTestsFiles/winds_SaltLakeCity.xml` | wind only; 49 M cells, stair-step buildings |
| `fire` | `data/InputFiles/fire.xml` | QES's fire demo, full length (100 s of fire, ~50 steps); 400×400×20 cells |
| `ffii` | `data/InputFiles/FFII.xml` | the FireFlux II field experiment that the QES-Fire paper validates against (1200 s of fire, ~200 steps); 80×150×40 cells. ⚠️ Never run on the GPU yet |

**Why the full fire runs, not a 2-step one:**
- **Setup:** with only 2 steps, setup is about half the run, which distorts every share.
- **Fire growth:** the fire grows, and the plume work grows with it. On the CPU, FFII's potential went from 0.09 s to 25 s per step over its first 26 steps (scratch `qes-fire/FFII.log`, 2026-10-13).

---

## What the log looks like
This is the whole log for one input. All numbers are real, from the A100 SLC run (job 16308989).
```
QES-Public GPU profile (-s 3)
machine: chpc   job: 16308989   date: 2026-10-16 03:31
GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 580.xx   CUDA 11.8

=== slc ===
whole run: 7.860 s
stage times: seconds, count, stage
    2.231     1  [Solver] Residual after
    1.727     1  [QES-WINDS] Defining solid
    0.719     1  [QES-WINDS] Applying rooftop
    0.175     1  [QES-WINDS] Applying Wind
    0.009     1  [ESRIShapefile] Loading completed.
    0.008     1  [QES-WINDS] Sorting buildings
GPU kernels (nsys): % of kernel time, total ms, launches, kernel
  73.6    1162.7   1000  SOR_RB_Global
  17.3     273.5    500  calculateErrorGlobal2
   8.5     134.3    500  saveLambdaGlobal
   0.2       3.9      1  finalVelocityGlobal
   0.2       3.6    500  applyNeumannBCGlobal
   0.2       2.7      1  divergenceGlobal
copies (nsys): % of copy time, total ms, count, kind
  78.4     230.5     13  [CUDA memcpy HtoD]
  21.2      62.5    503  [CUDA memcpy DtoH]
   0.4       1.1    501  [CUDA memset]
memory vs compute (ncu): DRAM %, SM %, launches measured, kernel
  73.6    39.4      2  SOR_RB_Global
  70.3    15.2      1  saveLambdaGlobal
```
(That old job measured only these two kernels with ncu. Your run lists every solver kernel except `finalVelocityGlobal`.)

## How to read it
1. **`stage times` vs `whole run`: is this stage worth touching at all?**
   - Divide by hand: here the wind solve is 2.23 / 7.86 = 28% of the run. That caps what any speedup of it can give (Amdahl's law).
   - The labels are the last `[...]` line QES printed before each time, so some need translating:

   | Label in the log | Stage, file | Runs on |
   |---|---|---|
   | `[Solver] Residual after` | wind solve, `Solver_GPU_GlobalMemory.cu` | GPU |
   | `[QES-Fire] GPU POTENTIAL` | plume winds, `PotentialGlobal.cu` | GPU **and CPU** (the block search and the final adding-up are CPU loops in that file) |
   | `[QES-Fire] LS init …` / `Initializing level` | `LSinitGlob.cu` | GPU |
   | `[QES-Fire] Computing level` | spread rate + heat, `LevelSetNB.cpp` | CPU |
   | `[QES-Fire] dt =` | moving the fire front, `Move.cpp` | CPU |
   | `[QES-WINDS] Defining solid`, `Sorting buildings`, `Applying …` | setup, `WINDSGeneralData.cpp` | CPU |
   | `[QES-Fire] Current time` | **one whole fire step** (it includes all the fire rows above) | — |

   - **The `Current time` row is the total of the fire loop,** so don't add it to the others.
2. **`GPU kernels`: which kernel takes the GPU's time?** The first column is its share of all kernel time.
3. **`copies`: how much time goes to moving data between CPU and GPU?** Compare its total ms with the kernels' total ms.
4. **`memory vs compute`: what limits that kernel?**
   - **DRAM high and above SM:** limited by moving bytes, so fewer bytes could help.
   - **Both low:** waiting on something else.
   - **SM high:** limited by math.
5. **Which arrays a kernel touches.** This comes from the code, not from the log:

   | Kernel | File | Big arrays it reads (writes) |
   |---|---|---|
   | `SOR_RB_Global` | `Solver_GPU_GlobalMemory.cu` | e, f, g, h, m, n, R, λ (λ) |
   | `saveLambdaGlobal` | same | λ (λ_old) |
   | `calculateErrorGlobal2` | same | λ, λ_old |
   | `applyNeumannBCGlobal` | same | λ, one plane (λ) |
   | `divergenceGlobal` | same | u0, v0, w0, e..n (R), once per solve |
   | `finalVelocityGlobal` | same | λ, u, v, w, f, h, n, icellflag (u, v, w), once per solve |
   | `PotGlob` | `PotentialGlobal.cu` | small lookup tables (Pot_u, Pot_v, Pot_w); one launch per merged fire |
   | `LSGlob` | `LSinitGlob.cu` | the 2-D fire-front map; once, at setup |

   Whether the *values* fit in fewer bits is a separate question, answered from the data (e.g. SLC's coefficients 0 and 0.25 fit; OKC's 0.01 doesn't).

---

## The rule that keeps the log short
**Only the lines printed with `echo`, `printf` or the three `awk` lines go into the log.** Everything a tool prints by itself goes to a file in `$WORK` or to `/dev/null`:
- the module list
- the build output
- QES's own output
- nsys's and ncu's progress messages

If a section comes out empty, run that command by hand without the `> …` part to see the error.

---

## Step 0: get each machine ready (once)
### The PC (RTX 4070 Super): WSL2
1. **In Windows:** update the NVIDIA driver.
2. **Enable counters:** NVIDIA Control Panel → Desktop menu → *Enable Developer Settings*, then Developer → *Manage GPU Performance Counters* → **Allow access to all users**. Without this, `ncu` prints a permission error.
3. **Install Ubuntu:** in PowerShell, as administrator: `wsl --install -d Ubuntu-22.04`, then reboot and open "Ubuntu".
4. **CUDA toolkit:** developer.nvidia.com/cuda-downloads → Linux → x86_64 → **WSL-Ubuntu** → follow the commands. Then add this line to `~/.bashrc`: `export PATH=/usr/local/cuda/bin:$PATH`.
   - **Never install a Linux NVIDIA driver inside WSL.** The Windows driver is shared.
5. **QES's libraries:**
   ```bash
   sudo apt update
   sudo apt install build-essential cmake git libboost-all-dev libgdal-dev libnetcdf-dev libnetcdf-c++4-dev
   ```
6. **Get the code** inside Linux, not under `/mnt/c`, which is much slower:
   ```bash
   git clone https://github.com/Juliatduffy/QES-benchmarking.git ~/QES
   ```
   (Your repo already contains QES-Public.)
7. **Check:**
   - `nvidia-smi` shows the 4070 Super.
   - `nvcc --version`, `nsys --version` and `ncu --version` each print a version.
   - `nsys stats --help-reports` lists the report names you'll need in Step 1. Look for `cuda_gpu_kern_sum` and `cuda_gpu_mem_time_sum`.
8. **Disk:** the two fire runs write a few GB of output (deleted at the end), so keep ~5 GB free.

### CHPC
- **Nothing to install:** the script loads QES's modules (CUDA 11.8 includes `nsys` and `ncu`).
- **Use the A100 node:** it's the only CHPC GPU you can get counters on. Its `#SBATCH` lines are in Step 1.
- **It's a preemptible guest node:** expect hours to a day of waiting.
- **The Titan X** would give times only (no counters), and never use node **kp297**: its GPU is broken.

**Claude helps with:** any install error. Paste it and I'll tell you the fix.

---

## Step 1: the skeleton (flag, settings, log header)
Copy the top of your `wind_compare.sh` and change it:

1. **`#SBATCH` lines for the A100.** Bash ignores them on the PC, because they're comments.
   ```bash
   #SBATCH -M notchpeak
   #SBATCH --account=owner-gpu-guest
   #SBATCH --partition=notchpeak-gpu-guest
   #SBATCH --qos=notchpeak-gpu-guest
   #SBATCH --gres=gpu:a100:1,nsight:1
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=48G
   #SBATCH -t 2:00:00
   #SBATCH -o ./Testing/logs/gpu_profile.out
   ```
2. **The machine flag:** `MACHINE=$1` (`chpc` or `pc`). Then one `if` block that sets these:

   | Setting | `chpc` | `pc` |
   |---|---|---|
   | `ARCH` (GPU architecture number) | `80` (A100) | `89` (RTX 4070 Super) |
   | load modules | `source QES-Public/CHPC/loadmodules_QES.sh > /dev/null 2>&1` | nothing |
   | `EXTRA_CMAKE` | `-DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include` | empty |
   | `WORK` (big temporary files) | `/scratch/general/vast/$USER/gpu_profile` | `$HOME/gpu_profile` |
   | `KERN_REPORT` (nsys's name for the kernel table) | `gpukernsum` | `cuda_gpu_kern_sum` |
   | `COPY_REPORT` (nsys's name for the copy table) | `gpumemtimesum` | `cuda_gpu_mem_time_sum` |

   - **Why the report names differ:** CHPC's nsys (CUDA 11.8) uses the old names; newer versions renamed them.
   - **Anything else:** print `usage: gpu_profile.sh chpc|pc` and `exit 1`.
3. **The log:** the same as your scripts. `LOG=~/QES/Testing/logs/$(date +%Y-%m-%d_%H-%M-%S)_gpu_profile_$MACHINE.log`, then `exec > "$LOG" 2>&1`.
4. **How `time` prints:** `TIMEFORMAT='whole run: %R s'`. This bash variable makes `time` print one short line instead of three.
5. **The header, 3 lines:**
   ```bash
   echo "QES-Public GPU profile (-s 3)"
   echo "machine: $MACHINE   job: ${SLURM_JOB_ID:-none}   date: $(date '+%Y-%m-%d %H:%M')"
   echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)   CUDA $(nvcc --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
   ```
   `${SLURM_JOB_ID:-none}` means "the job ID, or `none` on the PC".

**Test:** `bash gpu_profile.sh pc` makes a log with exactly those 3 lines. `bash gpu_profile.sh oops` prints the usage line.

---

## Step 2: build QES-Public with CUDA (once per machine)
```bash
BUILD=~/QES/build-gpu-public-sm$ARCH
mkdir -p $WORK
if [ ! -x $BUILD/qesFire/qesFire ]; then
  cmake -S ~/QES/QES-Public -B $BUILD -DENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH $EXTRA_CMAKE > $WORK/build.txt 2>&1
  make -C $BUILD -j8 qesWinds qesFire >> $WORK/build.txt 2>&1 || { echo "build failed, see $WORK/build.txt"; exit 1; }
fi
```
- **The build output goes to `$WORK/build.txt`, not the log.** The log only hears about it if the build fails.
- **Not** `-DHAS_CUDA_SUPPORT=OFF`: this build must include CUDA.
- **Check once:** `grep CUDA $WORK/build.txt` shows `Using CUDA Version` and **not** `Disabling CUDA Support`.
- **The QES-Public rule still applies:** the build directory is outside `QES-Public/`, and nothing is ever written inside it.

**Test:** run `$BUILD/qesWinds/qesWinds -q ~/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml -s 3` once by hand. It should print `Running Global Memory Solver (GPU)` and `Residual after 500 itertations`.

---

## Step 3: the four inputs, in one loop
**Fire setup, once, before the loop:**
```bash
mkdir -p $WORK/run $WORK/out
ln -sfn ~/QES/QES-Public/data $WORK/data                 # qesFire opens ../data/FireFiles/HeatPot.nc
ln -sfn ~/QES/QES-Public/data/FireFiles $WORK/FireFiles  # FFII.xml asks for ../FireFiles/FFII.nc
cd $WORK/run                                             # run everything from here: outside QES-Public
```
**Why the links:** both paths are resolved from the directory you launch in, not from the XML's folder. Starting in `$WORK/run` makes `../data` and `../FireFiles` land on the links, which point (read-only) into QES-Public.

**The loop:** `for CASE_NAME in okc slc fire ffii; do … done`. Inside it:
- **Print the case header first:** `echo; echo "=== $CASE_NAME ==="`.
- **Then set two variables:**
  - `CMD`: the full command
  - `NCU_PICK`: which kernel launches ncu measures (Step 6)

| CASE_NAME | CMD | NCU_PICK |
|---|---|---|
| `okc` | `$BUILD/qesWinds/qesWinds -q ~/QES/QES-Public/tests/runTestsFiles/winds_OklahomaCity.xml -s 3` | `-c 10` |
| `slc` | `$BUILD/qesWinds/qesWinds -q ~/QES/QES-Public/tests/runTestsFiles/winds_SaltLakeCity.xml -s 3` | `-c 10` |
| `fire` | `$BUILD/qesFire/qesFire -q ~/QES/QES-Public/data/InputFiles/fire.xml -s 3 -o $WORK/out/fire -b` | `-k PotGlob -c 4` |
| `ffii` | `$BUILD/qesFire/qesFire -q ~/QES/QES-Public/data/InputFiles/FFII.xml -s 3 -o $WORK/out/ffii -b` | `-k PotGlob -c 4` |

- **qesFire needs `-o … -b`:** without an output file it crashes.
- **Each fire run overwrites its own output,** so only one copy per case exists at a time.
- ⚠️ **Never pass `-f parallel` here.** That would swap the GPU plume code for your CPU one.

**Test:** the loop prints the four headers, and Step 4's run works for each case. Watch FFII in particular: it's the one that has never run on the GPU.

---

## Step 4: stage times (3 lines of code)
```bash
time $CMD > $WORK/${CASE_NAME}_qes.txt 2>&1
echo "stage times: seconds, count, stage"
awk '/^\[/ {stage = $1" "$2" "$3} /elapsed time/ {total[stage] += $(NF-1); count[stage]++} END {for (s in total) printf "%9.3f %5d  %s\n", total[s], count[s], s}' $WORK/${CASE_NAME}_qes.txt | sort -rn
```
- **`time`:** prints `whole run: 7.860 s` (because of `TIMEFORMAT` in Step 1). QES's own output goes to `${CASE_NAME}_qes.txt`; for FFII that's thousands of lines.
- **How the `awk` line works:** it's the same idea as the `awk` in your `wind_compare.sh`, applied to every stage at once:
  - **`/^\[/ {stage = …}`:** every line starting with `[` (like `[QES-Fire] GPU POTENTIAL`) becomes the current stage name. Its first 3 words are kept, so `[QES-Fire] Current time = 15:04:16` and `[QES-Fire] Current time = 15:04:52` count as the same stage.
  - **`/elapsed time/ {…}`:** each time line adds its number to the current stage. `$(NF-1)` = the second-to-last word, i.e. the number before `s`.
  - **`END {…}`:** print each stage's total and count.
  - **`sort -rn`:** biggest first.
- ✅ **Tested by Claude** on the old A100 OKC and SLC logs (the SLC output is the example above) and on the CPU FFII log.

---

## Step 5: GPU kernels and copies (`nsys`)
**A small helper function, defined once near the top of the script:**
```bash
# nsys_table <csv file>: print % of time, total ms, count, name, one row per kernel or copy kind
nsys_table() {
  sed 's/(.*//; s/"//g' "$1" | awk -F, '$1 ~ /^[0-9.]+$/ {printf "%6.1f %9.1f %6d  %s\n", $1, $2/1e6, $3, $NF}'
}
```
- **`sed 's/(.*//'`:** cuts each kernel name at its first `(`. That turns `SOR_RB_Global(float *, int, …)` into `SOR_RB_Global` and removes the commas inside the argument list.
- **`s/"//g`:** removes quote marks.
- **`$1 ~ /^[0-9.]+$/`:** keeps only rows that start with a number, which skips the header.
- **The columns nsys writes:** `Time (%)`, `Total Time (ns)`, `Instances`, … and the name last. `$2/1e6` turns nanoseconds into milliseconds.

**In the loop:**
```bash
nsys profile -t cuda -f true -o $WORK/$CASE_NAME $CMD > /dev/null 2>&1
nsys stats -r $KERN_REPORT,$COPY_REPORT -f csv -o $WORK/$CASE_NAME $WORK/$CASE_NAME.nsys-rep > /dev/null 2>&1
echo "GPU kernels (nsys): % of kernel time, total ms, launches, kernel"
nsys_table $WORK/${CASE_NAME}_$KERN_REPORT.csv
echo "copies (nsys): % of copy time, total ms, count, kind"
nsys_table $WORK/${CASE_NAME}_$COPY_REPORT.csv
```
- **`nsys profile`:** runs QES once and records every kernel launch and copy.
- **`nsys stats … -f csv -o $WORK/$CASE_NAME`:** writes one CSV file per report, named `<case>_<report>.csv`.
- ⚠️ **Not tested yet:** the CSV layout. Claude checked the column order against CHPC's printed tables, but not a CSV file. If a table comes out empty, run `head -3` on the CSV file and paste it.

---

## Step 6: memory vs compute (`ncu`)
```bash
ncu --section SpeedOfLight $NCU_PICK -f -o $WORK/$CASE_NAME $CMD > /dev/null 2>&1
echo "memory vs compute (ncu): DRAM %, SM %, launches measured, kernel"
ncu -i $WORK/$CASE_NAME.ncu-rep | awk '/Context [0-9]/ {kernel = $1; sub(/\(.*/, "", kernel)} /DRAM Throughput/ {dram[kernel] += $NF; n[kernel]++} /Compute \(SM\) Throughput/ {sm[kernel] += $NF} END {for (k in n) printf "%6.1f %6.1f %6d  %s\n", dram[k]/n[k], sm[k]/n[k], n[k], k}' | sort -rn
```
- **`--section SpeedOfLight`:** measures only the "GPU Speed Of Light Throughput" block, which holds `DRAM Throughput` and `Compute (SM) Throughput`.
- **How the `awk` line works:** the same pattern as Step 4.
  - The line naming each measured launch contains `Context 1, Stream 7`. Its first word, cut at `(`, is the kernel name.
  - It adds up each kernel's DRAM % and SM %, then prints the average per kernel.
  - ✅ **Tested by Claude** on the old A100 SLC ncu output (the last two rows of the example).
- **`NCU_PICK` for the wind cases, `-c 10`:** the first 10 launches cover every solver kernel except `finalVelocityGlobal`, which runs once at the very end. The 10 are divergence, then 2 iterations of save/SOR/SOR/boundary/error.
- **`NCU_PICK` for the fire cases, `-k PotGlob -c 4`:** a fire run starts with setup and a wind solve, so the first 10 launches would never reach the fire kernel. This names it. The wind kernels are already covered by okc and slc.
- **Speed:** ncu replays each measured launch ~11 times, so `-c` keeps it short. The rest of the run goes at normal speed.
- **Times come from nsys, not ncu.** ncu slows the GPU clock while it measures, so its durations are left out on purpose.

---

## Step 7: run it and clean up
1. **The PC first** (fast feedback): `bash gpu_profile.sh pc`.
2. **Then CHPC:** `sbatch gpu_profile.sh chpc`.
3. **At the end of the script,** delete the big files: `rm -f $WORK/out/*.nc $WORK/*.nsys-rep $WORK/*.sqlite $WORK/*.ncu-rep`. Keep the `*_qes.txt` and `*.csv` files; they're small and useful.

---

## Time estimate
| Step | Time |
|---|---|
| 0: PC setup (WSL2, CUDA, libraries) | 1–2 h; the biggest risk. CHPC needs none. |
| 1–3: skeleton, build, loop | ~45 min |
| 4–6: the three sections | ~45 min |
| 7: real run | each input runs 3 times. OKC and SLC take seconds; `fire` ~20 s per run on the A100 (~0.33 s per step); FFII is unknown, likely minutes. |

**If the PC setup drags:** develop on CHPC instead. Use a **Titan X** job (it often starts within minutes; add `--exclude=kp297`) to test Steps 1–5. Its ncu section will be empty; that's expected. Then submit the A100 run for the counters.

**Untested parts:**
- FFII on the GPU.
- The `FireFiles` link. The 2026-10-13 FFII run used a copy of the XML with an absolute path instead.
- The nsys CSV files (Step 5).
- `ncu` on the PC's newer CUDA.

Check each one on the first run.

---

## Already known (A100, job 16308989), useful for comparing
| Case | Fact |
|---|---|
| OKC | `calculateErrorGlobal2` is 78% of GPU kernel time; SOR only 18% |
| SLC | `SOR_RB_Global` is 74% of kernel time; DRAM 74% vs SM 39% |
| fire (`fire_short`) | per step: wind solve 0.27 s (~81%), potential 0.042 s (~13%), level set + move ~0.013 s |
| FitFloat on e..n | Titan X: solver 1.055× faster. A100: 0.75× (slower); SOR went from 73% DRAM to 18% DRAM / 51% SM |

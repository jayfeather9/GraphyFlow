# Cycle Counting with Vitis/XRT (baseline + 4 crossbar variants)

This document explains how to measure **hardware cycle counts** (not host “ms time”) for:

- **big-only**: `graphyflow_big` kernel executions
- **end-to-end (E2E)**: a whole iteration’s kernel span

…for **baseline omega** and all 4 crossbar variants (`plain`, `rr`, `voq`, `voq_rr`) on the **same dataset** (e.g. `2000 nodes / 4000 edges`).

Why this exists:
- The host prints times in `ms`, but under `hw_emu` those are **simulation / driver timestamps** and are not proportional to on-FPGA cycles.
- For cycle-accurate work you must use **Vitis/XRT trace artifacts** and extract kernel start/end at the device level.

---

## 0) One-time setup: where tracing is controlled

### 0.1 Runtime trace config: `generated_project/xrt.ini`

`generated_project/run.sh` always exports:

- `XRT_INI_PATH=./xrt.ini`

So the file `generated_project/xrt.ini` is the runtime switchboard for trace collection.

Performance note:
- A “full” trace configuration (e.g. `device_trace=fine`, `stall_trace=true`, `continuous_trace=true`) can make `hw_emu` runs **extremely slow** even for tiny graphs.
- If you only need correctness, temporarily switch to a minimal `xrt.ini` (or comment out most keys) and keep the full-trace version for cycle-counting runs.

### 0.2 Build-time instrumentation: `PROFILE=1`

Even if `xrt.ini` requests traces, you generally also need the xclbin to be linked with profiling infrastructure enabled.

In this repo, the knob is already wired in:

- `generated_project/scripts/kernel/kernel.mk`:
  - `make ... PROFILE=1` adds `--profile.exec all:all` to the `v++ -l` link step.

If you do **not** build with `PROFILE=1`, it is common to see empty or missing device trace outputs.

---

## 1) Build a profiled xclbin (recommended: per-variant build dirs)

Do **not** run different variants in the same `generated_project/` directory if you care about preserving artifacts; traces and logs will overwrite each other.

Use the parallel-build approach described here:

- `docs/parallel_hw_build_variants.md`

But add `PROFILE=1` to the build commands.

Example (hardware emulation):

```bash
source /home/feiyang/set_env.sh

# In each variant build directory:
make all TARGET=hw_emu PROFILE=1 2>&1 | tee logs/make_all_hw_emu.profile.log
```

Notes:
- For `hw_emu`, ensure the emulation config exists (`make emconfig TARGET=hw_emu` if needed).
- For real board runs, use `TARGET=hw` instead (server build recommended).

---

## 2) Run with a fixed dataset (example: 2000/4000)

Inside each build directory (baseline + 4 variants):

```bash
python3 gen_random_graph.py 2000 4000

ts=$(date +%Y%m%d_%H%M%S)
./run.sh hw_emu 2>&1 | tee "logs/run_hw_emu.profile.${ts}.log"
```

Expected correctness line:
- `SUCCESS: Results match!`

---

## 3) Collect trace artifacts (per directory)

After a profiled run you should see some combination of:

- `xrt.run_summary` (JSON index of trace artifacts)
- `opencl_trace.csv` and/or `N-opencl_trace.csv`
- `device_trace_*.csv` (device-side timeline events)
- `summary.csv` (summary tables, depending on XRT version)

Keep them per variant directory. If you want an explicit “artifact bundle”, copy:

```bash
mkdir -p artifacts
cp -f xrt.run_summary *.csv artifacts/ 2>/dev/null || true
```

If `device_trace_*.csv` is missing/empty:
- confirm you built the xclbin with `PROFILE=1` (check your link log for `--profile.exec`)
- confirm `run.sh` is pointing at the correct `xrt.ini`
- check that `xrt.ini` contains the `[Debug]` keys (this repo already provides a sane default)

---

## 4) View and export kernel executions (Vitis Analyzer)

From inside the build directory:

```bash
vitis -a --analyze xrt.run_summary
```

In Vitis Analyzer:
1) open the **Timeline Trace**
2) filter by kernel name (e.g. `graphyflow_big`)
3) ensure you can see individual CU invocations (`graphyflow_big_1`, `graphyflow_big_2`, `graphyflow_big_3`)
4) export the kernel execution table to CSV (menu varies by version; typically “Export” from the trace table)

What you need in the export:
- kernel/CU identifier
- start timestamp
- end timestamp (or duration)

This gives you a cycle-derivable ground truth.

---

## 5) Turn timestamps into per-iteration cycle counts

## 5.0 Converting time → cycles (when your export is time-based)

Depending on your XRT/Vitis version, exported kernel events may report:
- timestamps in **ns** (preferred), or
- durations in **ns**, or
- timestamps in **us/ms**.

To convert a duration to cycles, you need the kernel clock frequency used for that run.

Typical conversion:
- `cycles = duration_seconds * f_hz`
- `cycles = duration_ns * f_mhz / 1000`

Example at 250 MHz:
- `1 ns` ≈ `0.25 cycles`
- `1000 ns` = `1 us` = `250 cycles`

If Vitis Analyzer shows cycles directly for kernel events, use that (no conversion needed).

### 5.1 Big-only (per iteration)

Goal:
- for each iteration, compute `cycles_big_iter = max(duration_cycles(graphyflow_big_1..3))`

Rationale:
- the 3 big CUs run in parallel; the iteration’s big stage time is dominated by the slowest CU.

### 5.2 E2E (per iteration)

Goal:
- for each iteration, compute a single E2E cycle span that includes all kernels launched that iteration.

Typical definition:
- `cycles_e2e_iter = end_of_last_kernel_in_iter - start_of_first_kernel_in_iter`

### 5.3 Compare against baseline

For the same dataset, produce a table:

- baseline omega: `cycles_big_iter`, `cycles_e2e_iter`
- xbar plain:     same
- xbar rr:        same
- xbar voq:       same
- xbar voq_rr:    same

This is the comparison the project needs (not host ms).

---

## 6) Practical tips

- Start with a small dataset (`64/256`) to verify traces are generated and exports look sane.
- `hw_emu` is slow; `2000/4000` can take a long time to simulate. Consider running fewer iterations for profiling if the host supports it.
- Always log both build and run commands with `tee`.

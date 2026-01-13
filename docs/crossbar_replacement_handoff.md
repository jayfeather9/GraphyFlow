# Crossbar Replacement Handoff (Omega → XBAR) — Status + Next Steps

This file is a **single, verbose, end-to-end handoff** so that anyone can immediately continue the work without asking the previous agent questions.

It records:
- what was requested,
- what has been implemented so far,
- how to build/run with logs,
- what broke (and why),
- what must be done next (including commits and gitignore expectations),
- where to edit for each future variant.

---

## 0) What the user requested (authoritative requirements)

### Goal
Replace the current **8×8 omega network** used in the `graphyflow_big.cpp` pipeline with a **crossbar** network, then implement **4 crossbar variants**:

1. **plain** (no VOQ, no RR) — *implement first, test in hw_emu*  
2. **rr** (round-robin arbitration per output)
3. **voq** (virtual output queues)
4. **voq_rr** (VOQ + per-output round-robin)

### Crossbar behavior
- Crossbar has `8` input streams and `8` output streams.
- Each stream carries an item type with `end_flag`.
- Each **input stream ends independently**:
  - when an `end_flag` token arrives on that input, that input is “closed” and will produce no further data
  - other inputs may still have data
- The crossbar **stops** only when:
  - *all 8 inputs have ended*, and
  - *all internal buffers are drained*, and
  - output end tokens have been emitted (as appropriate)

### II=1 requirement
Every function that is part of the DATAFLOW streaming pipeline must:
- contain a loop with `#pragma HLS PIPELINE II=1`
- and the compile reports/log must show the achieved II is `1`

### Testing expectations
- Start with `hw_emu` on small random graphs (e.g. 64 nodes / 256 edges).
- If plain version works, try larger graphs (e.g. 1524 nodes / 2548 edges or bigger).
- If debugging is hard, create **minimal test designs** that isolate the crossbar network.

### Project organization
- Put all 4 variants in a new folder **outside** `generated_project`, e.g.:
  - `xbar_variants/plain`
  - `xbar_variants/rr`
  - `xbar_variants/voq`
  - `xbar_variants/voq_rr`
- You may split the kernel into multiple smaller kernels if needed, connected via streams.

### Logging
- Every compilation/run should be logged with `tee` (and not hide errors).

### Git discipline
- “Always remember to do git commit after you done some work.”
- Ensure all Vitis-generated intermediates are ignored (`_x`, `.Xil`, `xclbin/`, `*.log`, `.run`, etc).

---

## 1) Environment + build/run procedure (known working)

### New: parallel hardware builds on a server (`TARGET=hw`)
If you want to compile **baseline + all 4 crossbar variants** in parallel (e.g., on a server), use:

- `docs/parallel_hw_build_variants.md`

That document is written for a newcomer and explains how to:
- generate a clean baseline `generated_project/`,
- create **5 independent** build directories (baseline + 4 variants),
- swap in `xbar_variants/*/graphyflow_big.*`,
- run `make all TARGET=hw` in parallel with per-variant logs.

### Environment activation
The environment is sourced using fish’s `bass`:

```fish
bass source /home/feiyang/set_env.sh
```

`/home/feiyang/set_env.sh` sources Vitis 2024.1 + XRT:
```bash
source /opt/Xilinx2024/Vitis/2024.1/settings64.sh
source /opt/xilinx/xrt/setup.sh
```

If you’re using bash/zsh directly (no fish), the equivalent is:

```bash
source /home/feiyang/set_env.sh
```

### Standard build/run (with log capture)
From repo root:

```fish
cd generated_project
bass source /home/feiyang/set_env.sh
mkdir -p logs

# generate input graph
python3 gen_random_graph.py 64 256

# build (hw_emu) with log
set ts (date +%Y%m%d_%H%M%S)
make all TARGET=hw_emu &| tee "logs/make_all_hw_emu.$ts.log"

# run (hw_emu) with log
set ts (date +%Y%m%d_%H%M%S)
./run.sh hw_emu &| tee "logs/run_hw_emu.$ts.log"
```

For bash, the same idea is:

```bash
cd generated_project
source /home/feiyang/set_env.sh
mkdir -p logs

python3 gen_random_graph.py 64 256

ts=$(date +%Y%m%d_%H%M%S)
make all TARGET=hw_emu 2>&1 | tee "logs/make_all_hw_emu.$ts.log"

ts=$(date +%Y%m%d_%H%M%S)
./run.sh hw_emu 2>&1 | tee "logs/run_hw_emu.$ts.log"
```

### Important: socket restrictions and “local port errors”
Observed earlier:
- `v++`/`xcd` sometimes prints repeated lines like:
  - `ERROR: exception getting local port: open: Operation not permitted`

Cause:
- local socket syscalls can be blocked in a sandboxed environment.

Mitigation:
- run the build in an environment that allows local sockets (normal terminal, or whatever “full access” mode is available).
- When sockets are allowed, the log instead shows:
  - `Running Dispatch Server on port: <port>`
  - and the repeated “exception getting local port” spam disappears.

Related sandbox symptom:
- Vivado may also complain about not being able to write to `~/.Xilinx/.../XilinxTclStore`. That’s another sign you’re in a restricted environment; use a “full access” terminal/session.

### Fast compile-only (for II=1 checks)
If you want to rebuild only the `graphyflow_big` kernel XO (without linking the full xclbin), from `generated_project/`:

```bash
rm -f xclbin/graphyflow_big.hw_emu.xo xclbin/graphyflow_big.hw_emu.xo.compile_summary
ts=$(date +%Y%m%d_%H%M%S)
make xclbin/graphyflow_big.hw_emu.xo TARGET=hw_emu 2>&1 | tee "logs/make_big_xo_hw_emu.$ts.log"
```

---

## 2) Known-good baseline run (before crossbar changes)

Before starting crossbar work, a successful `hw_emu` build+run was achieved.

Artifacts:
- Build log example: `generated_project/logs/make_all_hw_emu.20260108_151316.log`
- Run log example: `generated_project/logs/run_hw_emu.20260108_152806.log`
- The run reported: `SUCCESS: Results match!`

This confirmed:
- The overall generated project flow can build and run `hw_emu`.
- The host verification path works (at least for that config).

---

## 3) Where the omega network is today (what to replace)

### Current omega-like network usage inside `graphyflow_big.cpp`
In `generated_project/scripts/kernel/graphyflow_big.cpp`, the network is currently implemented as **12 calls** to `switch2x2_2`, arranged as 3 stages of 2×2 switches for `N=8` lanes:
- Stage 0: 4 switches (pairs 0-1, 2-3, 4-5, 6-7)
- Stage 1: 4 switches (0-4, 1-5, 2-6, 3-7)
- Stage 2: 4 switches (0-4, 1-5, 2-6, 3-7)

The relevant streams:
- inputs to the network: `reduce_105_d2o_pair[8]`
- outputs of the network: `reduce_105_o2u_pair[8]`

The replacement target is the **block between**:
- `demux_1(stream_edge_data, reduce_105_d2o_pair, total_edge_sets);`
and:
- `Reduc_105_unit_reduce_single_pe(reduce_105_o2u_pair[pe_idx], ...)`

---

## 4) What was implemented so far (plain crossbar, “with nothing”)

### New folder created
Four variant folders exist under `xbar_variants/`:
- `xbar_variants/plain`
- `xbar_variants/rr`
- `xbar_variants/voq`
- `xbar_variants/voq_rr`

### Copy of kernel source used as the working base
Each variant currently carries a full copy of the baseline big-kernel source (so you can swap it into `generated_project/` without touching the generator yet):
- `xbar_variants/plain/graphyflow_big.cpp` + `xbar_variants/plain/graphyflow_big.h`
- `xbar_variants/rr/graphyflow_big.cpp` + `xbar_variants/rr/graphyflow_big.h`
- `xbar_variants/voq/graphyflow_big.cpp` + `xbar_variants/voq/graphyflow_big.h`
- `xbar_variants/voq_rr/graphyflow_big.cpp` + `xbar_variants/voq_rr/graphyflow_big.h`

### Plain crossbar implementation
In `xbar_variants/plain/graphyflow_big.cpp`, a new function was added:

- `static void crossbar_skid_fixed(hls::stream<update_t_big> in_streams[PE_NUM], hls::stream<update_t_big> out_streams[PE_NUM])`

Behavior (as implemented):
- The implementation is a **DATAFLOW crossbar**:
  - **8× input demux** tasks: `xbar_input_demux_fixed(...)`
    - reads one input stream until `end_flag`
    - routes each item to `in2out[in][dst]` where `dst = node_id[LOG_PE_NUM-1:0]`
    - on `end_flag`, **broadcasts an end token to all 8 outputs** for that input (one end per `dst`)
  - **8× output merge** tasks: `xbar_output_merge_fixed(...)`
    - for a fixed output `out_id`, merges the 8 input-to-this-output streams using a **3-stage 2:1 merge tree**
    - each merge node (`xbar_merge2_fixed(...)`) is fixed-priority (left input wins) and **filters end tokens**
    - the root emits **exactly one end token** when all 8 inputs have ended for this output

Why it was refactored this way:
- the earlier “scan 8 inputs per output” arbiter hit very low estimated fmax and couldn’t reach II=1 reliably
- the merge-tree version achieves II=1 and meets the ≥250MHz estimated-frequency goal for plain

### Omega replacement
The previous 12 `switch2x2_2(...)` calls were removed and replaced with:

```cpp
crossbar_skid_fixed(reduce_105_d2o_pair, reduce_105_o2u_pair);
```

### Implementation note (HLS friendliness)
The implementation was refactored into a **balanced 2:1 merge tree** so each merge node has small local control logic and an explicit `#pragma HLS PIPELINE II=1` loop, which substantially improves estimated fmax vs a single “8-way arbiter” block.

### Plain variant validated (hw_emu)
Validation was completed using the “copy into `generated_project/` and build/run” workflow:
- Build log: `generated_project/logs/make_all_hw_emu.20260108_175220.log`
- Run log: `generated_project/logs/run_hw_emu.20260108_180137.log` (reported `SUCCESS: Results match!`)
- II=1 evidence: `generated_project/_x/reports/graphyflow_big.hw_emu/v++_compile_graphyflow_big.hw_emu_guidance.html` contains `Final II = 1` for loop `LOOP_WHILE_XBAR_SKID_FIXED`.

### Remaining variants implemented (not yet end-to-end hw_emu validated)
These variants are implemented in `xbar_variants/` and were compile-checked (XO compile) for II=1:

- `rr`: `crossbar_skid_rr(...)` main loop label `LOOP_WHILE_XBAR_SKID_RR` (example log: `generated_project/logs/make_big_xo_hw_emu.rr.20260108_184945.log`)
- `voq`: `crossbar_voq_fixed(...)` main loop label `LOOP_WHILE_XBAR_VOQ_FIXED` and end-flush loop `LOOP_WHILE_XBAR_VOQ_FLUSH_ENDS` (example log: `generated_project/logs/make_big_xo_hw_emu.voq.20260108_200705.log`)
- `voq_rr`: `crossbar_voq_rr(...)` main loop label `LOOP_WHILE_XBAR_VOQ_RR` and end-flush loop `LOOP_WHILE_XBAR_VOQ_RR_FLUSH_ENDS` (example log: `generated_project/logs/make_big_xo_hw_emu.voq_rr.20260108_200903.log`)

Note on VOQ implementations:
- they avoid `.empty()`/`.full()` query fanout by tracking per-VOQ occupancy counters, and they flush end tokens in a separate loop after the data loop drains (to keep the data loop at II=1).

### End-to-end hw_emu validation (rr/voq/voq_rr)
All three variants were validated end-to-end via `make all TARGET=hw_emu` + `./run.sh hw_emu` after copying the variant sources into `generated_project/scripts/kernel/`:

- RR:
  - Build log: `generated_project/logs/make_all_hw_emu.xbar_rr.20260108_201913.log`
  - Run log: `generated_project/logs/run_hw_emu.xbar_rr.20260108_202629.log` (`SUCCESS: Results match!`)
- VOQ:
  - Build log: `generated_project/logs/make_all_hw_emu.xbar_voq.20260108_202928.log`
  - Run log: `generated_project/logs/run_hw_emu.xbar_voq.20260108_203705.log` (`SUCCESS: Results match!`)
- VOQ+RR:
  - Build log: `generated_project/logs/make_all_hw_emu.xbar_voq_rr.20260108_204139.log`
  - Run log: `generated_project/logs/run_hw_emu.xbar_voq_rr.20260108_204913.log` (`SUCCESS: Results match!`)

### Estimated frequency for `graphyflow_big` (Vitis `system_estimate_*.xtxt`)

All numbers below come from `v++ -c`’s `**** Estimated Fmax: ... MHz` line when building the **big kernel XO**:

```bash
cd generated_project
source /home/feiyang/set_env.sh
make xclbin/graphyflow_big.hw_emu.xo TARGET=hw_emu
```

Results (same platform, same toolchain):

| Variant | Estimated Fmax (MHz) | Evidence (log) |
|---|---:|---|
| baseline omega | 318.67 | `generated_project/logs/make_big_xo_hw_emu.baseline_omega.20260113_172449.log` |
| xbar plain | 256.17 | `generated_project/logs/make_big_xo_hw_emu.xbar_plain_tree.20260113_181408.log` |
| xbar rr | 318.67 | `generated_project/logs/make_big_xo_hw_emu.xbar_rr_tree2.20260113_182919.log` |
| xbar voq | 24.16 | `generated_project/logs/make_big_xo_hw_emu.xbar_voq.20260113_183642.log` |
| xbar voq_rr | 17.45 | `generated_project/logs/make_big_xo_hw_emu.xbar_voq_rr.20260113_184222.log` |

Notes:
- “xbar plain” and “xbar rr” were explicitly refactored to meet the ≥250MHz goal.
- `voq` and `voq_rr` currently have **very low estimated fmax** and will require a deeper redesign (the current implementation is functionally correct but not timing-friendly).

### Cycle counts on a shared dataset (2000 nodes / 4000 edges)
Dataset generation (done once, reused across runs):
- `cd generated_project && python3 gen_random_graph.py 2000 4000` (writes `generated_project/graph.txt`)

Important correction:
- The numbers printed by the host as `... Time = ... ms` are **runtime/driver timestamps under emulation**, and **must not** be converted into “hardware cycles” by assuming a fixed clock period.
- For cycle counts you must use Vitis/XRT profiling/trace (or hardware counters) and extract cycles from those artifacts.

What is still needed (TODO):
- Re-run baseline + 4 variants with **Vitis/XRT trace enabled** and extract **per-iteration** cycle counts:
  - **big-only**: `graphyflow_big` (e.g., max across the 3 big CUs per iteration)
  - **end-to-end**: full iteration span (all kernels that execute in the iteration)
- Compare against the **baseline omega** build on the same dataset.

Instructions (newcomer-friendly, includes export + parsing approach):
- `docs/vitis_cycle_counting.md`

Run logs for correctness (same dataset, hw_emu):
- Plain: `generated_project/logs/run_hw_emu.xbar_plain.n2000_e4000.20260108_205756.log`
- RR: `generated_project/logs/run_hw_emu.xbar_rr.n2000_e4000.20260108_211836.log`
- VOQ: `generated_project/logs/run_hw_emu.xbar_voq.n2000_e4000.20260108_214737.log`
- VOQ+RR: `generated_project/logs/run_hw_emu.xbar_voq_rr.n2000_e4000.20260108_220859.log`

---

## 5) How to integrate the variant into the build (current approach)

Because `generated_project/` is *generated output*, we are not trying to permanently “own” its sources in git. However, to test `hw_emu` quickly, we used this workflow:

1) develop variant in `xbar_variants/plain/graphyflow_big.cpp`
2) copy into generated project before building:

```bash
cp -a xbar_variants/plain/graphyflow_big.cpp generated_project/scripts/kernel/graphyflow_big.cpp
cp -a xbar_variants/plain/graphyflow_big.h   generated_project/scripts/kernel/graphyflow_big.h
```

This lets you validate the design in `hw_emu` without modifying the Python generator yet.

When switching variants, it helps to force a rebuild of the big kernel:
- delete `generated_project/xclbin/graphyflow_big.hw_emu.xo*` before running `make`.
- if you hit weird cached behavior, it’s safe to delete `generated_project/_x`, `generated_project/.Xil`, and `generated_project/.run` (they are regenerated).

**Long term**, once all variants are stable:
- add a generator option to emit the desired network variant (omega vs xbar variant).

---

## 6) Practical build blockers (sandbox + disk)

### Sandbox restrictions (observed)
In sandboxed environments, `v++`/Vivado may fail during `hw_emu` (e.g. “local port” errors, and/or Vivado complaining about not being able to write under `~/.Xilinx`).

Mitigation:
- run the build/run in a “full access” environment where local IPC and user-home writes are allowed.
- when working, the log shows `Running Dispatch Server on port: <port>` (instead of failing early).

### `hw_emu` link/elaborate failure: `pfm_top_wrapper` not found
Observed failure mode when linking `TARGET=hw_emu`:
- `ERROR: [VPL 60-704] Integration error, Step failed: config_hw_emu.elaborate`
- `ERROR: [XSIM 43-3225] Cannot find design unit xil_defaultlib.pfm_top_wrapper`
- plus many Vivado warnings about “IP is locked … customized with software release 2022.1 … different revision in the IP Catalog”

Notes / mitigations that are worth trying:
- Ensure `LIBRARY_PATH` is not exported in the environment that runs `v++ -l` (Vivado/xelab warns about it); run the link as:
  - `env -u LIBRARY_PATH make xclbin/graphyflow_kernels.hw_emu.xclbin TARGET=hw_emu`
- If your platform/IP was built with an older Vivado (2022.x), try using the matching toolchain for `hw_emu` link (source that version’s `settings64.sh`) so Vivado can elaborate the platform IP cleanly.
- If you already have a known-good `hw_emu` build directory, avoid deleting `.Xil/` and `.run/` unless necessary; rebuilding from scratch can surface tool/platform version mismatches.

### Disk usage (possible)
`generated_project/_x` can get very large during `hw_emu` builds. If disk space gets tight, it’s safe to delete:
- `generated_project/_x`
- `generated_project/.run`
- `generated_project/.Xil`

---

## 7) Current repository state (what is dirty, what is not)

### `generated_project/` is not tracked by git
`generated_project/` appears to be untracked. That means:
- changes inside it won’t be committed unless explicitly added (don’t add it).

### Commits made so far
- Generator defaults (11 little + 3 big, HBM, 250 MHz): commit `76b4560`
- Crossbar variant skeleton + plain implementation + docs: commit `05eb326`
- RR variant: commit `6f66ba5`
- VOQ variant: commit `738deb0`
- VOQ+RR variant: commit `f265368`

---

## 8) Next engineering steps (after plain passing)

### Step A — Scale input size
After correctness:
- generate a larger random graph or provide a real graph file
- run hw_emu again
- check throughput / runtime

### Step B — Implement the remaining variants
Implement in separate variant folders:

#### `xbar_variants/rr`
- same skid-buffer load phase as plain
- per-output `rr_ptr[out]` rotates scan start
- update pointer only on successful write

#### `xbar_variants/voq`
- maintain `voq[in][out]` FIFOs (depth = local_depth-like constant)
- each input reads at most 1 item per cycle, enqueues into its VOQ bucket
- arbitration:
  - outputs choose among inputs that have non-empty `voq[in][out]`
  - enforce “one item per input per cycle” constraint (`input_used[in]`)

#### `xbar_variants/voq_rr`
- same VOQ, but per-output RR ordering

### Step C — Validate rr/voq/voq_rr end-to-end in hw_emu
For each variant (`rr`, `voq`, `voq_rr`):
1) copy the variant into `generated_project/scripts/kernel/graphyflow_big.cpp` + `.h`
2) run `make all TARGET=hw_emu` (or at least the big-kernel XO compile for II checks)
3) run `./run.sh hw_emu` and confirm `SUCCESS: Results match!`

### Step E — Minimal-test kernel (if needed)
If integration debugging is painful, create a minimized HLS test kernel:
- 8 input streams driven by synthetic generators
- the crossbar
- 8 sinks that count per-destination items and check order/termination

This can be compiled faster than the full graphyflow_big.

---

## 9) Git + commit plan (after you get plain passing)

### Gitignore expectations
The repo `.gitignore` already contains Xilinx-related ignores:
- `_x/`, `xclbin/`, `.Xil/`, `.run/`, `*.log`, etc.

Consider adding explicit ignores for the generated project paths as extra safety:
- `generated_project/_x/`
- `generated_project/.Xil/`
- `generated_project/.run/`
- `generated_project/xclbin/`
- `generated_project/logs/`

### Committing strategy
Suggested incremental commits:
1) “Add xbar_variants skeleton + plain crossbar implementation”
2) “Add rr crossbar variant”
3) “Add voq crossbar variant”
4) “Add voq+rr crossbar variant”
5) “(optional) Add generator flag to emit network variant”

Each commit should be small, buildable, and testable.

---

## 10) Exact files to edit (cheat sheet)

### Variant sources (owned by us)
- `xbar_variants/plain/graphyflow_big.cpp`
- `xbar_variants/rr/graphyflow_big.cpp`
- `xbar_variants/voq/graphyflow_big.cpp`
- `xbar_variants/voq_rr/graphyflow_big.cpp`

### Generated project integration (for testing only)
- `generated_project/scripts/kernel/graphyflow_big.cpp`
- `generated_project/scripts/kernel/graphyflow_big.h`

### Build/run logs
- `generated_project/logs/`

---

## 11) What is still missing right now (as of writing this file)

1) **Cycle-count measurements** (baseline + 4 variants) are still not recorded in this doc.
   - Use `docs/vitis_cycle_counting.md` to re-run with profiling/trace enabled and extract per-iteration cycles (big-only + E2E), then paste results here.

2) **VOQ timing closure**:
   - `voq` and `voq_rr` are functionally correct in `hw_emu`, but their current estimated fmax is extremely low (see the table above).
   - They likely need a structural refactor similar to the merge-tree approach (i.e., break up wide arbitration / avoid long combinational paths) before they are usable for ≥250MHz targets.

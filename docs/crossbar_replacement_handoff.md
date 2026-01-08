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
Four folders were created (only `plain` has implementation so far; others are placeholders so the folder layout is tracked in git):
- `xbar_variants/plain`
- `xbar_variants/rr`
- `xbar_variants/voq`
- `xbar_variants/voq_rr`

### Copy of kernel source used as the working base
The baseline big-kernel source was copied into:
- `xbar_variants/plain/graphyflow_big.cpp`
- `xbar_variants/plain/graphyflow_big.h`

### Plain crossbar implementation
In `xbar_variants/plain/graphyflow_big.cpp`, a new function was added:

- `static void crossbar_skid_fixed(hls::stream<update_t_big> in_streams[PE_NUM], hls::stream<update_t_big> out_streams[PE_NUM])`

Behavior (as implemented):
- Per input:
  - one skid buffer item (`input_buf[i]`) + `input_valid[i]`
  - `end_flag` tokens are consumed and mark that input as closed (`input_closed[i]`)
- Per output, per cycle:
  - fixed priority scan of inputs 0..7
  - if an input’s buffered item maps to this output (destination computed from `node_id[LOG_PE_NUM-1:0]`), it can be written
- Termination:
  - once all inputs are closed and all skid buffers are empty, the crossbar enters a “flush end” phase
  - it writes one end token to each output stream and then stops

### Omega replacement
The previous 12 `switch2x2_2(...)` calls were removed and replaced with:

```cpp
crossbar_skid_fixed(reduce_105_d2o_pair, reduce_105_o2u_pair);
```

### Implementation note (HLS friendliness)
The implementation was refactored to avoid `break` inside fully-unrolled loops (priority selection is done via one-hot + encoder), to keep synthesis/pipelining predictable.

### Plain variant validated (hw_emu)
Validation was completed using the “copy into `generated_project/` and build/run” workflow:
- Build log: `generated_project/logs/make_all_hw_emu.20260108_175220.log`
- Run log: `generated_project/logs/run_hw_emu.20260108_180137.log` (reported `SUCCESS: Results match!`)
- II=1 evidence: `generated_project/_x/reports/graphyflow_big.hw_emu/v++_compile_graphyflow_big.hw_emu_guidance.html` contains `Final II = 1` for loop `LOOP_WHILE_XBAR_SKID_FIXED`.

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

**Long term**, once all variants are stable:
- add a generator option to emit the desired network variant (omega vs xbar variant).

---

## 6) Practical build blockers (sandbox + disk)

### Sandbox restrictions (observed)
In sandboxed environments, `v++`/Vivado may fail during `hw_emu` (e.g. “local port” errors, and/or Vivado complaining about not being able to write under `~/.Xilinx`).

Mitigation:
- run the build/run in a “full access” environment where local IPC and user-home writes are allowed.
- when working, the log shows `Running Dispatch Server on port: <port>` (instead of failing early).

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
- `xbar_variants/rr/graphyflow_big.cpp` (to be created)
- `xbar_variants/voq/graphyflow_big.cpp` (to be created)
- `xbar_variants/voq_rr/graphyflow_big.cpp` (to be created)

### Generated project integration (for testing only)
- `generated_project/scripts/kernel/graphyflow_big.cpp`
- `generated_project/scripts/kernel/graphyflow_big.h`

### Build/run logs
- `generated_project/logs/`

---

## 11) What is still missing right now (as of writing this file)

1) RR/VOQ/VOQ+RR variants are not implemented yet.
2) Larger-graph `hw_emu` runs haven’t been done yet (only small random graph validation was performed).

The plain crossbar is validated in `hw_emu` and has II=1 on its main loop; the remaining work is straightforward iterative implementation + testing + commits.

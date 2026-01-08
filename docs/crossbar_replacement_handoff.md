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
Four folders were created (only `plain` has code so far):
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

### Important note (likely future compile issue)
The initial implementation used:
- a `break;` inside a fully-unrolled scan loop.

In HLS, `break` inside an unrolled loop can be problematic or lead to unexpected synthesis behavior.
If compilation fails or II != 1, refactor the arbitration selection to avoid `break`:
- compute `winner` using a priority encoder style (no loop early-exit)
- or use `read_nb` / `write_nb` patterns

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

## 6) Major blocker encountered: disk full (build cannot proceed)

### What happened
While attempting to rebuild `hw_emu` for the new crossbar kernel, the system hit:
- `OSError: [Errno 28] No space left on device`

This prevented:
- `bass` from creating temporary files,
- `tee` from writing logs,
- and `make` from proceeding reliably.

### Root cause
`generated_project/_x` became enormous:
- `generated_project/_x` was measured at ~359 GB at one point.

This is consistent with Vitis/Vivado generating large intermediate trees during `hw_emu` builds.

### What was cleaned
The following were deleted to reclaim space:
- `generated_project/_x`
- `generated_project/.run`
- `generated_project/.Xil`

### But disk is still effectively full
Even after deleting those, `df -h /` still reported `Use% 100%` with almost no available space.
This means:
- there is still another large consumer elsewhere on `/`, OR
- space had not been fully released due to still-running processes, OR
- filesystem reserved space / delayed reclaim.

### Processes that were still running
There were long-running `make all TARGET=hw_emu` / `v++` / `vitis_hls` processes (some from prior aborted runs).
They were killed (TERM then KILL).

### What you must do next to unblock work
1) Confirm no Vitis/Vivado processes are still writing:
   ```bash
   ps -u feiyang -o pid,etime,cmd | rg -n "(v\\+\\+|vivado|xsim|vitis_hls|xcd|hw_emu)"
   ```
2) Identify where the remaining disk usage is:
   ```bash
   sudo du -sh /* 2>/dev/null | sort -h | tail -n 50
   du -sh /home/feiyang/* 2>/dev/null | sort -h | tail -n 50
   ```
3) Delete/move large, non-essential data until you have **tens of GB free**.
   - The build system can easily recreate `_x` again.
4) Only after reclaiming space, retry the build.

---

## 7) Current repository state (what is dirty, what is not)

### `generated_project/` is not tracked by git
`generated_project/` appears to be untracked. That means:
- changes inside it won’t be committed unless explicitly added (don’t add it).

### Tracked files modified earlier (not related to crossbar)
There are tracked modifications in:
- `graphyflow/kernel_numbers.py` (changed kernel counts)
- `graphyflow/project_generator.py` (changed `USE_DDR` and frequency)

These were made earlier to align the generated project with:
- 11 little + 3 big kernels
- HBM usage
- 250 MHz frequency

Before committing crossbar work, decide whether these config changes are intended for the main branch or should be reverted.

### Untracked docs
- `docs/agent_hw_emu_runbook.md` was created earlier.
- `docs/network_variants.md` exists but currently appears untracked as well.

Decide whether to add them to git.

---

## 8) Next engineering steps (once disk is unblocked)

### Step A — Compile the plain crossbar and confirm II=1
1) Copy plain variant into `generated_project/scripts/kernel/graphyflow_big.cpp`
2) Build `hw_emu` and log it with `tee`
3) Verify in the build log that the crossbar loop is II=1
   - Search the v++ HLS log lines for the crossbar loop label (add a unique loop label if needed).

If II != 1:
- reduce control complexity in the loop:
  - avoid `break` in unrolled loops,
  - avoid multiple writes to the same output in one cycle,
  - prefer `read_nb`/`write_nb` or explicit `.empty()`/`.full()` gating.

### Step B — Run hw_emu and verify correctness
Run:
```bash
./run.sh hw_emu |& tee logs/run_hw_emu.xbar_plain.<ts>.log
```
Success criteria:
- The host comparison reports `SUCCESS: Results match!`

### Step C — Scale input size
After correctness:
- generate a larger random graph or provide a real graph file
- run hw_emu again
- check throughput / runtime

### Step D — Implement the remaining variants
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

1) **Disk space is still effectively full**, preventing reliable builds.
2) Plain crossbar has been coded but **not yet validated** in `hw_emu` after the omega replacement, because builds were blocked by disk.
3) RR/VOQ/VOQ+RR variants are not implemented yet.
4) No commits have been made for the crossbar work yet.

Once disk space is restored and the plain crossbar passes `hw_emu`, the rest is straightforward iterative implementation + testing + commits.


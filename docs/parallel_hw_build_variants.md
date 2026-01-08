# Parallel `TARGET=hw` builds for 4 crossbar variants (and baseline)

This repo contains four `graphyflow_big` crossbar variants under `xbar_variants/`:

- `xbar_variants/plain` (fixed-priority skid-buffer crossbar)
- `xbar_variants/rr` (skid-buffer + per-output round-robin)
- `xbar_variants/voq` (virtual-output-queues + fixed priority)
- `xbar_variants/voq_rr` (VOQ + per-output round-robin)

The safest way to compile them **in parallel** is to build each variant in its **own generated project directory**, so Vitis/Vivado intermediate directories (`_x`, `.Xil`, `.run`, `xclbin/`) never collide.

This document is written for someone with **no prior knowledge** of this repo.

---

## 0) What you will produce

After these steps you will have five independent build directories (example names):

- `build_hw/baseline_omega/` → baseline generated project (omega network)
- `build_hw/xbar_plain/` → crossbar `plain`
- `build_hw/xbar_rr/` → crossbar `rr`
- `build_hw/xbar_voq/` → crossbar `voq`
- `build_hw/xbar_voq_rr/` → crossbar `voq_rr`

Each directory will build an `.xclbin` at:

- `xclbin/graphyflow_kernels.hw.xclbin`

---

## 1) Prerequisites on the server

You need:

- Vitis/Vivado installed (hardware target builds require Vivado).
- XRT installed (typically already present with Vitis).
- A valid platform file (`.xpfm`) for your target board (e.g. U55C).
- Enough disk space (parallel hardware builds can consume **tens of GB per build**).

This repo assumes an environment script exists at:

- `/home/feiyang/set_env.sh`

It should set at least:

- `XILINX_VITIS`
- `XILINX_XRT`

Quick check:

```bash
source /home/feiyang/set_env.sh
echo "$XILINX_VITIS"
echo "$XILINX_XRT"
v++ --version
```

If your server uses a different setup script, replace `/home/feiyang/set_env.sh` accordingly.

---

## 2) Clone repo + pick a clean workspace

Hardware builds create large intermediates. Prefer a large scratch path, e.g.:

```bash
export WORK=/scratch/$USER/voq_GraphyFlow
mkdir -p "$WORK"
cd "$WORK"

# clone however you normally do (git/ssh/internal mirror)
```

Then:

```bash
cd voq_GraphyFlow
git rev-parse HEAD
```

Record that commit hash in your run logs / ticket.

---

## 3) Generate the baseline Vitis project (`generated_project/`)

This repo uses `tests/dist.py` to generate a full Vitis project into `generated_project/`.

From repo root:

```bash
PYTHONPATH=$(pwd) python3 tests/dist.py
```

Notes:

- This **deletes** any existing `generated_project/` and recreates it.
- The generated project includes the baseline network (omega) inside `graphyflow_big`.

---

## 4) Create 5 independent build directories

From repo root:

```bash
rm -rf build_hw
mkdir -p build_hw

# copy the generated project five times (baseline + 4 variants)
cp -a generated_project build_hw/baseline_omega
cp -a generated_project build_hw/xbar_plain
cp -a generated_project build_hw/xbar_rr
cp -a generated_project build_hw/xbar_voq
cp -a generated_project build_hw/xbar_voq_rr
```

If `generated_project/` already contains build artifacts (`_x`, `.Xil`, etc.), remove them **before copying** to avoid wasting disk:

```bash
rm -rf generated_project/_x generated_project/.Xil generated_project/.run generated_project/xclbin generated_project/logs
```

---

## 5) Inject each crossbar variant (swap only `graphyflow_big.*`)

Each variant is a drop-in replacement for:

- `scripts/kernel/graphyflow_big.cpp`
- `scripts/kernel/graphyflow_big.h`

From repo root:

```bash
# plain
cp -f xbar_variants/plain/graphyflow_big.cpp build_hw/xbar_plain/scripts/kernel/graphyflow_big.cpp
cp -f xbar_variants/plain/graphyflow_big.h   build_hw/xbar_plain/scripts/kernel/graphyflow_big.h

# rr
cp -f xbar_variants/rr/graphyflow_big.cpp build_hw/xbar_rr/scripts/kernel/graphyflow_big.cpp
cp -f xbar_variants/rr/graphyflow_big.h   build_hw/xbar_rr/scripts/kernel/graphyflow_big.h

# voq
cp -f xbar_variants/voq/graphyflow_big.cpp build_hw/xbar_voq/scripts/kernel/graphyflow_big.cpp
cp -f xbar_variants/voq/graphyflow_big.h   build_hw/xbar_voq/scripts/kernel/graphyflow_big.h

# voq_rr
cp -f xbar_variants/voq_rr/graphyflow_big.cpp build_hw/xbar_voq_rr/scripts/kernel/graphyflow_big.cpp
cp -f xbar_variants/voq_rr/graphyflow_big.h   build_hw/xbar_voq_rr/scripts/kernel/graphyflow_big.h
```

Baseline directory `build_hw/baseline_omega/` is left unchanged.

---

## 6) Ensure the platform (`DEVICE`) matches your server

Each generated project has a `Makefile` that sets a platform path like:

```make
DEVICE := /opt/xilinx/platforms/<...>/<...>.xpfm
```

On the server, verify it exists:

```bash
ls -la /opt/xilinx/platforms || true
find /opt/xilinx/platforms -name "*.xpfm" | head
```

If the platform path is different, edit (or mass-edit) **all 5** Makefiles:

```bash
PLATFORM=/path/to/your/platform.xpfm

for d in build_hw/*; do
  sed -i "s|^DEVICE := .*|DEVICE := ${PLATFORM}|g" "$d/Makefile"
done
```

---

## 7) Compile in parallel (`TARGET=hw`) with per-variant logs

Hardware builds can take hours; always capture logs.

### Option A: simple background jobs

```bash
source /home/feiyang/set_env.sh

ts=$(date +%Y%m%d_%H%M%S)
for d in build_hw/*; do
  name=$(basename "$d")
  (
    set -euo pipefail
    cd "$d"
    mkdir -p logs
    make all TARGET=hw 2>&1 | tee "logs/make_all_hw.${name}.${ts}.log"
  ) &
done
wait
```

### Option B: GNU parallel (recommended if installed)

```bash
source /home/feiyang/set_env.sh
ts=$(date +%Y%m%d_%H%M%S)

ls -d build_hw/* | parallel --jobs 5 --tag '
  set -euo pipefail
  cd {}
  mkdir -p logs
  name=$(basename {})
  make all TARGET=hw 2>&1 | tee "logs/make_all_hw.${name}.'$ts'.log"
'
```

### Resource notes

- Running 5 hardware builds concurrently can overwhelm CPU/RAM/disk IO.
- If the server is constrained, reduce concurrency (`--jobs 2` or run fewer background jobs).

---

## 8) Quick success checks (per directory)

For each directory `build_hw/<variant>/` you should see:

- `xclbin/graphyflow_kernels.hw.xclbin`
- link summary: `xclbin/graphyflow_kernels.hw.xclbin.link_summary`
- a complete log under `logs/`

Example:

```bash
ls -la build_hw/xbar_plain/xclbin/graphyflow_kernels.hw.xclbin
```

---

## 9) Clean up to reclaim disk

In any build directory:

```bash
rm -rf _x .Xil .run xclbin logs
```

If you want to keep only the final `.xclbin`, delete everything except `xclbin/graphyflow_kernels.hw.xclbin` and the build log.

---

## 10) What to report back (minimum)

For each of the 5 builds (baseline + 4 variants), capture:

- repo commit SHA (`git rev-parse HEAD`)
- server hostname / tool versions (`v++ --version`, `xrt-smi examine` if applicable)
- platform `.xpfm` path used
- the build log path (the `logs/make_all_hw.*.log`)
- whether `xclbin/graphyflow_kernels.hw.xclbin` was produced successfully


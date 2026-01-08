# GraphyFlow `hw_emu` Runbook (Agent Notes)

This document records the exact steps and user instructions for building and running hardware emulation for the generated Vitis project under `generated_project/`.

## User Instructions (authoritative)

- Enter `generated_project/`.
- Activate the Vitis/XRT environment via: `bass source /home/feiyang/set_env.sh`.
- Generate a random graph using `gen_random_graph.py` before running.
- Compile hardware emulation with: `make all TARGET=hw_emu`.
- Run hardware emulation with: `./run.sh hw_emu`.
- Always capture compilation logs with `tee` (and keep the build failing if `make` fails).
- Use the **2024** environment (via the provided env script). Do **not** switch to 2022.x toolchains.

## Step-by-step Commands (recommended)

Run from the repo root:

```bash
cd generated_project

# 1) Activate environment (per user instruction)
# This is a fish-shell command (bass is a fish function)
bass source /home/feiyang/set_env.sh

# 2) Prepare log directory
mkdir -p logs

# 3) Generate input graph (writes ./graph.txt)
python3 gen_random_graph.py 64 256

# 4) Build hw_emu (capture logs with tee; keep exit code via pipefail)
(
  set -o pipefail
  make all TARGET=hw_emu 2>&1 | tee "logs/make_all_hw_emu.$(date +%Y%m%d_%H%M%S).log"
)

# 5) Run hw_emu (optional: also log)
(
  set -o pipefail
  ./run.sh hw_emu 2>&1 | tee "logs/run_hw_emu.$(date +%Y%m%d_%H%M%S).log"
)
```

## Notes / Troubleshooting

- If you are not using fish, replace `bass source ...` with `source /home/feiyang/set_env.sh` in a bash shell.
- Platform is set in `generated_project/Makefile` via `DEVICE := ...xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm`. If your machine uses a different platform, override at build time, e.g. `make all TARGET=hw_emu DEVICE=/path/to/platform.xpfm`.
- If you re-run builds frequently, consider `make cleanall` (also with `tee`) before rebuilding:
  `make cleanall 2>&1 | tee "logs/make_cleanall.$(date +%Y%m%d_%H%M%S).log"`.

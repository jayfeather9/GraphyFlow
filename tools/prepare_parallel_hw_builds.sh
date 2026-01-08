#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
prepare_parallel_hw_builds.sh

Creates per-variant build directories for parallel TARGET=hw compilation.

Default layout:
  build_hw/baseline_omega
  build_hw/xbar_plain
  build_hw/xbar_rr
  build_hw/xbar_voq
  build_hw/xbar_voq_rr

Usage:
  ./tools/prepare_parallel_hw_builds.sh
  ./tools/prepare_parallel_hw_builds.sh --build-root /scratch/$USER/build_hw
  ./tools/prepare_parallel_hw_builds.sh --no-generate
  ./tools/prepare_parallel_hw_builds.sh --platform /path/to/platform.xpfm

Options:
  --build-root <dir>   Where to create per-variant project copies (default: ./build_hw)
  --no-generate        Skip running tests/dist.py (assumes ./generated_project exists)
  --platform <xpfm>    Patch DEVICE := ... in each Makefile to this platform path
EOF
}

BUILD_ROOT="build_hw"
DO_GENERATE=1
PLATFORM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --build-root) BUILD_ROOT="$2"; shift 2;;
    --no-generate) DO_GENERATE=0; shift;;
    --platform) PLATFORM="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$DO_GENERATE" -eq 1 ]]; then
  echo "[1/4] Generating baseline project into ./generated_project via tests/dist.py"
  PYTHONPATH="$REPO_ROOT" python3 tests/dist.py
else
  echo "[1/4] Skipping generation (--no-generate)"
fi

if [[ ! -d generated_project/scripts/kernel ]]; then
  echo "ERROR: ./generated_project/scripts/kernel not found. Run without --no-generate." >&2
  exit 1
fi

echo "[2/4] Removing old build root and creating: $BUILD_ROOT"
rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

echo "[3/4] Copying baseline + creating variant directories"
cp -a generated_project "$BUILD_ROOT/baseline_omega"
cp -a generated_project "$BUILD_ROOT/xbar_plain"
cp -a generated_project "$BUILD_ROOT/xbar_rr"
cp -a generated_project "$BUILD_ROOT/xbar_voq"
cp -a generated_project "$BUILD_ROOT/xbar_voq_rr"

echo "[3/4] Injecting crossbar variants (graphyflow_big.* only)"
cp -f xbar_variants/plain/graphyflow_big.cpp "$BUILD_ROOT/xbar_plain/scripts/kernel/graphyflow_big.cpp"
cp -f xbar_variants/plain/graphyflow_big.h   "$BUILD_ROOT/xbar_plain/scripts/kernel/graphyflow_big.h"

cp -f xbar_variants/rr/graphyflow_big.cpp "$BUILD_ROOT/xbar_rr/scripts/kernel/graphyflow_big.cpp"
cp -f xbar_variants/rr/graphyflow_big.h   "$BUILD_ROOT/xbar_rr/scripts/kernel/graphyflow_big.h"

cp -f xbar_variants/voq/graphyflow_big.cpp "$BUILD_ROOT/xbar_voq/scripts/kernel/graphyflow_big.cpp"
cp -f xbar_variants/voq/graphyflow_big.h   "$BUILD_ROOT/xbar_voq/scripts/kernel/graphyflow_big.h"

cp -f xbar_variants/voq_rr/graphyflow_big.cpp "$BUILD_ROOT/xbar_voq_rr/scripts/kernel/graphyflow_big.cpp"
cp -f xbar_variants/voq_rr/graphyflow_big.h   "$BUILD_ROOT/xbar_voq_rr/scripts/kernel/graphyflow_big.h"

if [[ -n "$PLATFORM" ]]; then
  echo "[4/4] Patching platform in Makefiles to: $PLATFORM"
  for d in "$BUILD_ROOT"/*; do
    sed -i "s|^DEVICE := .*|DEVICE := ${PLATFORM}|g" "$d/Makefile"
  done
else
  echo "[4/4] Platform not patched (use --platform if needed)"
fi

cat <<EOF

Done.

Next (example parallel build):
  source /home/feiyang/set_env.sh
  ts=\$(date +%Y%m%d_%H%M%S)
  for d in $BUILD_ROOT/*; do
    name=\$(basename "\$d")
    (cd "\$d" && mkdir -p logs && make all TARGET=hw 2>&1 | tee "logs/make_all_hw.\${name}.\${ts}.log") &
  done
  wait
EOF


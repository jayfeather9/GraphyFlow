# `graphyflow_sssp` (Rust)

Fast Rust runner for the **SSSP/new_dist-style single-pass relaxation** used in DFIR simulation benchmarks.

It performs one relaxation pass:

`new_dist[dst] = min(new_dist[dst], dist[src] + weight)`

with `dist[source]=0` and all other `dist=INF`.

## Build

```bash
cd rust_sim/graphyflow_sssp
cargo build --release
```

## Run

```bash
./target/release/graphyflow_sssp --dataset /path/to/graph.txt --indexing auto --weight unit
```

Arguments:
- `--dataset`: edge list file (2 integers per line, optional 3rd weight column).
- `--indexing`: `auto` (default), `zero-based`, `one-based`.
- `--weight`: `unit` (default) or `third-column-u32`.
- `--source`: source node id after indexing adjustment (default `0`).

Output: a one-line JSON summary including `edges`, `nodes`, `updates`, and timing.


# DFIR Simulator Benchmarks (RMAT Datasets)

These timings measure the Python DFIR simulator (`graphyflow/simulate.py`) running a single-pass SSSP/new_dist-style relaxation:

- **Input**: edge list text file (two integers per line).
- **Indexing**: datasets detected as 1-based and shifted to 0-based during parsing.
- **Algorithm (DFIR)**:
  - `reduce_key`: `dst.id`
  - `reduce_transform`: `(src.distance + 1.0, dst)`
  - `reduce_method`: `min` on the distance field
  - `post-map`: `min(reduced_dist, node.distance)`
- **Initialization**: `distance[0]=0`, all others `INF`.

## Results

All runs below used the simulator optimizations committed in `86524df` (streaming Reduce + contiguous array storage).

| Dataset | Edges | Nodes | Parse time (s) | DFIR `run()` time (s) | Total time (s) |
|---|---:|---:|---:|---:|---:|
| `/data/feiyang/test/test/datasets/rmat-19-32.txt` | 15,483,988 | 524,288 | 11.136 (+1.969 shift) | 93.448 | 106.961 |
| `/data/feiyang/test/test/datasets/rmat-21-32.txt` | 63,541,114 | 2,097,152 | 47.335 | 543.589 | 590.926 |
| `/data/feiyang/test/test/datasets/rmat-24-16.txt` | 263,437,623 | 16,777,214 | 183.884 | 2346.450 | 2530.349 |

Notes:
- For `rmat-19-32.txt`, parsing includes an explicit “index shift” step (`+1.969 s`) in addition to file parsing (`11.136 s`).
- Totals include parsing + simulator setup + DFIR `run()`.


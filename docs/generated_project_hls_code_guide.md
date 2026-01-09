# `generated_project/` HLS C++ Code Guide (host + kernels)

This doc is a **thorough, newcomer-friendly introduction** to the generated Vitis project under `generated_project/`.

After reading it, you should be able to:
- understand how the host partitions the input graph into **little (dense)** and **big (sparse)** work,
- understand how data moves through the **multi-kernel, streaming** design in the xclbin,
- safely modify HLS modules (e.g., the big-kernel routing network / crossbar),
- rebuild and run `sw_emu` / `hw_emu` with logs, and prepare for `hw`.

If you’re trying to do **parallel `TARGET=hw` builds** of baseline + variants, also read:
- `docs/parallel_hw_build_variants.md`

---

## 0) Big picture

The design is a multi-kernel streaming graph accelerator for a Bellman-Ford / SSSP-style “min-plus” relaxation:

```
new_dist[dst] = min( old_dist[dst],  old_dist[src] + weight )
```

Key implementation ideas:
- The host partitions the graph by **destination vertex** (dst-centric).
- “Dense” destination sets go to **little kernels** (11 CUs by default).
- “Sparse” destination sets go to **big kernels** (3 CUs by default).
- Each iteration:
  1) host updates node-distance buffers on the device,
  2) launches all kernels for one iteration,
  3) reads back updated destination distances,
  4) checks convergence, repeats until converged.

The xclbin contains:
- 11× `graphyflow_little` compute units (CUs)
- 3× `graphyflow_big` CUs
- 1× `little_merger` (ap_ctrl_none “always on” stream reduction)
- 1× `big_merger` (ap_ctrl_none “always on” stream reduction)
- 1× `apply_kernel` (applies min(old, update) to a packed dst-distance array)
- 1× `hbm_writer` (serves node-distance reads to kernels and writes apply results to memory)

Instantiation is configured in `generated_project/system.cfg`.

---

## 1) How to build + run (sw_emu / hw_emu)

### 1.1 Environment activation

This repo uses `/home/feiyang/set_env.sh` on the author machine.

Bash:
```bash
source /home/feiyang/set_env.sh
```

Fish:
```fish
bass source /home/feiyang/set_env.sh
```

Sanity check:
```bash
echo "$XILINX_VITIS"
echo "$XILINX_XRT"
v++ --version
```

### 1.2 Generate a graph input

`generated_project/graph.txt` is the default input used by `generated_project/run.sh`.

Generate a random graph (nodes, edges):
```bash
cd generated_project
python3 gen_random_graph.py 64 256
```

### 1.3 Build (recommended with logs)

```bash
cd generated_project
source /home/feiyang/set_env.sh
mkdir -p logs

ts=$(date +%Y%m%d_%H%M%S)
make all TARGET=hw_emu 2>&1 | tee "logs/make_all_hw_emu.${ts}.log"
```

Targets:
- `TARGET=sw_emu` (fast compile, software-level behavior)
- `TARGET=hw_emu` (slow, runs RTL/SoC simulation under xsim)
- `TARGET=hw` (full implementation, for actual FPGA)

### 1.4 Run (hw_emu)

```bash
cd generated_project
source /home/feiyang/set_env.sh
mkdir -p logs

ts=$(date +%Y%m%d_%H%M%S)
./run.sh hw_emu 2>&1 | tee "logs/run_hw_emu.${ts}.log"
```

Expected success line:
- `SUCCESS: Results match!`

Important files:
- `generated_project/run.sh` sets `XCL_EMULATION_MODE`, `LD_PRELOAD`, and picks `xclbin/graphyflow_kernels.${TARGET}.xclbin`.

### 1.5 Clean / rebuild tips

If you change kernel code and want to force recompilation:
```bash
cd generated_project
rm -f xclbin/graphyflow_big.hw_emu.xo xclbin/graphyflow_big.hw_emu.xo.compile_summary
make xclbin/graphyflow_big.hw_emu.xo TARGET=hw_emu
```

If caches get confusing:
```bash
cd generated_project
rm -rf _x .Xil .run xclbin
```

---

## 2) `generated_project/` directory map

Top level:
- `generated_project/Makefile`: main entry (`make all TARGET=...`)
- `generated_project/system.cfg`: kernel instantiation + stream connections + HBM mappings
- `generated_project/run.sh`: run wrapper (sets env vars, calls `graphyflow_host`)
- `generated_project/gen_random_graph.py`: creates `graph.txt`
- `generated_project/graph.txt`: input graph edge list
- `generated_project/xclbin/`: build outputs (`*.xo`, `*.xclbin`)
- `generated_project/scripts/`: sources used to build kernels + host

Kernel sources:
- `generated_project/scripts/kernel/shared_kernel_params.h`: shared typedefs + kernel prototypes
- `generated_project/scripts/kernel/graphyflow_little.cpp` / `.h`: “little” kernel (dense destinations)
- `generated_project/scripts/kernel/graphyflow_big.cpp` / `.h`: “big” kernel (sparse destinations, contains omega/xbar network)
- `generated_project/scripts/kernel/little_merger.cpp`: merges 11 little CU outputs (min reduction)
- `generated_project/scripts/kernel/big_merger.cpp`: merges 3 big CU outputs (min reduction)
- `generated_project/scripts/kernel/apply_kernel.cpp`: applies updates to packed dst-distance memory
- `generated_project/scripts/kernel/hbm_writer.cpp`: serves node-distance reads and writes apply results
- `generated_project/scripts/kernel/kernel.mk`: v++ compile/link rules

Host sources:
- `generated_project/scripts/host/host.cpp`: `main()` and CLI parsing
- `generated_project/scripts/host/fpga_executor.cpp`: per-iteration loop (enqueue kernels, wait, profile prints)
- `generated_project/scripts/host/generated_host.cpp` / `.h`: packing, buffers, enqueue logic, convergence update
- `generated_project/scripts/host/graph_loader.cpp` / `.h`: loads `graph.txt` or `.mtx` into `GraphCSR`
- `generated_project/scripts/host/graph_preprocess/graph_preprocess.cpp` / `.h`: partitions + pipelines the graph
- `generated_project/scripts/host/acc_setup/acc_setup.cpp` / `.h`: OpenCL/XRT setup, kernel handle creation
- `generated_project/scripts/host/host_verifier.cpp` / `.h`: CPU “golden” verifier
- `generated_project/scripts/host/host_bellman_ford.cpp` / `.h`: host reference iteration logic

---

## 3) Host execution flow (what runs, in what order)

### 3.1 `main()` (loading + run + verify)

Entry point: `generated_project/scripts/host/host.cpp`

High-level:
1) load graph from file via `load_graph_from_file()` (`generated_project/scripts/host/graph_loader.cpp`)
2) run FPGA implementation: `run_fpga_kernel(...)` (`generated_project/scripts/host/fpga_executor.cpp`)
3) run CPU verifier: `verify_on_host(...)` (`generated_project/scripts/host/host_verifier.cpp`)
4) compare vectors and print `SUCCESS: Results match!` or mismatches

There is also `--analyze` mode in `generated_project/scripts/host/host.cpp` which prints a model-based estimate and PE load distribution; it is not used for correctness.

### 3.2 `run_fpga_kernel()` (the per-iteration loop)

File: `generated_project/scripts/host/fpga_executor.cpp`

Key steps:
1) `partitionGraph(&graph)` → produces a `PartitionContainer` describing dense vs sparse partitions and their pipeline edge lists.
2) `initAccelerator(xclbin_path)` → programs device and creates OpenCL kernel handles.
3) `AlgorithmHost` orchestrates data packing and kernel runs:
   - `prepare_data(container, start_node)`
   - `setup_buffers(container)`
   - iteration loop:
     - `update_data(container)` (refresh node-distance buffers)
     - `transfer_data_to_fpga(container)` (HBM buffer migrate)
     - `execute_kernel_iteration(container)` (enqueue all iteration kernels)
     - wait on queues, read back results
     - `check_convergence_and_update(container)` (update global distances, decide if done)

The host prints per-kernel OpenCL event times (`CL_PROFILING_COMMAND_START/END`) in each iteration.

---

## 4) Graph input format and CSR representation

### 4.1 Input file formats

Loader: `generated_project/scripts/host/graph_loader.cpp`

- `.txt`: assumed 0-based by default (auto-detects 1-based and converts)
- `.mtx`: assumed 1-based (skips `%` comment lines)

Each edge line:
- `src dst [weight]`
- if `weight` missing, defaults to 1

### 4.2 CSR (`GraphCSR`)

Definition: `generated_project/scripts/host/common.h`

`GraphCSR` stores:
- `offsets[u]..offsets[u+1]` range in `columns`/`weights` for outgoing edges of `u`
- `columns[i]` = destination vertex id
- `weights[i]` = edge weight

Host-side dummy edges:
- partitioner uses dummy destination `0x7FFFFFFF` for padding/alignment
- verifier ignores dummy edges using `(v & 0x40000000) != 0` check

---

## 5) Graph partitioning into little vs big (dense vs sparse)

Partitioner: `generated_project/scripts/host/graph_preprocess/graph_preprocess.cpp`

### 5.1 Partition strategy (dst-centric)

`partitionGraph()`:
1) collects all unique destination vertices and their indegree
2) sorts unique destinations by indegree descending
3) assigns high-indegree destinations primarily to **little** partitions
4) assigns remaining destinations to **big** partitions
5) assigns each edge to the partition responsible for its destination
6) within each partition:
   - compresses vertex IDs to a local ID space (`vtx_map` / `vtx_map_rev`)
   - maps all destination vertices first to low local IDs `[0..num_dsts-1]`
   - distributes edges across pipelines
   - builds per-pipeline CSR

Limits that control how many destinations fit per partition:
- `LITTLE_MAX_DST` and `BIG_MAX_DST` in `generated_project/scripts/host/common.h`
  - `EMULATION`: 512 each
  - `non-EMULATION`: little 65536, big 524288

Pipeline counts:
- `LITTLE_KERNEL_NUM` and `BIG_KERNEL_NUM` in `generated_project/scripts/host/host_config.h`
  - current default: 11 little, 3 big

### 5.2 Pipeline edge distribution

For each partition:
- edges are first rewritten to local IDs and sorted by `src`
- edges are then split approximately evenly into `num_pipelines` slices
- each pipeline has its own `PipelineEdges` CSR:
  - `pipeline_edges[pip].offsets` (size `num_vertices+1`)
  - `pipeline_edges[pip].columns`, `pipeline_edges[pip].weights`

Padding/alignment behavior:
- all pipelines are padded to a multiple of **8 edges** (required by 512-bit edge packing: 8×64-bit edges per bus word)
- little partitions have extra padding at **SRC_BUFFER_SIZE** boundaries to keep the little-kernel property streaming aligned

Why “dense” vs “sparse”:
- “dense” = high-indegree destination sets, optimized with `graphyflow_little`’s ping-pong source property buffering
- “sparse” = lower-indegree destination sets, optimized with `graphyflow_big`’s cacheline-request-based property fetch and per-PE reduction

---

## 6) Device-side top-level interconnect (how kernels talk)

The xclbin kernels are connected using streams specified in:
- `generated_project/system.cfg`

Important parts:
- `[connectivity] nk=...`: how many CUs of each kernel exist
- `sp=...:HBM[...]`: which HBM bank each kernel argument is mapped to
- `stream_connect=...`: stream connections between kernels

High-level stream topology:

```
graphyflow_little_[1..11]
  ppb_req_stream  --->  hbm_writer ppb_req_stream_[1..11]
  ppb_resp_stream <---  hbm_writer ppb_resp_stream_[1..11]
  kernel_out_stream ---> little_merger little_kernel_[1..11]_out_stream

graphyflow_big_[1..3]
  cacheline_req_stream  --->  hbm_writer cacheline_req_stream_[1..3]
  cacheline_resp_stream <---  hbm_writer cacheline_resp_stream_[1..3]
  kernel_out_stream ---> big_merger big_kernel_[1..3]_out_stream

little_merger kernel_out_stream ---> apply_kernel little_kernel_out_stream
big_merger    kernel_out_stream ---> apply_kernel big_kernel_out_stream

apply_kernel kernel_out_stream ---> hbm_writer write_burst_stream
```

Key property:
- `little_merger` and `big_merger` are `ap_ctrl_none` kernels (see `#pragma HLS interface ap_ctrl_none port = return`), so they behave like always-on stream operators in the design.

---

## 7) Host-side packing: how data is laid out in HBM

All device memory transfers are in 512-bit words:
- `AXI_BUS_WIDTH = 512`
- `bus_word_t = ap_uint<512>`

### 7.1 Edge packing (used by both big and little)

Each edge is packed into 64 bits inside a `bus_word_t`:
- lower 32 bits: destination id
- upper 32 bits: source id

Because `512 / 64 = 8`, each `bus_word_t` holds 8 edges.

Both kernels compute:
- `edges_per_word = AXI_BUS_WIDTH / (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH)` = 8
- `num_wide_reads = num_edges / edges_per_word`

So `num_edges` **must** be divisible by 8; the partitioner pads as needed.

### 7.2 Node-distance packing (used by hbm_writer)

Distances are 32-bit `distance_t` values. The host packs them as raw bytes into 512-bit words:
- `512 / 32 = 16` distances per `bus_word_t`

Two different memory “views” are used:

1) **Per-partition node arrays** (used to serve source-distance reads):
   - big partitions: packed `num_vertices` distances
   - little partitions: packed `num_vertices` distances
   - stored in the many “src_prop_*” HBM buffers used by `hbm_writer`

2) **Packed destination array** (used for apply + convergence checking):
   - layout: `[all little dst words][all big dst words]`
   - stored in the `apply_kernel`’s `node_props` argument, and written out by `hbm_writer` to its `output` pointer

The offsets for these packed arrays are tracked in `PartitionBuffer` fields:
- `dense_buffers[i].src_buf_offset` (where this partition’s src buffers begin)
- `sparse_buffers[i].node_prop_offset` (where this partition’s node props begin)
- `dense_buffers[i].dst_prop_offset` / `sparse_buffers[i].dst_prop_offset` (where this partition’s dst props lie inside the apply/output layout)

These are computed in `generated_project/scripts/host/generated_host.cpp` during `prepare_data()`.

---

## 8) Kernel behaviors (what each module does)

### 8.1 `hbm_writer` (memory service + output writeback)

File: `generated_project/scripts/kernel/hbm_writer.cpp`

Responsibilities:
1) For each little CU:
   - reads `ppb_request_pkt_t` requests (which “source buffer round” to load)
   - streams back 4096-node “source buffers” as 512-bit bursts (`ppb_response_pkt_t`)
2) For each big CU:
   - reads cacheline requests (`cacheline_request_pkt_t`): which 512-bit word index to read
   - returns cacheline responses (`cacheline_response_pkt_t`) targeted to the requesting PE
3) Writes final updated packed dst-distance words to `output[]` via `write_burst_w_dst_pkt_t` from `apply_kernel`

Termination:
- the host passes `num_partitions_little` / `num_partitions_big` so loaders know how many end markers to expect.

### 8.2 `graphyflow_little` (dense destinations)

Files:
- `generated_project/scripts/kernel/graphyflow_little.cpp`
- `generated_project/scripts/kernel/graphyflow_little.h`

High-level pipeline (inside `#pragma HLS DATAFLOW`):
1) **Edge load**: reads `edge_props` in 512-bit words → unpacks to `edge_descriptor_batch_t` (8 edges per batch).
2) **Request/response property manager**: `request_manager(...)`
   - maintains ping-pong BRAM buffers holding 4096-node distance chunks for all 8 PEs
   - issues PPB requests for upcoming `SRC_BUFFER_SIZE` “rounds”
   - receives responses from `hbm_writer` and fills ping-pong buffers
   - joins (src distance + weight) with edges to produce `update_tuple_t_little` stream
3) **Reduction**: `Reduc_105_unit_reduce(...)`
   - maintains per-PE URAM arrays of destination reductions (min)
   - updates are keyed by local destination ID
4) **Drain/pack outputs**: partial drains merge PE halves, then output is produced as `little_out_pkt_t` (64-bit) and later packed by `little_merger`

Notes:
- weight is currently treated as a constant `1.0` in-kernel.
- dummy edges are filtered by checking top bit of the 20-bit destination id (`node_id.range(19, 19)`).

### 8.3 `graphyflow_big` (sparse destinations)

Files:
- `generated_project/scripts/kernel/graphyflow_big.cpp`
- `generated_project/scripts/kernel/graphyflow_big.h`

High-level pipeline (inside `#pragma HLS DATAFLOW`):
1) **Edge load**: reads `edge_props` in 512-bit words → creates:
   - `edge_stream` (edge batches)
   - `stream_src_ids` (source ID bursts)
2) **COO-style source property fetch pipeline**:
   - `dist_req_packer(...)`: converts source IDs into “cacheline indices” and detects when a new cacheline is needed
   - `cacheline_req_sender(...)`: emits cacheline read requests for the PEs that need them
   - `stream2axistream(...)` / `axistream2stream(...)`: bridge internal structs to AXI stream packets for inter-kernel streaming
   - `node_prop_resp_receiver(...)`: routes returned cachelines to per-PE cacheline streams
   - `merge_node_props(...)`: joins edge batches with source distances, emits `update_tuple_t_big`
3) **Demux into lanes**: `demux_1(...)` converts batched updates into 8 independent input streams.
4) **Routing network (omega or crossbar)**:
   - historically: 12× `switch2x2_2(...)` stages (omega-like network)
   - crossbar variants: `crossbar_*` functions (see `xbar_variants/`)
   - destination port selection is by `node_id.range(LOG_PE_NUM-1,0)` (low 3 bits)
5) **Per-PE reduction**:
   - `Reduc_105_unit_reduce_single_pe(...)` maintains a URAM array for that PE’s slice of destination-word space and performs min-reduction
   - drains pack 8 PE outputs into 512-bit bursts (`write_burst_pkt_t`)

This is the primary file to modify for “routing network” experiments.

### 8.4 `little_merger` (merge across 11 little pipelines)

File: `generated_project/scripts/kernel/little_merger.cpp`

- Reads one 64-bit `little_out_pkt_t` from each little CU per step (non-blocking reads until all are available).
- Computes elementwise min across the 11 values.
- Packs 8× 64-bit results into one 512-bit `write_burst_pkt_t` and outputs to `apply_kernel`.

### 8.5 `big_merger` (merge across 3 big pipelines)

File: `generated_project/scripts/kernel/big_merger.cpp`

- Reads one 512-bit `write_burst_pkt_t` from each big CU per step.
- Computes elementwise min across the 3 packets (16 lanes × 32-bit each).
- Outputs merged 512-bit words to `apply_kernel`.

### 8.6 `apply_kernel` (apply min(old, update) + generate write bursts)

File: `generated_project/scripts/kernel/apply_kernel.cpp`

Inputs:
- merged little output stream
- merged big output stream
- a packed `node_props` memory array holding current destination distances

Behavior:
1) merges little/big streams into a unified stream with explicit `dest_addr`:
   - `merge_big_little_writes(...)`
2) for each packet:
   - reads old word from `node_props[dest_addr]`
   - computes `min(old, update)` per 32-bit lane
   - outputs `write_burst_w_dst_pkt_t {data, dest_addr}` to `hbm_writer`

The host uses `little_kernel_length`, `big_kernel_length`, and offsets to define how many destination words belong to each set and where they live in the packed output layout.

---

## 9) Where to change things (common workflows)

### 9.1 Modify the big-kernel routing network

File to edit in the generated project:
- `generated_project/scripts/kernel/graphyflow_big.cpp`

Search for the “Replace omega network…” block near the end of the top function:
- it sits between `demux_1(...)` and `Reduc_105_unit_reduce_single_pe(...)`.

For crossbar variants, prefer editing and swapping from:
- `xbar_variants/plain/graphyflow_big.cpp`
- `xbar_variants/rr/graphyflow_big.cpp`
- `xbar_variants/voq/graphyflow_big.cpp`
- `xbar_variants/voq_rr/graphyflow_big.cpp`

Then copy into the generated project before building.

### 9.2 Change how partitioning splits work

Partitioner:
- `generated_project/scripts/host/graph_preprocess/graph_preprocess.cpp`

Controls:
- `LITTLE_KERNEL_NUM`, `BIG_KERNEL_NUM`: `generated_project/scripts/host/host_config.h`
- destination caps: `LITTLE_MAX_DST`, `BIG_MAX_DST`: `generated_project/scripts/host/common.h`

If you change CU counts, you must also update:
- `generated_project/system.cfg` `nk=` counts
- and the `stream_connect=` list for the correct number of streams

### 9.3 Warning: `generated_project/` is generated output

`generated_project/` is produced by `tests/dist.py` from template files.

If you regenerate it (by re-running `PYTHONPATH=$(pwd) python3 tests/dist.py`), it will overwrite:
- `generated_project/scripts/*`
- top-level build scripts and configs

For persistent changes, consider:
- editing the template under `tmp_work/` / `graphyflow/project_template/` (if that’s how your generation flow is set up), or
- keeping experimental kernel changes in `xbar_variants/` and copying them into build dirs.

---

## 10) Practical debugging notes

- Hardware emulation is slow; start with small graphs (`64/256`).
- Many loops assume “multiple of 8 edges” alignment; if you change packing, update padding logic in the partitioner and host packer.
- Stream deadlocks usually come from missing end tokens or mismatched expected counts (especially in request/response services).
- For stream topology, always consult `generated_project/system.cfg` (it is the ground truth for CU connectivity).


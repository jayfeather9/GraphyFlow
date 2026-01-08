# Network Variants (Dataflow) — Omega vs Crossbar (5 variants)

This repo’s simulator models a **cycle-accurate**, **II=1**, hardware-style dataflow network for `N` parallel lanes (`N` must be a power of two, e.g. `N=8`). The network only routes on integer `dst` fields; payload is ignored.

This document summarizes the **5 supported network variants** when run in the common “dataflow pipeline” shape:

```
per-lane sources  ──►  (lockstep) demux/packer  ──►  network  ──►  per-lane sinks
```

The 5 variants are:

1) Omega network (multi-layer 2×2 switches)
2) Crossbar (skid buffer + fixed priority arbitration)
3) Crossbar (skid buffer + per-output round-robin arbitration)
4) Crossbar (VOQ buffer + fixed priority arbitration)
5) Crossbar (VOQ buffer + per-output round-robin arbitration)

All variants share the same:
- `N` lanes and `layers = log2(N)` routing bits
- per-lane bounded FIFOs (“streams”) between stages
- cycle-by-cycle handshake/backpressure rules
- optional **front “demux packer”** that injects in lockstep across all lanes

The intent is to mimic the semantics of the HLS code style (e.g. `#pragma HLS PIPELINE II=1` and `#pragma HLS DATAFLOW`) closely enough to reason about congestion, stalls, and utilization.

---

## 0) Common concepts and simulator rules

### 0.1 Tokens (`dst` only, plus `end`)

Each token is:
- `dst`: integer destination key used for routing
- `end`: boolean end-of-stream marker

Only `dst` affects routing decisions. All networks must eventually terminate when all inputs have ended and all internal buffering has drained (details differ by topology; see “End behavior” below).

### 0.2 Streams = bounded FIFOs with 1R/1W per cycle

Every connection between blocks is modeled as a bounded FIFO (“stream”) with:
- `depth`: maximum occupancy
- at most **one read and one write** scheduled per stream per cycle
- backpressure: if a FIFO is full, the producer cannot write (unless a read is also scheduled that cycle; see below)

#### “Read+Write in same cycle” allowance (hardware-style bypass)

When a FIFO is currently full (occupancy == depth), a write is still allowed **if** a read from that same FIFO is scheduled in the same cycle. This models the common hardware case where a FIFO can accept a write in the same cycle it performs a read (net occupancy unchanged).

This is a key detail for realistic II=1 dataflow: it avoids artificial stalls when a consumer reads from a full FIFO and a producer writes into it in the same cycle.

### 0.3 Routing bits: `bit_mask` + `bit_order` => `layers` bits

The network does not use “all bits of dst”. Instead, it extracts exactly `layers` bits from `dst` using:

- `bit_mask`: a bitmask selecting which positions of `dst` participate
  - Example for `N=8` (`layers=3`): `bit_mask=0x7` means use bits `[2:0]`.
- `bit_order`: order in which the selected positions are interpreted as the routing “bit index”
  - `lsb_to_msb`: selected positions are used in ascending order (e.g. 0,1,2)
  - `msb_to_lsb`: selected positions are used in descending order (e.g. 2,1,0)

The extracted `layers` bits form an `out_port` integer in `[0, N-1]`.

### 0.4 Demux/packer in front (“packed inputs”)

When enabled (`packed_inputs: true`), input injection is done by a **lockstep packer**:

- It conceptually maintains `N` “lanes” of source tokens (either memory-provided lists or port files).
- Each cycle it forms a *pack* of length `N` containing the next token for each lane:
  - if a lane has no more tokens: lane contributes a `Bubble` (no write)
  - otherwise lane contributes `Item(dst,end)`
- It then tries to **commit the entire pack atomically** into the `N` network input streams:
  - if **any** lane in the pack needs to write an `Item` but its target input FIFO is full, then **none** of the lanes write that cycle (whole pack stalls and is retried next cycle)
  - if all required writes can proceed, then all `Item` lanes write in the same cycle

This implements the user requirement:
> “each cycle accepts a pack of 8 datas … if one of the 8 inputs stalls all 8 should stall”

Important subtleties:
- Lanes that are already finished contribute bubbles and do not block pack commit.
- `end` tokens are treated like regular items for injection (they occupy FIFO space and are written into the network), but they change “termination state” downstream.

### 0.5 Dataflow evaluation order (per cycle)

Conceptually, all blocks are “running in parallel” (DATAFLOW). In the simulator, each cycle is:

1. **Schedule** all reads/writes that blocks *want* to do this cycle, based on current FIFO states
2. **Commit** those reads/writes simultaneously at the end of the cycle

Scheduling is performed in a fixed order (sinks, then network, then sources/packer), but because commits are simultaneous and because “full-but-read-same-cycle” writes are allowed, this order is designed to approximate a realistic handshake-driven dataflow.

### 0.6 End behavior (termination rules)

There are two distinct end semantics in this repo:

#### Omega network end behavior
- `end` tokens are explicitly propagated through the omega switches and are observed by the final sinks.
- Termination condition: all sinks have received `end`, all sources ended, and all internal FIFOs drained.

#### Crossbar end behavior (“mode 3” in code)
- Each input lane eventually injects an `end` token.
- The crossbar **consumes** `end` internally and uses it to mark the corresponding input as “closed”.
- The crossbar does **not** forward `end` tokens to output FIFOs; sinks therefore never see `end`.
- Termination condition: all inputs closed, internal buffers empty, and all FIFOs drained.

This matches the idea that a crossbar can treat `end` as control-plane “close input” rather than data-plane traffic.

---

## 1) Variant A — Omega network (multi-layer 2×2 switches)

### 1.1 Topology (for N=8: 3 layers)

Omega is a multi-stage interconnection network built from 2×2 switches.

- `N` lanes
- `layers = log2(N)`
- Each layer has `N/2` independent 2×2 switches
- Between layers, wires are permuted by the omega shuffle permutation (“perfect shuffle”)

For `N=8`, there are `3` layers and `4` switches per layer.

### 1.2 Each switch is modeled as HLS `switch2x2` (sender + receiver + local streams)

Each 2×2 switch implements exactly the HLS structure you provided:

```
in1 ─┐
     ├─► sender(i) ─► l1_1/l1_2/l1_3/l1_4 ─► receiver(i) ─► out1/out2
in2 ─┘
```

Internal structure:
- `sender(i)`:
  - reads up to 1 item from `in1` and up to 1 item from `in2` per cycle (II=1)
  - for each non-end item, selects which internal FIFO it goes to based on a single routing bit (the `i`-th selected routing bit)
  - when both input ends have been seen, it broadcasts end tokens into *all four* internal FIFOs (matching the HLS code)
- `receiver(i)`:
  - produces up to 1 item for `out1` and up to 1 item for `out2` per cycle (II=1)
  - for `out1`, it tries internal FIFO `l1_1` first, else `l1_3` (priority selection)
  - for `out2`, it tries `l1_2` first, else `l1_4`
  - it waits until it has observed end on all four internal FIFOs, then emits end on both outputs

### 1.3 Internal and external streams (depths)

Omega uses **two kinds of FIFO depth**:

- `local_depth`: depth of the 4 internal FIFOs inside each 2×2 switch (`l1_1..l1_4`)
- `global_depth`: depth of inter-switch FIFOs (and, by default, the output FIFOs)

Additionally, when packed inputs are enabled, the demux/packer writes into the network input FIFOs whose depth is `input_depth` (defaults to `global_depth` unless overridden).

So for each 2×2 switch:
- 2 input FIFOs: produced by previous stage (or demux), consumed by sender
- 4 local FIFOs: produced by sender, consumed by receiver, depth=`local_depth`
- 2 output FIFOs: produced by receiver, consumed by next stage (or sinks), depth=`global_depth`

### 1.4 Per-cycle behavior (II=1 semantics)

Within a cycle:
- Sender may read from each input if:
  - input FIFO not empty, and
  - the chosen internal FIFO has available space (or will be read this cycle)
- Receiver may read from each internal FIFO according to priority rules if:
  - the chosen internal FIFO has data, and
  - the corresponding output FIFO has space (or will be read this cycle)

Because the sender and receiver are separated by internal FIFOs, sender and receiver can both be active every cycle as long as the internal FIFOs are not creating backpressure.

### 1.5 Key characteristics (what tends to dominate performance)

- **Fixed pathing / limited adaptivity**: routing decisions are bit-by-bit per layer. Congestion patterns depend strongly on the distribution of selected `dst` bits.
- **Localized contention**: only pairs of lanes contend at each switch, but contention can amplify across layers.
- **Head-of-line blocking**: within each input FIFO, items are in-order; if the head item needs a busy path, later items behind it cannot bypass.
- **Internal buffering matters**: the 4 internal FIFOs per switch are crucial; too shallow `local_depth` can stall sender/receiver quickly, while larger depths can absorb burstiness.

---

## 2) Variant B — Crossbar (skid buffers + fixed priority arbitration)

This variant is an `N×N` crossbar-like switch with:
- per-input **skid buffer** (1 item)
- per-output **fixed priority** selection among inputs

### 2.1 Topology

```
in_stream[0..N-1]  ─►  (buffers + arbitration)  ─►  out_stream[0..N-1]
```

There are no internal FIFOs between “sender and receiver” because this is not built out of 2×2 switches. The only buffering is the explicit skid buffers plus the external FIFO depths on the input/output streams.

### 2.2 Skid buffer behavior (per input)

Per input `i`:
- A skid buffer stores **at most one** pending item (`input_buf[i]`) and a valid bit (`input_valid[i]`).
- If `input_valid[i]` is false and the input FIFO has data, the crossbar tries to read one item into the skid buffer (load phase).
- If it reads an `end` item:
  - it marks the input as “closed”
  - it does not place it into the skid buffer
  - it is not forwarded to any output

This corresponds to the HLS “input buffer” and “input_valid” arrays.

### 2.3 Arbitration (fixed priority per output)

Each output `out_id` decides at most one “winner input” per cycle:
- It scans inputs in a deterministic order (0,1,2,…,N-1), and selects the first input `check` such that:
  - `input_valid[check] == true`
  - `route(dst_of_input_buf[check]) == out_id`

Because `dst` maps to exactly one output, a given buffered item is eligible for only one output.

### 2.4 Execution (write phase)

For each output `out_id` that has a winner:
- If the output FIFO is full (and will not be read this cycle), the output stalls and the buffered item remains in the skid buffer.
- Otherwise, it writes the item to the output FIFO and clears `input_valid[winner]` so that input can load a new item in later cycles.

### 2.5 Key characteristics

- **Very small internal buffering**: 1-item skid per input.
- **Head-of-line blocking is severe**: each input can hold only one item, and if its output is congested, that input cannot accept any new traffic.
- **Deterministic unfairness**: fixed priority means low-numbered inputs can dominate if there is persistent contention for the same output.

---

## 3) Variant C — Crossbar (skid buffers + per-output round-robin arbitration)

This variant is identical to Variant B except for the arbitration policy.

### 3.1 What “per-output round-robin” means

Instead of always checking input 0 first, each output `out_id` maintains its own pointer:
- `rr_ptr[out_id]` in `[0, N-1]`

Each cycle, output `out_id` scans inputs in this order:

```
rr_ptr[out_id], rr_ptr[out_id]+1, ..., N-1, 0, 1, ..., rr_ptr[out_id]-1
```

and picks the first eligible input.

If it successfully grants `src_idx` for that output *and* the output FIFO write succeeds, it advances:

```
rr_ptr[out_id] = (src_idx + 1) mod N
```

So each output independently rotates which input gets priority next time.

### 3.2 Why “per-output” matters (vs a single global RR)

There is not one global arbiter; there are `N` arbiters (one per output).

This matters because:
- fairness is tracked independently per output
- contention for output 3 does not affect who gets chosen for output 4

### 3.3 Key characteristics

- Same buffering limits as skid+fixed.
- Usually improves fairness and reduces starvation under persistent contention.
- Still suffers from head-of-line blocking because each input holds only one pending item.

---

## 4) Variant D — Crossbar (VOQ buffers + fixed priority arbitration)

This variant replaces per-input skid buffers with **VOQ (Virtual Output Queues)**.

### 4.1 VOQ structure

Instead of one pending slot per input, each input `i` has `N` per-output queues:

```
voq[i][0], voq[i][1], ..., voq[i][N-1]
```

Meaning:
- If an item arriving on input `i` is destined to output `j`, it goes into `voq[i][j]`.

Each VOQ queue has a bounded capacity:
- `voq_depth = local_depth`

So total buffering capacity scales as `N×N×local_depth` items (conceptually).

### 4.2 VOQ load phase (input side decoupling)

For each input `i`, in the load phase:
- If the input FIFO has data, peek at the next item.
- If it is `end`, consume it and mark input `i` as closed.
- Otherwise:
  - compute `out = route(dst)`
  - check whether `voq[i][out]` has space
    - if full, do **not** read the input FIFO this cycle (input stalls)
    - if not full, read 1 item and push it into `voq[i][out]`

This decouples input loading from output congestion **as long as there is space in the VOQ bucket** for that destination.

### 4.3 Why VOQ helps (classic head-of-line blocking fix)

In a skid-buffer crossbar:
- input `i` holds 1 item; if that item targets a congested output, input `i` cannot accept any new items even if they would go to a free output.

With VOQ:
- input `i` can still accept new items for other outputs even while one output is congested, because they go into different buckets.

VOQ is the standard mechanism to reduce head-of-line blocking in input-buffered crossbars.

### 4.4 Arbitration with fixed priority (and the “input-used” constraint)

Outputs still choose which input to serve each cycle, but now they choose among inputs that have a non-empty `voq[input][out_id]`.

Implementation details that matter:

1) **Candidate list per output**
   - For each output `out_id`, build a list of candidate inputs `input` such that `voq[input][out_id]` is non-empty.
   - In fixed mode, “priority order” is input 0..N-1 (deterministic).

2) **One item per input per cycle**
   - Even though an input may have items queued for many outputs, real hardware typically limits each input to launching **one** item per cycle (because each input has one physical data path into the crossbar).
   - The simulator enforces this with an `input_used[input]` boolean:
     - once an input is selected for some output in this cycle, it cannot be selected by another output in the same cycle

3) **Greedy matching order**
   - Outputs are processed in increasing order `out_id = 0..N-1`.
   - For each output, the simulator selects the first candidate input that is not already used.
   - This means output 0 gets first “choice” of inputs, then output 1, etc.

This “greedy matching + input-used” is a practical approximation of the “each input can be matched to at most one output per cycle” constraint without solving a full maximum matching every cycle.

### 4.5 Key characteristics

- Much larger buffering than skid.
- Better absorption of burstiness and reduced head-of-line blocking.
- Fixed priority can still create starvation patterns, but VOQ often reduces how catastrophic they are.
- Greedy matching order can bias earlier outputs (0 before 7).

---

## 5) Variant E — Crossbar (VOQ buffers + per-output round-robin arbitration)

This combines VOQ buffering (Variant D) with per-output round-robin ordering (Variant C concept).

### 5.1 Per-output RR pointer with VOQ

Each output `out_id` maintains `rr_ptr[out_id]`.

When building the candidate list for output `out_id`, inputs are enumerated starting at:

```
start = rr_ptr[out_id] mod N
```

so the candidate list is in round-robin order rather than always starting at input 0.

After a successful grant to `src_idx` (and a successful write to the output FIFO), it updates:

```
rr_ptr[out_id] = (src_idx + 1) mod N
```

### 5.2 Interactions with the “input-used” constraint

Even with per-output RR:
- an input can still be used at most once per cycle
- outputs are still processed sequentially (0..N-1)

This means:
- fairness improves relative to fixed priority, because each output rotates which input gets checked first
- but it is not a perfect “global fairness” mechanism because the greedy matching order and input-used constraint can still bias outcomes

### 5.3 Key characteristics

- Best fairness among the crossbar variants in this repo.
- Highest control complexity (maintaining `N` pointers and scanning candidates per output).
- If traffic is highly skewed to a few outputs, RR helps prevent one input from permanently dominating those outputs.

---

## 6) Side-by-side summary (what differs across the 5 variants)

### 6.1 Buffering model

- Omega:
  - local FIFOs inside each 2×2 switch (`4×local_depth` per switch)
  - inter-switch FIFOs (`global_depth`)
  - inherent multi-layer structure spreads contention
- Crossbar skid:
  - 1-item register per input (skid buffer)
  - no per-destination buffering at the input
- Crossbar VOQ:
  - `N×N` queues, each depth=`local_depth`
  - input can accept items for uncongested outputs even if one output is congested

### 6.2 Arbitration/scheduling

- Omega:
  - fixed routing bit per stage (no arbitration beyond 2×2 local choices)
  - receiver has a fixed priority between its two candidate internal FIFOs per output
- Crossbar fixed:
  - each output scans inputs 0..N-1
  - deterministic priority
- Crossbar per-output RR:
  - each output has its own rotating start pointer
  - fairness is tracked independently per output

### 6.3 End semantics

- Omega:
  - end tokens propagate through switches and reach sinks
- Crossbar:
  - end tokens are consumed internally and only used to close inputs

### 6.4 Demux/packer interactions

With the demux/packer enabled:
- injection into the network input FIFOs occurs in lockstep “packs”
- a single full input FIFO can stall *all* lanes’ injection in that cycle
- this can change effective burstiness and can sometimes improve or harm utilization depending on how input imbalance interacts with network backpressure

# GraphyFlow Backend Documentation: Kernel Generation

## 1. Introduction

The kernel generation portion of the backend is responsible for the most critical task in the GraphyFlow framework: translating the abstract DFG-IR into high-performance, synthesizable HLS C++ code that will run on the FPGA. This process is orchestrated by the `BackendManager` class, the core of the compiler, located in `graphyflow/backend_manager.py` [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].

## 2. The `BackendManager` and Its Compilation Phases

The `BackendManager` is a stateful class that takes a `ComponentCollection` (the DFG-IR) as input and manages a multi-phase process to produce the final kernel source files (`.h` and `.cpp`). The main entry point is its `generate_backend` method.

### Phase 1: Type Analysis (`_analyze_and_map_types`)

This initial phase traverses the entire DFG-IR graph to understand and map all data types.

* **Type Mapping**: It maps abstract `DfirType`s (e.g., `FloatType`, `SpecialType("node")`) from the IR to concrete HLS C++ types (`HLSType`), which are defined in `graphyflow/backend_defines.py` [cite: CCFSys2025_GraphyFlow/graphyflow/backend_defines.py]. For instance, `FloatType` is mapped to `ap_fixed<32, 16>` for computation and a `int32_t` Plain Old Data (POD) type for memory interfaces to ensure correct bit-level representation [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].
* **Struct Generation**: It dynamically generates C++ `struct` definitions for all tuple types and graph element types (nodes/edges) discovered in the IR [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].
* **Batching**: This is a key concept for parallelism. To process multiple data elements simultaneously, the backend wraps the base data type into a "batch" struct. This batch struct contains an array of `PE_NUM` (Processing Element Number) data elements, along with control flags like `end_flag` and `end_pos`. Data flows between hardware components in these batches [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].

### Phase 2: Function & Stream Definition (`_define_functions_and_streams`)

Once types are understood, this phase outlines the hardware architecture.

* **Function Scaffolding**: It creates an `HLSFunction` object for each `Component` in the DFG-IR. An `HLSFunction` is a Python representation of a future C++ function [cite: CCFSys2025_GraphyFlow/graphyflow/backend_defines.py].
* **Signature Definition**: It defines the function signature for each `HLSFunction`. In a dataflow design, parameters are typically `hls::stream` references that pass the batched data from one component to the next [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].
* **Stream Instantiation**: It identifies all the connections between components in the DFG-IR and declares the necessary intermediate `hls::stream` objects that will connect these functions in the top-level dataflow region [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].

### Phase 3: Code Body Translation (`_translate_functions`)

This is where the actual C++ logic for each component is generated. The manager iterates through each `HLSFunction` and, based on its corresponding `Component` type, generates the C++ code for its body.

* **Standard Streaming Boilerplate**: Most components are translated into a function with a standard `while(true)` loop that continuously reads from input streams, processes one batch of data, writes to output streams, and breaks when an `end_flag` is detected [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].
* **Component-Specific Logic**: Inside this loop, specific logic is generated. For example, a `BinOpComponent` with the `ADD` operation is translated into a `for` loop (unrolled with `#pragma HLS UNROLL`) that iterates `PE_NUM` times, performing an addition on each element of the input batches and storing the result in the output batch [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].

### Phase 4: AXI Wrapper and Top-Level Generation

The final phase constructs the top-level kernel file that Vitis will compile.

* **Data Movers**: Helper functions (`mem_to_stream_func`, `stream_to_mem_func`) are generated to handle the movement of data between the off-chip global memory (accessed via AXI) and the on-chip streams used by the processing pipeline [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].
* **Dataflow Core**: A function (`dataflow_core_func`) is created to encapsulate the main processing pipeline. Inside this function, all the component functions and intermediate streams from Phase 2 and 3 are instantiated and connected [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].
* **Top-Level Kernel (`_generate_axi_kernel_wrapper`)**: The final, top-level function is generated with `extern "C"`. This function signature includes pointers for AXI memory access. Inside, it calls the data mover and dataflow core functions and is marked with `#pragma HLS DATAFLOW` to instruct the HLS compiler to build a streaming pipeline [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].

## 3. Special Handling for `ReduceComponent`

The `ReduceComponent` is the most complex component and is handled specially. It is decomposed into a super-block of multiple functions and streams rather than a single function [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].

1.  **`pre_process` Function**: This function inlines the user's `reduce_key` and `reduce_transform` lambda logic. It reads the main input stream and produces two output streams: one for keys and one for transformed values.
2.  **Interconnection Network**: A series of utility functions (Stream Zipper, Demux, and an Omega Network) are instantiated. These work together to shuffle and route the transformed data to the correct processing element based on its key. This hardware network essentially performs a parallel sort/group-by operation.
3.  **`unit_reduce` Function**: This is the stateful core of the reduction. It contains on-chip memories (BRAMs/URAMs) to store the aggregated value for each key. It reads shuffled data from the interconnection network, performs the user's `reduce_method` logic (which is inlined), and updates the appropriate memory location [cite: CCFSys2025_GraphyFlow/graphyflow/backend_manager.py].

## 4. Utility Hardware Generators (`backend_utils.py`)

To promote modularity, the `BackendManager` does not contain hardcoded logic for complex, reusable hardware structures. Instead, it calls generator functions from `graphyflow/backend_utils.py` [cite: CCFSys2025_GraphyFlow/graphyflow/backend_utils.py]. This file contains functions that return fully formed `HLSFunction` objects for hardware modules like:

* **Omega Network (`generate_omega_network`)**: A multi-stage interconnection network used for efficient, conflict-free parallel data shuffling. It's built from smaller 2x2 switch elements and is fundamental to the performance of the `reduce_by` operation [cite: CCFSys2025_GraphyFlow/graphyflow/backend_utils.py].
* **Stream Demux/Zipper**: Utility functions to split a batched stream into multiple parallel streams (`generate_demux`) or combine multiple streams into one (`generate_stream_zipper`) [cite: CCFSys2025_GraphyFlow/graphyflow/backend_utils.py].

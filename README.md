# GraphyFlow: A High-Level Synthesis Framework for Graph Computing on FPGAs

## 1\. Overview

GraphyFlow is a Python-based framework designed to simplify the development of graph algorithms for hardware acceleration on FPGAs. It provides a high-level, functional API for developers to express complex graph computations. The framework then automatically translates this high-level description into a Dataflow-Graph Intermediate Representation (DFG-IR), which is subsequently compiled into a complete, runnable Vitis HLS project, including HLS C++ for the kernel and C++ for the host application.

The primary goal of GraphyFlow is to abstract away the complexities of HLS and FPGA project management, allowing domain experts to focus on the algorithm's logic while still leveraging the performance of hardware acceleration.

## 2\. Core Concepts & Architecture

GraphyFlow employs a multi-stage compilation architecture to transform a high-level Python definition into a low-level hardware implementation.

```mermaid
graph TD
    A[<b>Step 1: Algorithm Definition</b><br/>User writes a graph algorithm in Python using the GraphyFlow API<br/><i>(e.g., tests/new_dist.py)</i>] --> B{<b>Step 2: Frontend Compilation</b><br/>The Python API builds a high-level graph representation<br/><i>(graphyflow/global_graph.py)</i>};
    B --> C[<b>Step 3: DFG-IR Generation</b><br/>The graph is converted into a<br/>Dataflow-Graph Intermediate Representation<br/><i>(graphyflow/dataflow_ir.py)</i>];
    C --> D{<b>Step 4: Backend Code Generation</b><br/>The Backend Manager traverses the DFG-IR to generate<br/>HLS C++, Host C++, and build scripts<br/><i>(graphyflow/backend_manager.py)</i>};
    D --> E[<b>Step 5: Project Assembly</b><br/>All generated and static template files<br/>are assembled into a complete Vitis project<br/><i>(graphyflow/project_generator.py)</i>];
```

## 3\. Prerequisites

To use GraphyFlow and build the generated projects, you will need the following software installed and configured on your system:

  * **Python 3.x**: For running the GraphyFlow framework itself.
  * **Xilinx Vitis**: The core toolchain for HLS and building the FPGA binaries. The project files seem to be configured for **Vitis 2022.2**, so using this version is recommended for compatibility.
  * **Xilinx Runtime (XRT)**: Required for communication between the host and the FPGA. This is typically installed with Vitis.
  * **Environment Variables**: Ensure that the Vitis and XRT environment setup scripts have been sourced (e.g., `source /opt/Xilinx/Vitis/2022.2/settings64.sh` and `source /opt/xilinx/xrt/setup.sh`).

## 4\. Directory Structure

The GraphyFlow repository is organized as follows:

```
GraphyFlow/
├── graphyflow/              # Core framework source code
│   ├── backend_manager.py     # The main backend compiler
│   ├── dataflow_ir.py         # Defines the Intermediate Representation
│   ├── global_graph.py        # The user-facing Python API for algorithm definition
│   ├── project_generator.py   # Assembles the final Vitis project
│   └── project_template/      # Static template files for a Vitis project
├── tests/                   # Example scripts demonstrating how to use GraphyFlow
│   ├── new_dist.py            # The primary example for generating a project
│   └── ...
└── README.md                # This README file
```

## 5\. How to Run the Example

The following steps guide you through generating, building, and running the provided distance computation project.

### Step 1: Generate the Vitis Project

The framework uses the pre-defined algorithm in `tests/new_dist.py` to generate a complete Vitis project. This script begins by defining the data structure of the graph's nodes and edges for the compiler:

```python
# tests/new_dist.py (Snippet)
# --- Define graph properties ---
g = GlobalGraph(
    properties={
        "node": {"distance": dfir.FloatType()},
        "edge": {"weight": dfir.FloatType()},
    }
)
```

To generate the project, execute the following command from the root of the `GraphyFlow` directory:

```bash
PYTHONPATH=$(pwd) python3 tests/new_dist.py
```

This command temporarily adds the `GraphyFlow` project root to your Python path, allowing the script to import the necessary framework modules. After it completes, a new directory `generated_project/` will be created.

### Step 2: Build the FPGA Kernel and Host Executable

Navigate into the generated project directory and use the provided `Makefile` to build the project. You must specify a target platform.

```bash
cd generated_project/
make check TARGET=sw_emu
```

  * The `make check` command is a convenient wrapper that first builds everything (`all`) and then runs the simulation (`run.sh`).
  * `TARGET` can be one of the following:
      * `sw_emu`: Software emulation (fastest build).
      * `hw_emu`: Hardware emulation (more accurate, slower build).
      * `hw`: Full hardware synthesis (very slow, for running on the actual FPGA).

### Step 3: Run the Project

If you built the project using `make all` instead of `make check`, you can run the software emulation manually:

```bash
./run.sh sw_emu
```

The script will execute the host program, which loads the generated FPGA binary (`.xclbin`), prepares a sample graph, runs the computation, verifies the result against a CPU implementation, and prints a success or failure message.

## 6\. Algorithm Definition API

You can define graph algorithms by chaining together a series of high-level dataflow operators.

  * `n.map_(map_func)`

      * **Description**: Applies a lambda function `map_func` to each element of the input data array `n`.
      * **Lambda Input**: A single element from the array `n`.
      * **Lambda Output**: The transformed element.

  * `n.filter(filter_func)`

      * **Description**: Filters the input array `n`, keeping only the elements for which `filter_func` returns `True`.
      * **Lambda Input**: A single element from the array `n`.
      * **Lambda Output**: A boolean value.

  * `n.reduce_by(reduce_key, reduce_transform, reduce_method)`

      * **Description**: A powerful operator that performs a grouped reduction.
      * **`reduce_key`**: A lambda that computes a key for each element. Elements with the same key are grouped together.
      * **`reduce_transform`**: A lambda that transforms each element within a group before reduction.
      * **`reduce_method`**: A lambda that takes two transformed elements and combines them into one. This function must be commutative and associative (e.g., `add`, `min`, `max`). The final output is an array containing one reduced value for each group.

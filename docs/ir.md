# GraphyFlow Intermediate Representation (IR) Documentation

## 1. Introduction

The GraphyFlow Intermediate Representation, referred to as DFG-IR (Dataflow-Graph IR), serves as the "middle-end" of the compiler. It is a structured, language-agnostic representation of the graph algorithm, acting as the crucial bridge between the high-level Python frontend and the low-level C++/HLS backend. After the Python frontend parses the user's algorithm, it generates a DFG-IR graph, which the backend then consumes to produce C++ code.

The core definitions for the DFG-IR are located in `graphyflow/dataflow_ir.py` and `graphyflow/dataflow_ir_datatype.py`.

## 2. Key Concepts and Data Structures

The DFG-IR models the algorithm as a directed graph where nodes are computational units (`Component`) and edges, represented by `Port` connections, signify the flow of data streams.

### 2.1. DfirType: The Type System

Before data flows, its type must be defined. The DFG-IR has its own abstract type system, defined in `graphyflow/dataflow_ir_datatype.py`.

* **`DfirType`**: The base class for all types in the IR.
* **Basic Types**: Includes `IntType`, `FloatType`, and `BoolType` for scalar values.
* **`SpecialType`**: Represents graph-specific concepts, namely `"node"` and `"edge"`.
* **`ArrayType`**: A container type representing a stream or array of elements of another `DfirType`. Most data flows through the graph as `ArrayType`.
* **`TupleType`**: Represents a collection of other types, used when multiple values are grouped together, for example, as the output of a `map_` operation.
* **`OptionalType`**: A special container that holds a value and a validity flag. This is crucial for implementing the `filter` operation.

### 2.2. `Component`
A `Component` is the fundamental node in the DFG-IR graph. Each component represents a single, elementary operation, such as an addition, a data copy, or an attribute access. The high-level frontend operators (`map_`, `filter`, etc.) are decomposed into a graph of these fine-grained components during the conversion from Python lambda functions.

### 2.3. `Port`
Each `Component` has a set of input (`i_...`) and output (`o_...`) `Port`s. These are the connection points for the data streams. An output `Port` of one component connects to an input `Port` of another, defining the exact path of the dataflow. Each port has a specific `DfirType` associated with it.

### 2.4. `ComponentCollection`
This class acts as a container for the entire DFG-IR graph. It holds the list of all `Component` instances and tracks the overall `inputs` and `outputs` of the graph (i.e., ports that are not connected internally). This is the final object that the frontend produces and the backend consumes.

## 3. Component Reference

The DFG-IR consists of various component types, each mapping to a specific, simple operation.

* **`IOComponent`**: Represents the entry or exit point of the entire dataflow graph (e.g., reading the initial stream of edges from global memory).
* **`ConstantComponent`**: Injects a constant value (e.g., `2.0`, `True`) into the dataflow graph for use in operations.
* **`BinOpComponent`**: Represents a binary operation (e.g., `+`, `-`, `==`, `<`). It has two input ports (`i_0`, `i_1`) and one output port (`o_0`). Its specific operation is defined by an enum value from `dfir.BinOp`.
* **`UnaryOpComponent`**: Represents a unary operation (e.g., negation, type casting, or attribute access like `edge.src`). It has one input port (`i_0`) and one output port (`o_0`). Its operation is defined by an enum value from `dfir.UnaryOp` and may include a `select_index` for attribute/tuple access.
* **`GatherComponent` / `ScatterComponent`**: These are used to manage tuples. `GatherComponent` combines multiple input streams into a single output stream of tuples. `ScatterComponent` does the opposite, taking a stream of tuples and splitting it into multiple output streams, one for each element of the tuple. They are essential for handling lambdas with multiple arguments.
* **`ConditionalComponent` / `CollectComponent`**: This pair of components work together to implement the `filter` logic.
    1.  `ConditionalComponent` takes a data stream (`i_data`) and a boolean stream (`i_cond`). It outputs a stream of `OptionalType`, where each element is valid only if its corresponding boolean condition was `True`.
    2.  `CollectComponent` takes the stream of `OptionalType` and filters out all invalid elements, producing a dense, continuous output stream of the original data type.
* **`ReduceComponent`**: This is a special, complex component that encapsulates the control flow and stateful logic for a grouped reduction. It does not contain the user's logic itself. Instead, it has special ports (e.g., `o_reduce_key_in`, `i_reduce_key_out`, `o_reduce_unit_start_0`) that connect to sub-graphs. These sub-graphs, composed of simpler components like `BinOpComponent`, implement the user-defined logic for the `reduce_key`, `reduce_transform`, and `reduce_method` lambdas.

## 4. IR Passes

The DFG-IR is not merely a static representation; it can be transformed and optimized by "passes". A pass is a function that takes a `ComponentCollection` and returns a modified version.

An example is the `delete_placeholder_components_pass` found in `graphyflow/passes.py`. During the initial IR construction from the frontend's lambda tracing, temporary `PlaceholderComponent` nodes are created. This pass traverses the graph, finds these placeholders, and stitches their upstream and downstream components directly together, effectively removing the redundant placeholder. This simplifies the graph, making it cleaner and easier for the backend to process.

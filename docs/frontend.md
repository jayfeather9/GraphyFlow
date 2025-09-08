# GraphyFlow Frontend Documentation

## 1\. Introduction

The GraphyFlow frontend is the user-facing Python API that allows developers to define complex graph algorithms in a high-level, functional style. The core idea is to represent the data as a stream that flows through a series of processing nodes (operators), each performing a specific transformation. This API abstracts away the underlying hardware details, enabling developers to focus purely on the algorithm's logic. The primary source file for this API is `graphyflow/global_graph.py`.

## 2\. Core Concepts

### The `GlobalGraph` Class

This class is the main container and the starting point for any algorithm definition. When you instantiate it, you must define the data schema for your graph's nodes and edges via the `properties` dictionary. This schema is critical as it informs the compiler about the data structures and types it will be working with.

**Initialization Example:**
The following example from `tests/bellman_ford.py` defines a graph where each node has a `distance` property and each edge has a `weight` property. GraphyFlow automatically adds standard properties like `id` for nodes and `src`/`dst` for edges.

```python
import graphyflow.dataflow_ir as dfir
from graphyflow.global_graph import GlobalGraph

# Instantiate the main graph object
g = GlobalGraph(
    properties={
        "node": {"distance": dfir.FloatType()},
        "edge": {"weight": dfir.FloatType()},
    }
)
```

### The `PseudoElement` Object

This object represents the stream of data flowing through your graph's processing steps. You do not create it directly. Instead, you obtain the initial stream by calling `g.add_graph_input()`. Each subsequent operator call (like `.map_` or `.filter`) returns a new `PseudoElement` instance, representing the transformed stream. This allows for a fluent, chainable API style.

## 3\. Algorithm Construction Workflow

Defining an algorithm in GraphyFlow follows a clear, sequential pattern:

1.  **Initialize `GlobalGraph`**: Define the data schema for your nodes and edges.
2.  **Create Initial Stream**: Use `g.add_graph_input()` to create the initial `PseudoElement` data stream from the graph's edges or nodes.
3.  **Chain Operators**: Call a series of operator methods on the `PseudoElement` object to build the dataflow pipeline. Each call transforms the stream and passes it to the next operator.

## 4\. Detailed Operator Reference

Operators are methods of the `PseudoElement` object that transform the data stream. They are the core building blocks of an algorithm.

### `add_graph_input(type_)`

This is the starting point of any algorithm. It creates an initial data stream from the graph's nodes or edges.

  * `type_`: A string, either `"edge"` or `"node"`.

**Example:**

```python
# Creates a stream of all edges in the graph
edges = g.add_graph_input("edge")
```

### `map_(map_func)`

Performs an element-wise transformation on a data stream. It applies the `map_func` lambda to each element in the input stream and produces a new stream with the results.

  * `map_func`: A Python lambda function.
      * **Input**: It receives arguments that are unpacked from the elements of the input stream. For example, if the stream contains tuples `(a, b)`, the lambda should be written as `lambda a, b: ...`. If the stream contains `edge` objects, it should be `lambda edge: ...`.
      * **Output**: The value returned by the lambda defines the structure and content of the elements in the *new*, transformed stream. Returning a tuple is common for creating multi-part data elements.

**Example from `tests/bellman_ford.py`**:
This `map_` call takes a stream of `edge` objects and transforms it into a stream of tuples, each containing the source node's distance, the destination node object, and the edge's weight.

```python
# Input stream 'edges' contains 'edge' objects
# Output stream 'pdu' will contain (float, node, float) tuples
pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
```

### `filter(filter_func)`

Selectively keeps or discards elements from a data stream based on a predicate.

  * `filter_func`: A Python lambda function.
      * **Input**: It receives arguments unpacked from the elements of the input stream.
      * **Output**: It **must** return a boolean (`True` or `False`). If `True`, the element is kept in the stream; if `False`, it is discarded.

**Example from `tests/bellman_ford.py`**:
This `filter` call takes a stream of `(src_dist, dst, edge_w)` tuples and keeps only those where the edge weight is non-negative.

```python
# Input stream 'pdu' contains (float, node, float) tuples
pdu = pdu.filter(filter_func=lambda src_dist, dst, edge_w: edge_w >= 0.0)
```

### `reduce_by(reduce_key, reduce_transform, reduce_method)`

A powerful and highly parallelizable operator for performing grouped reductions, analogous to the combine/reduce phase in MapReduce. It groups elements by a key and then aggregates the elements within each group.

  * `reduce_key`: A lambda that computes a key for each element. All elements that produce the same key will be processed together in a single reduction group.
  * `reduce_transform`: A lambda that is applied to each element *before* it enters the reduction phase. It prepares or transforms the data into the format expected by `reduce_method`.
  * `reduce_method`: A lambda that defines the core aggregation logic. It must be a binary function (taking two arguments) that is **commutative and associative** (e.g., addition, minimum, maximum). It takes two transformed elements from the same group and combines them into a single element of the same format.

**Example from `tests/bellman_ford.py`**:
This is the core of the Bellman-Ford relaxation step.

```python
from graphyflow.lambda_func import lambda_min

min_dist = pdu.reduce_by(
    # 1. Group by the destination node's ID. All updates for the same node go to the same group.
    reduce_key=lambda src_dist, dst, edge_w: dst.id,

    # 2. Before reduction, transform the input tuple into a (new_potential_distance, destination_node) tuple.
    reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),

    # 3. For each group, repeatedly apply this method. It takes two (dist, node) tuples and
    #    returns a new one with the minimum distance, effectively finding the minimum distance for that destination node.
    reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
)
```

## 5\. Lambda Functions and Tracing

It is important to understand that GraphyFlow does not execute the Python lambdas directly during compilation. Instead, it uses a **tracing mechanism** defined in `graphyflow/lambda_func.py` to analyze the operations within the lambda.

When you write `edge.src.distance + edge.weight`, the `+` operation is intercepted by the tracer, which records that an addition operation occurred. This sequence of traced operations is then converted into the DFG-IR components.

This has one important implication: standard Python functions that cannot be traced (like the built-in `min()`) may not work as expected in all contexts, especially within `reduce_method`. For this reason, GraphyFlow provides traceable equivalents like `lambda_min` to ensure the compiler can correctly interpret the intended operation. Basic arithmetic (`+`, `*`, `<`, `==`, etc.) and attribute access (`.`) are automatically traced.

# Middle-End IR Architecture

## Overview

The middle-end IR (Intermediate Representation) provides a typed, graph-based representation of the computation where operations are connected through typed streams.

## Key Components

### 1. Type System (`IRType`)

All data in the IR is strongly typed:

- **`IntType(width)`**: Integer types with configurable bit width (e.g., Int16, Int32)
- **`FloatType()`**: Floating-point type
- **`BoolType()`**: Boolean type
- **`NodeType()`**: Graph node type
- **`EdgeType()`**: Graph edge type
- **`ArrayType(element_type)`**: Array/stream of elements
- **`TupleType(element_types)`**: Tuple with multiple typed elements

### 2. Stream Connections (`Stream`)

Streams represent typed data flow between operations:

```
Stream#id<Type>
  source: IRNode (producer)
  consumers: [(IRNode, port), ...] (consumers with input ports)
```

Each stream:
- Has a unique ID
- Is strongly typed
- Tracks its source node and output port
- Tracks all consumer nodes and their input ports

### 3. IR Nodes (`IRNode`)

All operations are represented as nodes with typed inputs and outputs:

#### Basic Operations

- **`InputNode(name, type)`**: Lambda function input parameter
- **`ConstNode(value, type)`**: Constant value
- **`AttrAccessNode(attr, input_type, output_type)`**: Attribute access (e.g., `.dst`, `.weight`)
- **`BinOpNode(op, type)`**: Binary operations (+, -, *, /, <, <=, >, >=, ==, !=, min, max)

#### High-Level Operations

- **`MapIR`**: Map operation over arrays
  - Input: Array stream
  - Contains: Lambda body as IR nodes
  - Output: Array of transformed elements

- **`ReduceIR`**: Reduce/aggregation operation
  - Inputs: Key stream + value stream(s)
  - Contains: Lambda body for reduction function
  - Output: Array of reduced values (optionally with keys)

### 4. IR Graph (`IRGraph`)

Container for the entire computation:
- Tracks all nodes and streams
- Provides methods to visualize the graph structure
- Shows connections between operations

## Example: Shortest Path (tmp.py)

### Frontend Code
```python
edges = graph.start_iteration([gf.ALL_EDGES])
keys = gf.map([edges], lambda e: e.dst)
updated_dists = gf.map([edges], lambda e: e.src.dist + e.weight)
reduced_dists = gf.reduce(keys=keys, values=[updated_dists], 
                         method=lambda a, b: gf.min(a, b), 
                         output_with_key=True)
```

### Middle-End IR Structure

```
Stream#0<Array<Edge>> (input edges)
    ↓
Map#2: lambda e: e.dst
    Lambda: Input(arg0:Edge) → AttrAccess(.dst) → Node
    Output: Stream#2<Array<Node>>
    
Stream#0<Array<Edge>> (input edges)
    ↓
Map#8: lambda e: e.src.dist + e.weight
    Lambda: Input(arg0:Edge) 
            → AttrAccess(.src) → Node
            → AttrAccess(.dist) → Int16
            + AttrAccess(.weight) → Int32
            → BinOp(+) → Int16
    Output: Stream#7<Array<Int16>>

Stream#2<Array<Node>> (keys) + Stream#7<Array<Int16>> (values)
    ↓
Reduce#12: lambda a, b: min(a, b)
    Keys: Stream#2
    Values: Stream#7
    Lambda: Input(arg0:Int16), Input(arg1:Int16) 
            → BinOp(min) → Int16
    Output: Stream#10<Array<Tuple<Int16, Node>>>
```

## Key Features

1. **Type Safety**: Every stream and node has explicit types
2. **Traceability**: Each node tracks its inputs and outputs
3. **Connection Graph**: Streams explicitly connect producers to consumers
4. **Lambda Transparency**: Lambda bodies are fully expanded into IR nodes
5. **Composability**: Operations can be chained through streams
6. **Analyzability**: The graph structure can be analyzed and optimized

## Conversion Flow

```
Frontend (Python DSL)
    ↓ trace lambdas
OpTracer expressions
    ↓ convert to IR
Middle-End IR (typed graph)
    ↓ (future)
Backend code generation
```

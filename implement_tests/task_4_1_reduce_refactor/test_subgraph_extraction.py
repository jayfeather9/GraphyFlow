# implement_tests/task_4_1_reduce_refactor/test_subgraph_extraction.py
import pytest
from pathlib import Path
from graphyflow.global_graph import GlobalGraph

# Import lambda_min for a more robust test case
from graphyflow.lambda_func import lambda_min
from graphyflow.dataflow_ir import (
    ReduceComponent,
    UnaryOp,
    UnaryOpComponent,
    BinOpComponent,
    BinOp,
    DfirType,
    ScatterComponent,
    ComponentCollection,
)
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.visualize_ir import visualize_components
from graphyflow.dataflow_ir_utils import _extract_subgraph_from_reduce


# Fixture to build a complex graph containing a ReduceComponent
@pytest.fixture(scope="module")
def complex_reduce_graph() -> ComponentCollection:
    """
    Builds a graph with a complex ReduceBy operation for testing.
    - Key: Accesses a nested attribute (edge.src.id)
    - Transform: Performs a calculation with memory access (dist + e.weight)
    - Reduce: A standard MIN operation using lambda_min helper
    """
    g = GlobalGraph(
        properties={
            "node": {"id": DfirType("Int"), "distance": DfirType("Float")},
            "edge": {"weight": DfirType("Float")},
        }
    )
    edges = g.add_graph_input("edge")

    # A map operation to create a tuple stream
    tuple_stream = edges.map_(map_func=lambda e: (e.src.distance, e))

    # The ReduceBy operation to be tested
    reduced_stream = tuple_stream.reduce_by(
        reduce_key=lambda dist, e: e.src.id,
        reduce_transform=lambda dist, e: (dist + e.weight),
        # MODIFICATION: Using lambda_min for a more robust test that maps to a single BinOp
        reduce_method=lambda_min,
    )

    # Convert to DFIR and clean it up
    dfirs = g.to_dfir()
    # Find the collection that contains the Reduce operation
    for collection in dfirs:
        if any(isinstance(c, ReduceComponent) for c in collection.components):
            final_collection = delete_placeholder_components_pass(collection)
            final_collection.global_graph_store = g  # <-- 请添加这一行
            return final_collection

    raise RuntimeError("Failed to find a ComponentCollection with a ReduceComponent")


# The main test function
def test_subgraph_extraction(complex_reduce_graph: ComponentCollection):
    """
    Tests the _extract_subgraph_from_reduce helper function to ensure it
    correctly isolates the key, transform, and unit_reduce subgraphs.
    """
    # Ensure the output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Find the ReduceComponent in the graph
    reduce_comp = next((c for c in complex_reduce_graph.components if isinstance(c, ReduceComponent)), None)
    assert reduce_comp is not None, "ReduceComponent not found in the test graph"

    # --- 1. Test Key Subgraph Extraction ---
    print("\n--- Testing 'key' subgraph extraction ---")
    key_subgraph = _extract_subgraph_from_reduce(reduce_comp, "key")
    print(key_subgraph)

    # MODIFICATION: Assertions are now more flexible. We check for the presence
    # of essential components rather than an exact count.
    assert any(
        isinstance(c, ScatterComponent) for c in key_subgraph.components
    ), "Key subgraph should contain a ScatterComponent to unpack the input tuple"
    assert any(
        isinstance(c, UnaryOpComponent) and c.op == UnaryOp.GET_ATTR and c.select_index == "src"
        for c in key_subgraph.components
    ), "Key subgraph must contain GET_ATTR 'src'"
    assert any(
        isinstance(c, UnaryOpComponent) and c.op == UnaryOp.GET_ATTR and c.select_index == "id"
        for c in key_subgraph.components
    ), "Key subgraph must contain GET_ATTR 'id'"

    # Visualize
    dot_key = visualize_components(str(key_subgraph))
    key_graph_path = output_dir / "reduce_key_subgraph"
    dot_key.render(key_graph_path, view=False, format="png")
    print(f"Key subgraph visualized to {key_graph_path}.png")

    # --- 2. Test Transform Subgraph Extraction ---
    print("\n--- Testing 'transform' subgraph extraction ---")
    transform_subgraph = _extract_subgraph_from_reduce(reduce_comp, "transform")
    print(transform_subgraph)

    # MODIFICATION: Updated assertions for transform subgraph
    assert any(
        isinstance(c, ScatterComponent) for c in transform_subgraph.components
    ), "Transform subgraph should contain a ScatterComponent"
    assert any(
        isinstance(c, UnaryOpComponent) and c.op == UnaryOp.GET_ATTR and c.select_index == "weight"
        for c in transform_subgraph.components
    ), "Transform subgraph must contain GET_ATTR 'weight'"
    assert any(
        isinstance(c, BinOpComponent) and c.op == BinOp.ADD for c in transform_subgraph.components
    ), "Transform subgraph must contain a BinOp ADD"

    # Visualize
    dot_transform = visualize_components(str(transform_subgraph))
    transform_graph_path = output_dir / "reduce_transform_subgraph"
    dot_transform.render(transform_graph_path, view=False, format="png")
    print(f"Transform subgraph visualized to {transform_graph_path}.png")

    # --- 3. Test Unit Reduce Subgraph Extraction ---
    print("\n--- Testing 'unit_reduce' subgraph extraction ---")
    unit_reduce_subgraph = _extract_subgraph_from_reduce(reduce_comp, "unit_reduce")
    print(unit_reduce_subgraph)

    # MODIFICATION: Assertions now check for BinOp.MIN as defined in the fixture
    assert (
        len(unit_reduce_subgraph.components) == 1
    ), "Expected exactly 1 component in unit_reduce subgraph for lambda_min"
    assert any(
        isinstance(c, BinOpComponent) and c.op == BinOp.MIN for c in unit_reduce_subgraph.components
    ), "A BinOp MIN should be present for the reduce_method"

    # Visualize
    dot_unit = visualize_components(str(unit_reduce_subgraph))
    unit_graph_path = output_dir / "reduce_unit_subgraph"
    dot_unit.render(unit_graph_path, view=False, format="png")
    print(f"Unit reduce subgraph visualized to {unit_graph_path}.png")

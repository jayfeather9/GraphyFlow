# implement_tests/task_4_1_reduce_refactor/test_subgraph_refactoring.py
import pytest
from pathlib import Path
from graphyflow.global_graph import GlobalGraph
from graphyflow.dataflow_ir import ReduceComponent, FusedOpComponent, UnaryOp, UnaryOpComponent
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.visualize_ir import visualize_components
from graphyflow.dataflow_ir_utils import _refactor_and_consolidate_reduce_subgraphs

# We reuse the same complex graph fixture from the previous test
from .test_subgraph_extraction import complex_reduce_graph


def test_subgraph_refactoring_and_consolidation(complex_reduce_graph: GlobalGraph):
    """
    Tests the refactoring of subgraphs and consolidation of their memory accesses.
    """
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # We need the GlobalGraph object to resolve types during refactoring
    # The fixture provides the ComponentCollection, so we need to rebuild the GlobalGraph
    # For simplicity in this test, we'll re-instantiate it.
    g = complex_reduce_graph.global_graph_store  # Assuming the fixture is modified to store this

    # Find the ReduceComponent in the graph
    reduce_comp = next((c for c in complex_reduce_graph.components if isinstance(c, ReduceComponent)), None)
    assert reduce_comp is not None, "ReduceComponent not found in the test graph"

    # --- Run the consolidation function ---
    print("\n--- Testing subgraph refactoring and consolidation ---")
    results = _refactor_and_consolidate_reduce_subgraphs(reduce_comp, g)

    # --- 1. Assert FusedOp Components ---
    fused_op_key = results["fused_op_key"]
    fused_op_transform = results["fused_op_transform"]
    fused_op_unit = results["fused_op_unit_reduce"]

    assert isinstance(fused_op_key, FusedOpComponent)
    assert isinstance(fused_op_transform, FusedOpComponent)
    assert isinstance(fused_op_unit, FusedOpComponent)

    # Check that memory access ops are gone from the fused components' subgraphs
    for comp in fused_op_key.sub_graph.components:
        if isinstance(comp, UnaryOpComponent):
            assert comp.op != UnaryOp.GET_ATTR, "GET_ATTR op found in fused_op_key"

    for comp in fused_op_transform.sub_graph.components:
        if isinstance(comp, UnaryOpComponent):
            assert comp.op != UnaryOp.GET_ATTR, "GET_ATTR op found in fused_op_transform"

    print("Verified: FusedOpComponents contain no memory access operations.")

    # Visualize the internal logic of the new FusedOp components
    dot_fused_key = visualize_components(str(fused_op_key.sub_graph))
    dot_fused_key.render(output_dir / "fused_op_key_subgraph", view=False, format="png")

    dot_fused_transform = visualize_components(str(fused_op_transform.sub_graph))
    dot_fused_transform.render(output_dir / "fused_op_transform_subgraph", view=False, format="png")

    dot_fused_unit = visualize_components(str(fused_op_unit.sub_graph))
    dot_fused_unit.render(output_dir / "fused_op_unit_subgraph", view=False, format="png")
    print("Visualized internal subgraphs of FusedOpComponents.")

    # --- 2. Assert Unified Access Pattern ---
    unified_pattern = results["unified_access_pattern"]
    print("\n--- Unified Access Pattern ---")
    print(unified_pattern)

    # Convert to a set of tuples for easy comparison
    pattern_set = {(in_idx, base, tuple(path)) for in_idx, base, path in unified_pattern}

    # Define the expected memory accesses from the test case
    expected_patterns = {
        (0, "edge", ("src", "distance")),
        (0, "edge", ("src", "id")),
        (0, "edge", ("weight",)),
    }

    print("Expected Patterns:", expected_patterns)
    print("Actual Patterns:", pattern_set)

    assert (
        pattern_set == expected_patterns
    ), "The consolidated access pattern does not match the expected one."
    print("Verified: Unified access pattern is correct.")

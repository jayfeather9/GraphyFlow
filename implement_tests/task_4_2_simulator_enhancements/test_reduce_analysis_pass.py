import pytest
from graphyflow.dataflow_ir import ReduceComponent
from graphyflow.passes import analyze_reduce_comp

# Reuse the complex graph fixture from a previous test file
# This requires pytest to find it. Make sure __init__.py files are present.
# Assuming this file is in the same directory as test_subgraph_extraction.py
# from ..test_subgraph_extraction import complex_reduce_graph
from ..task_4_1_reduce_refactor.test_subgraph_extraction import complex_reduce_graph


def test_simplify_reduce_pass_analysis(complex_reduce_graph):
    """
    Tests the simplify_reduce_comp_pass to ensure it correctly analyzes
    a complex ReduceComponent and returns an accurate refactoring plan.
    """
    # The fixture provides a ComponentCollection. The pass also needs the GlobalGraph.
    # The fixture was modified in the test file to attach the graph object.
    collection = complex_reduce_graph
    g = collection.global_graph_store

    # --- 1. Run the analysis pass ---
    results = analyze_reduce_comp(collection, g)

    # log results to file out.txt formatted tab=2
    with open("out.txt", "w") as f:
        for key, result in results.items():
            import pprint

            f.write(f"ReduceComponent ID: {key}\n")
            f.write(pprint.pformat(result.__dict__, indent=2))
            f.write("\n\n")

    # --- 2. Validate the results ---
    assert len(results) == 1, "The pass should have found exactly one ReduceComponent"

    reduce_comp = next(c for c in collection.components if isinstance(c, ReduceComponent))
    analysis_result = results[reduce_comp.readable_id]

    # --- 2a. Validate Consolidated Memory Accesses ---
    mem_accesses = analysis_result.consolidated_mem_access

    # For `complex_reduce_graph`, the inputs to reduce_by are (dist, e)
    # Key: `e.src.id` -> base is `e`, at input index 1.
    # Transform: `(e.src.distance, dist + e.weight)` -> uses `e` and `dist`

    # Convert to a set of tuples for easier comparison
    access_set = {(info.base_type, info.mem_path, info.scatter_path_to_base) for info in mem_accesses}

    # The original lambda `lambda dist, e: ...` means the input tuple is (dist, e).
    # To get to the base object `e` for all memory accesses, the scatter path is `(1,)`.
    expected_accesses = {
        # from key: e.src.id
        ("edge", ("src", "id"), (1,)),
        # from transform: e.src.distance
        ("edge", ("src", "distance"), (1,)),
        # from transform: e.weight
        ("edge", ("weight",), (1,)),
    }

    assert len(access_set) == 3
    # This assertion is very strict and is the core of the test
    assert access_set == expected_accesses, "Consolidated memory access plan is incorrect"

    # --- 2b. Validate FusedOp components ---
    assert analysis_result.key_analysis.refactored_fused_op is not None
    assert analysis_result.transform_analysis.refactored_fused_op is not None
    # Check the new unit_analysis field
    assert analysis_result.unit_analysis.refactored_fused_op is not None

    # --- 2c. Validate Unit Subgraph Provenance ---
    unit_provenance = analysis_result.unit_analysis.input_provenance
    # The unit_reduce subgraph has two inputs.
    assert len(unit_provenance) == 4

    unit_fused_op = analysis_result.unit_analysis.refactored_fused_op
    # assert (0, (0,)), (0, (1,)), (1, (0, )), (1, (1,)) in unit_provenance.values()
    assert (0, (1,)) in unit_provenance.values()  # dist
    assert (0, (0,)) in unit_provenance.values()  # e
    assert (1, (1,)) in unit_provenance.values()  # dist
    assert (1, (0,)) in unit_provenance.values()  # e

    print("\nSimplify Reduce Pass Analysis Verified:")
    print(f"Found {len(mem_accesses)} unique memory accesses.")
    for access in mem_accesses:
        print(f"  - {access}")
    print("  - Verified unit_reduce input provenance.")

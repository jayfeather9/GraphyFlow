import pytest
import copy
import graphyflow.dataflow_ir as dfir
from graphyflow.passes import optimize_reduce_comp
from graphyflow.simulate import DfirSimulator

# Reuse the complex graph fixture as it provides a perfect, fully-connected test case
from ..task_4_1_reduce_refactor.test_subgraph_extraction import complex_reduce_graph


@pytest.fixture
def test_data():
    """Provides consistent simulation data for tests."""
    node_data = {
        10: {"id": 10, "distance": 100.0},
        20: {"id": 20, "distance": 50.0},
        30: {"id": 30, "distance": 0.0},
    }
    edge_data = {
        101: {"src": 10, "dst": 20, "weight": 5.0},
        102: {"src": 20, "dst": 30, "weight": 12.0},
        103: {"src": 10, "dst": 30, "weight": 2.0},
    }
    # This input will be tuple_stream: (e.src.distance, e)
    initial_input_data = [(100.0, 101), (50.0, 102), (100.0, 103)]
    return node_data, edge_data, initial_input_data


def test_full_reduce_optimization_and_functional_equivalence(complex_reduce_graph, test_data):
    """
    Tests the final, complete reduce optimization pass.
    1. Verifies the final graph structure.
    2. Verifies functional equivalence by comparing simulation results.
    """
    # --- 1. Setup ---
    original_collection = complex_reduce_graph
    g = original_collection.global_graph_store
    reduce_comp_orig = next(c for c in original_collection.components if isinstance(c, dfir.ReduceComponent))
    node_data, edge_data, initial_input_data = test_data

    # --- 2. Get baseline result by simulating the ORIGINAL logic ---
    # To test the original reduce logic in isolation, we must manually
    # disconnect its i_0 and o_0 ports from the larger graph.
    original_collection_isolated = copy.deepcopy(original_collection)
    reduce_comp_isolated = next(
        c for c in original_collection_isolated.components if c.readable_id == reduce_comp_orig.readable_id
    )

    # Disconnect i_0 from its upstream provider
    if reduce_comp_isolated.get_port("i_0").connected:
        reduce_comp_isolated.get_port("i_0").connection.disconnect()

    # Disconnect o_0 from the graph output
    original_collection_isolated.outputs.clear()
    original_collection_isolated.update_ports()

    sim_orig = DfirSimulator(original_collection_isolated, g)
    sim_orig.add_nodes(list(node_data.keys()), node_data)
    sim_orig.add_edges(
        edges={i: (d["src"], d["dst"]) for i, d in edge_data.items()},
        props=edge_data,
    )

    original_results = sim_orig.run_flow(
        from_ports_values={reduce_comp_isolated.get_port("i_0"): initial_input_data},
        to_ports=[reduce_comp_isolated.get_port("o_0")],
    )
    original_output = original_results[reduce_comp_isolated.get_port("o_0")]

    # --- 3. Execute the full optimization pass ---
    # Per the new contract, we must pass a disconnected ReduceComponent
    reduce_comp_disconnected = copy.deepcopy(reduce_comp_orig)
    if reduce_comp_disconnected.get_port("i_0").connected:
        reduce_comp_disconnected.get_port("i_0").connection.disconnect()
    if reduce_comp_disconnected.get_port("o_0").connected:
        # This shouldn't happen if it's a graph output, but we check anyway
        reduce_comp_disconnected.get_port("o_0").connection.disconnect()

    optimized_cc = optimize_reduce_comp(reduce_comp_disconnected, g)

    # --- 4. Verify the structure of the optimized graph ---
    assert len(optimized_cc.inputs) == 1, "Optimized graph should have one main input"
    assert len(optimized_cc.outputs) == 1, "Optimized graph should have one main output"

    scatter_comp = next((c for c in optimized_cc.components if isinstance(c, dfir.ScatterComponent)), None)
    mem_read_comp = next(
        (c for c in optimized_cc.components if isinstance(c, dfir.MemoryReadComponent)), None
    )
    modified_reduce = next((c for c in optimized_cc.components if isinstance(c, dfir.ReduceComponent)), None)

    assert scatter_comp is not None
    assert mem_read_comp is not None
    assert modified_reduce is not None

    assert scatter_comp.get_port("i_0") == optimized_cc.inputs[0]
    assert mem_read_comp.in_ports[0].connection.parent == scatter_comp
    assert modified_reduce.get_port("o_0") == optimized_cc.outputs[0]

    # --- 5. Verify functional equivalence by simulating the OPTIMIZED graph ---
    sim_opt = DfirSimulator(optimized_cc, g)
    sim_opt.add_nodes(list(node_data.keys()), node_data)
    sim_opt.add_edges(
        edges={i: (d["src"], d["dst"]) for i, d in edge_data.items()},
        props=edge_data,
    )

    optimized_results = sim_opt.run({optimized_cc.inputs[0].name: initial_input_data})
    optimized_output = optimized_results[optimized_cc.outputs[0].name]

    # --- 6. Compare results ---
    # The order of reduce results is not guaranteed, so sort them for comparison.
    # The output is a list of tuples. We sort by the first element of the tuple.
    sorted_original = sorted(original_output, key=lambda x: str(x))
    sorted_optimized = sorted(optimized_output, key=lambda x: str(x))

    print(f"Original Result (sorted): {sorted_original}")
    print(f"Optimized Result (sorted): {sorted_optimized}")

    assert sorted_original == sorted_optimized, "Optimized graph output does not match original"

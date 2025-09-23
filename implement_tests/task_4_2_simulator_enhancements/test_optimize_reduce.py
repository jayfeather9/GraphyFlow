import pytest
import copy
import random
import graphyflow.dataflow_ir as dfir
from graphyflow.passes import optimize_reduce_comp
from graphyflow.simulate import DfirSimulator

# Reuse the complex graph fixture as it provides a perfect, fully-connected test case
from ..task_4_1_reduce_refactor.test_subgraph_extraction import complex_reduce_graph

# --- Global variables to control the size of the randomly generated graph ---
NUM_NODES = 500
NUM_EDGES = 2000


def _generate_random_test_data(num_nodes: int, num_edges: int):
    """
    Generates a random, connected graph with properties for simulation.
    """
    # --- 1. Generate a connected graph structure ---
    nodes = list(range(num_nodes))
    edges = set()

    # Ensure connectivity by creating a spanning tree first
    in_tree = {random.choice(nodes)}
    not_in_tree = list(set(nodes) - in_tree)
    random.shuffle(not_in_tree)

    for node_to_add in not_in_tree:
        connect_to = random.choice(list(in_tree))
        # Randomize edge direction
        u, v = (node_to_add, connect_to) if random.random() < 0.5 else (connect_to, node_to_add)
        edges.add((u, v))
        in_tree.add(node_to_add)

    # Add remaining edges randomly to reach the desired count
    while len(edges) < num_edges:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and (u, v) not in edges:
            edges.add((u, v))

    # --- 2. Create node and edge property dictionaries ---
    node_data = {i: {"id": i, "distance": round(random.uniform(0.0, 1000.0), 2)} for i in nodes}

    edge_list = sorted(list(edges))  # Sort for consistent edge IDs
    edge_data = {
        i
        + 101: {  # Use an offset for edge IDs
            "src": u,
            "dst": v,
            "weight": round(random.uniform(1.0, 20.0), 2),
        }
        for i, (u, v) in enumerate(edge_list)
    }

    # --- 3. Create the initial input data for the simulation ---
    # The format is (e.src.distance, e_id)
    initial_input_data = [(node_data[props["src"]]["distance"], eid) for eid, props in edge_data.items()]

    return node_data, edge_data, initial_input_data


@pytest.fixture
def test_data():
    """Provides consistent simulation data for tests by generating a random graph."""
    # Ensure the number of edges is valid for a connected graph
    if NUM_EDGES < NUM_NODES - 1:
        raise ValueError(
            f"NUM_EDGES ({NUM_EDGES}) must be at least NUM_NODES - 1 ({NUM_NODES - 1}) for a connected graph."
        )

    return _generate_random_test_data(NUM_NODES, NUM_EDGES)


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
    original_collection_isolated = copy.deepcopy(original_collection)
    reduce_comp_isolated = next(
        c for c in original_collection_isolated.components if c.readable_id == reduce_comp_orig.readable_id
    )

    if reduce_comp_isolated.get_port("i_0").connected:
        reduce_comp_isolated.get_port("i_0").connection.disconnect()

    original_collection_isolated.outputs = [reduce_comp_isolated.get_port("o_0")]
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
    reduce_comp_disconnected = copy.deepcopy(reduce_comp_orig)
    if reduce_comp_disconnected.get_port("i_0").connected:
        reduce_comp_disconnected.get_port("i_0").connection.disconnect()
    if reduce_comp_disconnected.get_port("o_0").connected:
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
    sorted_original = sorted(original_output, key=lambda x: str(x))
    sorted_optimized = sorted(optimized_output, key=lambda x: str(x))

    print(f"Original Result (sorted): {sorted_original}")
    print(f"Optimized Result (sorted): {sorted_optimized}")

    assert sorted_original == sorted_optimized, "Optimized graph output does not match original"

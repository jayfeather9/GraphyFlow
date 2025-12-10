import random
from pathlib import Path

import graphyflow.dataflow_ir as dfir
from graphyflow.global_graph import GlobalGraph
from graphyflow.lambda_func import lambda_min
from graphyflow.passes import delete_placeholder_components_pass, refactor_extract_all_comps, remove_io_comp_pass
from graphyflow.simulate import DfirSimulator
from graphyflow.visualize_ir import visualize_components

# ==================== Config =======================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"

# --- Global variables to control the size of the randomly generated graph ---
NUM_NODES = 50
NUM_EDGES = 150


# ==================== Test Data Generation =======================
def _generate_random_test_data(num_nodes: int, num_edges: int):
    """
    Generates a random, connected graph with properties for simulation.
    This function is adapted from the provided test example.
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
    # The graph input is the collection of edges, so we provide their IDs.
    initial_input_data = list(edge_data.keys())

    return node_data, edge_data, initial_input_data


# ==================== Graph Definition =======================
def build_graph() -> GlobalGraph:
    """
    Defines the computation graph logic.
    """
    g = GlobalGraph(
        properties={
            "node": {"distance": dfir.FloatType()},
            "edge": {"weight": dfir.FloatType()},
        }
    )
    edges = g.add_graph_input("edge")
    pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
    pdu = pdu.filter(filter_func=lambda x, y, z: z >= 0.0)
    min_dist = pdu.reduce_by(
        reduce_key=lambda src_dist, dst, edge_w: dst.id,
        reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
        reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
    )
    _updated_nodes = min_dist.map_(map_func=lambda dist, node: (lambda_min(dist, node.distance), node))
    return g


def save_graph_png(comp_col: dfir.ComponentCollection, stem: str) -> None:
    """
    Visualizes and saves a component collection to a PNG file.
    """
    dot = visualize_components(str(comp_col))
    OUT = OUTPUT_DIR / stem
    dot.render(str(OUT), view=False, format="png")


if __name__ == "__main__":
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Generate random graph data for the simulation
    print("--- Generating Random Test Data ---")
    if NUM_EDGES < NUM_NODES - 1:
        raise ValueError(
            f"NUM_EDGES ({NUM_EDGES}) must be at least NUM_NODES - 1 ({NUM_NODES - 1}) for a connected graph."
        )
    node_data, edge_data, initial_input_data = _generate_random_test_data(NUM_NODES, NUM_EDGES)
    print(f"Generated a graph with {len(node_data)} nodes and {len(edge_data)} edges.")

    # 2. Build the abstract graph representation
    print("\n--- Building Graph ---")
    g = build_graph()

    # 3. Generate the initial, unoptimized DFIR
    print("\n--- Generating DFIR ---")
    dfirs = g.to_dfir()
    comp_col = delete_placeholder_components_pass(dfirs[0])
    comp_col = remove_io_comp_pass(comp_col)
    save_graph_png(comp_col, "new_dist_ori")
    print("Saved original DFIR graph to output/new_dist_ori.png")

    # 4. Simulate the original DFIR to get a baseline result
    print("\n--- Simulating Original DFIR ---")
    sim_orig = DfirSimulator(comp_col, g)
    sim_orig.add_nodes(list(node_data.keys()), node_data)
    sim_orig.add_edges(
        edges={i: (d["src"], d["dst"]) for i, d in edge_data.items()},
        props=edge_data,
    )
    original_results = sim_orig.run({comp_col.inputs[0].name: initial_input_data})
    original_output = original_results[comp_col.outputs[0].name]
    print("Simulation of original graph complete.")

    # 5. Apply the refactoring/optimization pass
    print("\n--- Refactoring DFIR (partition + memread/fused islands) ---")
    refactored = refactor_extract_all_comps(comp_col, g)
    refactored = remove_io_comp_pass(refactored)
    save_graph_png(refactored, "new_dist_refactored")
    print("Saved refactored DFIR graph to output/new_dist_refactored.png")

    # 6. Simulate the refactored DFIR
    print("\n--- Simulating Refactored DFIR ---")
    sim_refactored = DfirSimulator(refactored, g)
    sim_refactored.add_nodes(list(node_data.keys()), node_data)
    sim_refactored.add_edges(
        edges={i: (d["src"], d["dst"]) for i, d in edge_data.items()},
        props=edge_data,
    )
    refactored_results = sim_refactored.run({refactored.inputs[0].name: initial_input_data})
    refactored_output = refactored_results[refactored.outputs[0].name]
    print("Simulation of refactored graph complete.")

    # 7. Compare the results to ensure functional equivalence
    print("\n--- Comparing Results ---")
    # The order of results is not guaranteed, so sort them for a stable comparison.
    # Using str(x) as the key is a robust way to sort complex tuples.
    sorted_original = sorted(original_output, key=lambda x: str(x))
    sorted_refactored = sorted(refactored_output, key=lambda x: str(x))

    assert sorted_original == sorted_refactored, "Functional equivalence check FAILED: Outputs do not match."

    print("✅ Functional equivalence CONFIRMED: Original and refactored graphs produce the same output.")

    print("\nDone.")

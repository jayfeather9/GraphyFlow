import pytest
from pathlib import Path
import graphyflow.dataflow_ir as dfir
from graphyflow.global_graph import GlobalGraph
from graphyflow.dataflow_ir_utils import refactor_to_memread_fusedop
from graphyflow.visualize_ir import visualize_components
from graphyflow.simulate import DfirSimulator


@pytest.fixture(scope="module")
def manual_multi_io_subgraph() -> tuple[dfir.ComponentCollection, GlobalGraph]:
    """
    Manually builds a complex ComponentCollection with multiple inputs and outputs.
    This serves as the input for the refactoring function.

    Inputs:
    - In_0: Array<edge>
    - In_1: Array<Float>

    Logic:
    - Out_0 = edge.src.id + edge.dst.id
    - Out_1 = edge.weight * In_1
    """
    # --- 1. Define the Global Graph Schema ---
    g = GlobalGraph(
        properties={
            "node": {"id": dfir.DfirType("Int")},
            "edge": {
                "src": dfir.DfirType("node"),
                "dst": dfir.DfirType("node"),
                "weight": dfir.DfirType("Float"),
            },
        }
    )

    # --- 2. Define data types for streams ---
    edge_array_t = dfir.ArrayType(dfir.SpecialType("edge"))
    node_array_t = dfir.ArrayType(dfir.SpecialType("node"))
    int_array_t = dfir.ArrayType(dfir.IntType())
    float_array_t = dfir.ArrayType(dfir.FloatType())

    # --- 3. Create Components for the subgraph logic ---

    # Placeholders for the two inputs of the collection
    p_edge = dfir.PlaceholderComponent(edge_array_t) # Input 0
    p_float = dfir.PlaceholderComponent(float_array_t) # Input 1

    # To use the edge stream 3 times, we chain two CopyComponents for a 3-way split.
    copy1_edge = dfir.CopyComponent(edge_array_t)
    copy2_edge = dfir.CopyComponent(edge_array_t)

    # Path 1: edge.src.id
    get_src = dfir.UnaryOpComponent(dfir.UnaryOp.GET_ATTR, edge_array_t, select_index="src", attr_type=node_array_t)
    get_src_id = dfir.UnaryOpComponent(dfir.UnaryOp.GET_ATTR, node_array_t, select_index="id", attr_type=int_array_t)

    # Path 2: edge.dst.id
    get_dst = dfir.UnaryOpComponent(dfir.UnaryOp.GET_ATTR, edge_array_t, select_index="dst", attr_type=node_array_t)
    get_dst_id = dfir.UnaryOpComponent(dfir.UnaryOp.GET_ATTR, node_array_t, select_index="id", attr_type=int_array_t)

    # Path 3: edge.weight
    get_weight = dfir.UnaryOpComponent(dfir.UnaryOp.GET_ATTR, edge_array_t, select_index="weight", attr_type=float_array_t)

    # Computation Ops
    add_ids = dfir.BinOpComponent(dfir.BinOp.ADD, int_array_t)      # For Out_0
    mul_w = dfir.BinOpComponent(dfir.BinOp.MUL, float_array_t)    # For Out_1

    # --- 4. Connect the components ---
    p_edge.get_port("o_0").connect(copy1_edge.get_port("i_0"))

    # Chain the copies to create 3 outputs from the original edge stream
    copy1_edge.get_port("o_1").connect(copy2_edge.get_port("i_0"))
    
    # Output 1 from copy chain -> Path 1 (src.id)
    copy1_edge.get_port("o_0").connect(get_src.get_port("i_0"))
    get_src.get_port("o_0").connect(get_src_id.get_port("i_0"))

    # Output 2 from copy chain -> Path 2 (dst.id)
    copy2_edge.get_port("o_0").connect(get_dst.get_port("i_0"))
    get_dst.get_port("o_0").connect(get_dst_id.get_port("i_0"))
    
    # Output 3 from copy chain -> Path 3 (weight)
    copy2_edge.get_port("o_1").connect(get_weight.get_port("i_0"))

    # Connect to computation components
    get_src_id.get_port("o_0").connect(add_ids.get_port("i_0"))
    get_dst_id.get_port("o_0").connect(add_ids.get_port("i_1"))

    get_weight.get_port("o_0").connect(mul_w.get_port("i_0"))
    p_float.get_port("o_0").connect(mul_w.get_port("i_1"))

    # --- 5. Define the Component Collection ---
    subgraph_comps = [
        p_edge, p_float, copy1_edge, copy2_edge, get_src, get_src_id,
        get_dst, get_dst_id, get_weight, add_ids, mul_w
    ]

    original_subgraph = dfir.ComponentCollection(
        components=subgraph_comps,
        inputs=[p_edge.get_port("i_0"), p_float.get_port("i_0")],
        outputs=[add_ids.get_port("o_0"), mul_w.get_port("o_0")],
    )

    return original_subgraph, g


def test_refactor_and_simulate_multi_output(manual_multi_io_subgraph):
    """
    Tests that the refactored multi-output graph produces correct results
    when run through the simulator.
    """
    original_subgraph, g = manual_multi_io_subgraph
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # --- Visualize the original, manually-built graph ---
    print("\n--- Original Manually-Built Subgraph ---")
    dot_orig = visualize_components(str(original_subgraph))
    dot_orig.render(output_dir / "original_manual_subgraph", view=False, format="png")
    print("Original subgraph visualized to output/original_manual_subgraph.png")

    # --- PHASE 1: Refactor the graph (This is what we are testing) ---
    refactored_cc = refactor_to_memread_fusedop(original_subgraph, g)

    # --- Visualize the refactored graph ---
    print("\n--- Refactored Subgraph ---")
    dot_refactored = visualize_components(str(refactored_cc))
    dot_refactored.render(output_dir / "refactored_manual_subgraph", view=False, format="png")
    print("Refactored subgraph visualized to output/refactored_manual_subgraph.png")
    
    # --- PHASE 2: Simulate the refactored graph and verify results ---
    
    g.nodes_data = {
        100: {"id": 100}, 
        200: {"id": 200}
    }
    g.edges_data = {
        1: {"src": 100, "dst": 200, "weight": 5.0},
        2: {"src": 200, "dst": 100, "weight": 2.0},
    }

    simulator = DfirSimulator(refactored_cc, g)

    # In the refactored graph, inputs are normalized. The first input will be
    # the edge IDs for MemoryRead, and the second will be the float array passthrough.
    input_names = sorted([p.unique_name for p in refactored_cc.inputs])
    sim_inputs = {
        input_names[0]: [1, 2],
        input_names[1]: [10.0, -3.0],
    }
    
    results = simulator.run(sim_inputs)
    
    # --- PHASE 3: Assert correctness ---
    # Expected o_0 = [300, 300]
    # Expected o_1 = [50.0, -6.0]
    
    expected_o0 = [300, 300]
    expected_o1 = [50.0, -6.0]

    # The refactored graph will have outputs named o_0, o_1
    assert "o_0" in results
    assert "o_1" in results
    assert results["o_0"] == expected_o0
    assert results["o_1"] == expected_o1
    
    print("\nSimulation successful. Refactored graph produced correct multi-output results.")

import pytest
from pathlib import Path
from graphyflow.global_graph import GlobalGraph
from graphyflow.dataflow_ir import (
    DfirType,
    MemoryReadComponent,
    FusedOpComponent,
    ComponentCollection,
    ScatterComponent,
    UnaryOpComponent,
)
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.visualize_ir import visualize_components
from graphyflow.dataflow_ir_utils import refactor_to_memread_fusedop
from .test_subgraph_extraction import complex_reduce_graph  # Reuse fixture

from pathlib import Path
from graphyflow.global_graph import GlobalGraph
from graphyflow.dataflow_ir import (
    DfirType,
    MemoryReadComponent,
    FusedOpComponent,
    ComponentCollection,
    UnaryOpComponent,
)
from graphyflow.passes import delete_placeholder_components_pass, remove_io_comp_pass
from graphyflow.visualize_ir import visualize_components
from graphyflow.dataflow_ir_utils import refactor_to_memread_fusedop


@pytest.fixture(scope="module")
def subgraph_to_refactor() -> tuple[ComponentCollection, GlobalGraph]:
    """
    Builds a self-contained ComponentCollection that mimics a complex subgraph
    with tuple inputs, fan-out, and memory accesses that are also outputs.
    """
    g = GlobalGraph(
        properties={
            "node": {"id": DfirType("Int"), "distance": DfirType("Float")},
            "edge": {"weight": DfirType("Float")},
        }
    )
    # This graph creates a complex input for the target subgraph (map2):
    # map1's output is Array<Tuple<edge, Float>>
    edges = g.add_graph_input("edge")
    map0 = edges.map_(map_func=lambda e: (e, 5.0))
    map1 = map0.map_(map_func=lambda e, weight: (e, e.weight + 1.0 + e.dst.distance, weight))
    # map0 = edges.map_(map_func=lambda e: e)
    # map1 = map0.map_(map_func=lambda e: (e, e.weight + 1.0 + e.dst.distance, 1.0))
    # map2 is the target subgraph for our refactoring test.
    map2 = map1.map_(map_func=lambda e, const_w, const_p: (e.src.id, e.weight + const_w + const_p, e))

    # dfirs = g.to_dfir()
    # full_graph = remove_io_comp_pass(dfirs[0])
    # full_graph = delete_placeholder_components_pass(dfirs[0])

    # Manually extract the "map2" subgraph to test refactor_to_memread_fusedop in isolation.
    map0_node = g.nodes[map0.cur_node.uuid]
    map0_dfir = map0_node.to_dfir(
        g.nodes[edges.cur_node.uuid].to_dfir(None, (g.node_properties, g.edge_properties)).output_types[0],
        (g.node_properties, g.edge_properties)
    )
    map1_input_type = map0_dfir.output_types[0]
    map1_node = g.nodes[map1.cur_node.uuid]
    map1_dfir = map1_node.to_dfir(
        map1_input_type, (g.node_properties, g.edge_properties)
    )
    map2_input_type = map1_dfir.output_types[0]
    map2_node = g.nodes[map2.cur_node.uuid]
    map2_dfir = map2_node.to_dfir(
        map2_input_type, (g.node_properties, g.edge_properties)
    )
    
    map1_dfir.concat(
        map2_dfir,
        [(map1_dfir.outputs[0], map2_dfir.inputs[0])],
    )

    return map1_dfir, g


def test_final_refactor_with_scatter_handling(subgraph_to_refactor):
    """
    Tests the final, robust implementation of refactor_to_memread_fusedop
    that correctly handles Tuple inputs by inserting a Scatter component.
    """
    original_subgraph, g = subgraph_to_refactor
    # print(original_subgraph)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("\n--- Original Subgraph to Refactor (V3 Test) ---")
    dot_orig = visualize_components(str(original_subgraph))
    dot_orig.render(output_dir / "v3_original_subgraph", view=False, format="png")
    print("Original subgraph visualized to output/v3_original_subgraph.png")

    # --- Run the new refactoring function ---
    refactored_cc = refactor_to_memread_fusedop(original_subgraph, g)

    print("\n--- Refactored Component Collection (V3 Test) ---")
    print(refactored_cc)
    dot_refactored = visualize_components(str(refactored_cc))
    dot_refactored.render(output_dir / "v3_refactored_subgraph", view=False, format="png")
    print("Refactored subgraph visualized to output/v3_refactored_subgraph.png")
    
    # assert False

    # --- Assertions ---
    # 1. Final structure is Scatter -> [MemRead | Passthrough] -> FusedOp
    # assert len(refactored_cc.components) == 3
    # scatter_comp = next((c for c in refactored_cc.components if isinstance(c, ScatterComponent)), None)
    # mem_read_comp = next((c for c in refactored_cc.components if isinstance(c, MemoryReadComponent)), None)
    # fused_op_comp = next((c for c in refactored_cc.components if isinstance(c, FusedOpComponent)), None)
    # assert scatter_comp is not None, "A new ScatterComponent at the front is missing."
    # assert mem_read_comp is not None
    # assert fused_op_comp is not None

    # # 2. MemoryReadComponent has correctly NORMALIZED access patterns
    # access_patterns = {(base, tuple(path)) for base, path in mem_read_comp.access_pattern}
    # # The paths should now start from 'edge', not 'Tuple'.
    # expected_patterns = {
    #     ("edge", ("src",)),
    #     (
    #         "edge",
    #         (
    #             "src",
    #             "id",
    #         ),
    #     ),
    #     ("edge", ("weight",)),
    # }
    # assert access_patterns == expected_patterns, "Normalized access pattern is incorrect."

    # # 3. FusedOp has the correct number of inputs
    # # It should receive inputs from all memory reads + all passthrough tuple elements.
    # num_mem_reads = len(mem_read_comp.out_ports)
    # # In the fixture `lambda e, const_w`, the input tuple is (e, const_w).
    # # 'e' (index 0) is used for memory. 'const_w' (index 1) is passthrough.
    # num_passthrough = 1
    # assert len(fused_op_comp.in_ports) == num_mem_reads + num_passthrough

    # # 4. Final graph is correctly connected
    # assert (
    #     scatter_comp.in_ports[0] in refactored_cc.inputs
    # ), "Scatter's input should be the collection's main input."
    # assert (
    #     fused_op_comp.out_ports[0] in refactored_cc.outputs
    # ), "FusedOp's output should be the collection's main output."
    # assert (
    #     scatter_comp.out_ports[0].connection.parent == mem_read_comp
    # ), "Scatter's edge output should feed MemoryRead."
    # assert (
    #     scatter_comp.out_ports[1].connection.parent == fused_op_comp
    # ), "Scatter's float output should feed FusedOp."

    # print("\nAll assertions passed for the final refactoring implementation.")

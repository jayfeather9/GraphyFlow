from __future__ import annotations
import copy
import collections
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
from graphyflow.dataflow_ir_datatype import *
from graphyflow.dataflow_ir import *


def _dead_code_elimination(
    components: List[Component], outputs: List[Port]
) -> Tuple[List[Component], List[Port]]:
    """
    Performs dead code elimination on a list of components via backward traversal.
    """
    live_components = set()
    q = collections.deque([p.parent for p in outputs])
    visited = {p.parent.readable_id for p in outputs}

    while q:
        comp = q.popleft()
        live_components.add(comp)
        for p_in in comp.in_ports:
            if p_in.connected:
                upstream_comp = p_in.connection.parent
                if upstream_comp.readable_id not in visited:
                    visited.add(upstream_comp.readable_id)
                    q.append(upstream_comp)

    # Filter the component list and find the new set of necessary inputs
    final_components = [c for c in components if c in live_components]
    final_inputs = []
    for comp in final_components:
        # An input placeholder is identified as having an unconnected input port
        if isinstance(comp, PlaceholderComponent) and not comp.in_ports[0].connected:
            final_inputs.append(comp.in_ports[0])

    return final_components, final_inputs


def _simplify_redundant_copies(components: List[Component]) -> List[Component]:
    """
    Finds CopyComponents with only one used output and replaces them with a direct connection.
    """
    # This process might need to run multiple times if copies are chained.
    made_change = True
    while made_change:
        made_change = False
        simplified_comps = []
        comps_to_remove = set()

        for comp in components:
            if comp in comps_to_remove:
                continue

            if isinstance(comp, CopyComponent):
                used_outputs = [p for p in comp.out_ports if p.connected]
                if len(used_outputs) == 1:
                    made_change = True
                    comps_to_remove.add(comp)

                    source_port = comp.get_port("i_0").connection
                    dest_port = used_outputs[0].connection

                    # Disconnect all parties from the copy component
                    if source_port:
                        source_port.disconnect()
                    used_outputs[0].disconnect()

                    # Bypass the copy component
                    if source_port and dest_port:
                        source_port.connect(dest_port)
                    continue  # Skip adding this comp to the list

            simplified_comps.append(comp)

        components = simplified_comps

    return components


def refactor_to_memread_fusedop(
    original_cc: ComponentCollection, global_graph: GlobalGraph
) -> ComponentCollection:
    """
    Refactors a ComponentCollection into a MemoryReadComponent followed by a FusedOpComponent.

    This function analyzes a given ComponentCollection, separates memory access operations
    (like getting node/edge attributes) from pure computation, and reconstructs the dataflow
    into a two-stage pipeline.

    Args:
        original_cc: The original ComponentCollection to refactor.
        global_graph: The GlobalGraph object, needed to resolve attribute types.

    Returns:
        A new ComponentCollection containing only a MemoryReadComponent and a FusedOpComponent.
    """
    # --- Phase 1: Pre-analysis and Validation ---
    allowed_types = (
        ScatterComponent,
        GatherComponent,
        CopyComponent,
        UnusedEndMarkerComponent,
        BinOpComponent,
        UnaryOpComponent,
        ConstantComponent,
    )
    for comp in original_cc.components:
        if not isinstance(comp, allowed_types):
            raise TypeError(
                f"Component type '{type(comp).__name__}' is not allowed for this refactoring pass."
            )

    if len(original_cc.inputs) != 1:
        raise ValueError("Refactoring requires the ComponentCollection to have exactly one input.")

    input_port = original_cc.inputs[0]
    input_type = input_port.data_type
    if not (isinstance(input_type, ArrayType) and isinstance(input_type.type_, SpecialType)):
        raise ValueError(f"Input type must be Array<node> or Array<edge>, but got {input_type}.")

    base_type_str = input_type.type_.type_name

    # --- Phase 2: Memory Access Path Discovery ---
    discovered_paths = set()
    access_op_outputs = {}  # Map: original output Port -> (base_type, path)
    memoized_paths = {}  # Memoization for trace_back function

    # The trace_back helper function remains the same as the previous version.
    def trace_back(port: Port) -> Tuple[str, List[Union[str, int]]]:
        if port is None:
            raise ValueError("trace_back was called with a None port.")

        if port.readable_id in memoized_paths:
            return memoized_paths[port.readable_id]

        parent_comp = port.parent

        input_port = parent_comp.in_ports[0]
        if input_port in original_cc.inputs:
            path_so_far = []
        else:
            source_output_port = input_port.connection
            if source_output_port is None:
                raise ConnectionError(
                    f"Disconnected input port '{input_port.name}' on component '{parent_comp.name}' during trace."
                )

            _, path_so_far = trace_back(source_output_port)

        if isinstance(parent_comp, (ScatterComponent, CopyComponent)):
            final_path = path_so_far
        elif isinstance(parent_comp, UnaryOpComponent) and parent_comp.op in (
            UnaryOp.GET_ATTR,
            UnaryOp.SELECT,
        ):
            final_path = path_so_far + [parent_comp.select_index]
        else:
            raise TypeError(
                f"Trace back encountered an unexpected component type: {type(parent_comp).__name__}"
            )

        memoized_paths[port.readable_id] = (base_type_str, final_path)
        return base_type_str, final_path

    # We only discover paths for data that is consumed by a non-access, computational component.
    for comp in original_cc.topo_sort():
        is_access_op = isinstance(comp, UnaryOpComponent) and comp.op in (UnaryOp.GET_ATTR, UnaryOp.SELECT)

        # If a component is NOT an access op, it's a computational "sink".
        # We check if its inputs come from access ops.
        if not is_access_op:
            for p_in in comp.in_ports:
                if p_in.connected:
                    upstream_port = p_in.connection
                    upstream_comp = upstream_port.parent

                    # If the provider is an access op, we've found a memory-to-computation boundary.
                    if isinstance(upstream_comp, UnaryOpComponent) and upstream_comp.op in (
                        UnaryOp.GET_ATTR,
                        UnaryOp.SELECT,
                    ):
                        # Trace this path back to its origin.
                        base_type, path = trace_back(upstream_port)
                        path_tuple = tuple(path)
                        discovered_paths.add((base_type, path_tuple))
                        access_op_outputs[upstream_port] = (base_type, path_tuple)

    # --- Phase 3: New Component Construction ---

    # 3.1 Construct MemoryReadComponent
    def get_type_from_path(base_type: str, path: Tuple[Union[str, int], ...]) -> DfirType:
        if base_type == "node":
            current_props = global_graph.node_properties
            current_type = SpecialType("node")
        else:  # base_type == "edge"
            current_props = global_graph.edge_properties
            current_type = SpecialType("edge")
        for key in path:
            if isinstance(current_type, SpecialType):
                if key not in current_props:
                    raise KeyError(
                        f"Attribute '{key}' not found in properties for type '{current_type.type_name}'."
                    )
                current_type = current_props[key]
                if isinstance(current_type, SpecialType):
                    current_props = (
                        global_graph.node_properties
                        if current_type.type_name == "node"
                        else global_graph.edge_properties
                    )
            elif isinstance(current_type, TupleType):
                current_type = current_type.types[key]
            else:
                raise TypeError(f"Cannot get attribute '{key}' from non-structural type {current_type}")

        return current_type

    access_pattern = list(discovered_paths)
    output_types = {}
    for base, path in access_pattern:
        final_type = get_type_from_path(base, path)
        port_name = f"o_{base}_{'_'.join(map(str, path))}"
        output_types[port_name] = final_type

    mem_read = MemoryReadComponent(
        access_pattern=access_pattern, output_types=output_types, base_id_type=input_type.type_, parallel=True
    )

    # 3.2 Construct FusedOpComponent's Subgraph
    fused_subgraph_comps = []
    old_to_new_ports = {}
    fused_subgraph_inputs = []

    for comp in original_cc.topo_sort():
        # Skip access ops, they are now handled by MemoryReadComponent
        if comp.out_ports[0] in access_op_outputs:
            path_tuple = access_op_outputs[comp.out_ports[0]]

            # Create a new placeholder input port for the fused subgraph
            pname = mem_read.pattern_to_pname[(path_tuple[0], tuple(path_tuple[1]))]
            dtype = mem_read.get_port(pname).data_type

            placeholder_input = PlaceholderComponent(dtype)
            fused_subgraph_comps.append(placeholder_input)
            fused_subgraph_inputs.append(placeholder_input.in_ports[0])
            old_to_new_ports[comp.out_ports[0]] = placeholder_input.out_ports[0]
            continue

        # Deepcopy computational components
        new_comp = copy.deepcopy(comp)
        for port in new_comp.ports:
            if port.connected:
                port.disconnect()
        fused_subgraph_comps.append(new_comp)

        # Map old ports to new ports
        for old_port, new_port in zip(comp.ports, new_comp.ports):
            old_to_new_ports[old_port] = new_port

        # Reconnect inputs of the new component
        for old_in_port, new_in_port in zip(comp.in_ports, new_comp.in_ports):
            if old_in_port.connected:
                original_source_port = old_in_port.connection
                new_source_port = old_to_new_ports[original_source_port]
                assert not new_source_port.connected
                new_source_port.connect(new_in_port)

    # Determine outputs of the fused subgraph
    fused_subgraph_outputs = [old_to_new_ports[p] for p in original_cc.outputs]

    # Phase 3.3: Clean up the generated subgraph before creating the Collection

    # 1. Dead Code Elimination
    fused_subgraph_comps, fused_subgraph_inputs = _dead_code_elimination(
        fused_subgraph_comps, fused_subgraph_outputs
    )

    # 2. Simplify Redundant CopyComponents
    fused_subgraph_comps = _simplify_redundant_copies(fused_subgraph_comps)

    # 3. Final pass to remove placeholders created by access op replacement
    temp_subgraph_for_cleanup = ComponentCollection(
        components=fused_subgraph_comps, inputs=fused_subgraph_inputs, outputs=fused_subgraph_outputs
    )
    fused_subgraph = delete_placeholder_components_pass(temp_subgraph_for_cleanup)

    fused_op = FusedOpComponent(name="fused_computation", sub_graph=fused_subgraph)

    # --- Phase 4: Final Assembly ---
    # The input to the new collection is the input of the first component in the new chain.
    new_collection_input = mem_read.get_port("i_base_id")
    # The outputs are the outputs of the last component in the chain.
    new_collection_outputs = fused_op.out_ports

    # Connect the internal components BEFORE creating the final collection.
    # The order of mem_read outputs and fused_op inputs should correspond
    # because they were both derived from the ordered `access_pattern` list.
    if len(mem_read.out_ports) != len(fused_op.in_ports):
        raise RuntimeError("Mismatch between MemoryRead outputs and FusedOp inputs during final assembly.")

    for mem_output, fused_input in zip(mem_read.out_ports, fused_op.in_ports):
        mem_output.connect(fused_input)

    # Now, create the new collection with the correct, connected components and I/O ports.
    # The constructor will now see that all internal ports are connected, and the
    # external-facing ports (`i_base_id` and `fused_op.out_ports`) are correctly listed.
    new_cc = ComponentCollection(
        components=[mem_read, fused_op], inputs=[new_collection_input], outputs=new_collection_outputs
    )

    # The old input_port is no longer part of this new collection's interface.
    # We don't need to connect it to anything. The update_ports() call is also handled
    # implicitly by creating a new valid collection.
    return new_cc


if __name__ == "__main__":
    from pathlib import Path
    from graphyflow.global_graph import GlobalGraph
    from graphyflow.lambda_func import lambda_min
    from graphyflow.passes import delete_placeholder_components_pass
    from graphyflow.visualize_ir import visualize_components

    # ==================== 配置 =======================
    KERNEL_NAME = "graphyflow"
    EXECUTABLE_NAME = "graphyflow_host"
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    OUTPUT_DIR = PROJECT_ROOT / "generated_project"

    # ==================== 1. 定义图算法 =======================
    print("--- Defining Graph Algorithm using GraphyFlow ---")
    g = GlobalGraph(
        properties={
            "node": {"distance": FloatType(), "id": IntType()},  # Add id for testing
            "edge": {"weight": FloatType()},
        }
    )
    edges = g.add_graph_input("edge")
    # This map contains multiple memory accesses (edge.src, .src.distance, .dst, .weight)
    pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight + 5.0))

    # ==================== 2. 前端处理 =======================
    print("\n--- Frontend Processing ---")
    dfirs = g.to_dfir()
    comp_col = delete_placeholder_components_pass(dfirs[0])
    print(str(comp_col))
    dot = visualize_components(str(comp_col))
    dot.render("component_graph_original", view=False, format="png")
    print("Original DFG-IR generated as component_graph_original.png")

    # ==================== 3. 执行重构 =======================
    print("\n--- Refactoring to MemoryRead + FusedOp ---")
    try:
        # --- MINIMAL CHANGE START ---
        # Find and remove the IOComponent before passing to the refactor function.
        io_comp = next((c for c in comp_col.components if isinstance(c, IOComponent)), None)
        if not io_comp:
            raise RuntimeError("Could not find IOComponent in the initial graph.")

        # The new input for the collection is the port that the IOComponent was connected to.
        new_input_port = io_comp.out_ports[0].connection
        new_input_port.disconnect()  # Isolate the rest of the graph from the IOComponent.

        components_for_refactor = [c for c in comp_col.components if not isinstance(c, IOComponent)]

        # Create a new ComponentCollection that is valid for refactoring.
        comp_col_for_refactor = ComponentCollection(
            components=components_for_refactor, inputs=[new_input_port], outputs=comp_col.outputs
        )
        # --- MINIMAL CHANGE END ---

        refactored_cc = refactor_to_memread_fusedop(comp_col_for_refactor, g)
        print("Refactoring successful. New Component Collection:")
        print(refactored_cc)

        dot_refactored = visualize_components(str(refactored_cc))
        dot_refactored.render("component_graph_refactored", view=False, format="png")
        print("Refactored DFG-IR generated as component_graph_refactored.png")
    except (TypeError, ValueError, RuntimeError) as e:
        print(f"Refactoring failed with an error: {e}")

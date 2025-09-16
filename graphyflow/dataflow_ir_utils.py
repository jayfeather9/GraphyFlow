from __future__ import annotations
import copy
import collections
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple, Set
from graphyflow.dataflow_ir_datatype import *
from graphyflow.dataflow_ir import *
from graphyflow.global_graph import GlobalGraph
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.visualize_ir import visualize_components
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataOrigin:
    """Stores the origin of a data stream for analysis."""

    # The ultimate source: either a primary input port or a computed port.
    source_port: Port
    # The accumulated memory access path from the source.
    access_path: Tuple[Union[str, int], ...] = field(default_factory=tuple)


# ======================================================================== #
#                  PASS 1: SIMPLIFY REDUCE COMPONENT                       #
# ======================================================================== #


def _extract_subgraph_from_reduce(reduce_comp: ReduceComponent, subgraph_type: str) -> ComponentCollection:
    """
    Extracts a subgraph (key, transform, or unit_reduce) from a ReduceComponent
    and returns it as a new, self-contained ComponentCollection where external
    inputs are replaced by PlaceholderComponents.
    """
    if subgraph_type == "key":
        start_ports = [reduce_comp.get_port("o_reduce_key_in")]
        end_port = reduce_comp.get_port("i_reduce_key_out")
    elif subgraph_type == "transform":
        start_ports = [reduce_comp.get_port("o_reduce_transform_in")]
        end_port = reduce_comp.get_port("i_reduce_transform_out")
    elif subgraph_type == "unit_reduce":
        start_ports = [
            reduce_comp.get_port("o_reduce_unit_start_0"),
            reduce_comp.get_port("o_reduce_unit_start_1"),
        ]
        end_port = reduce_comp.get_port("i_reduce_unit_end")
    else:
        raise ValueError(f"Unknown subgraph_type: {subgraph_type}")

    subgraph_comps_set = set()
    q = collections.deque()
    visited_ids = set()

    # The end component is the one connected to the end_port
    end_comp = end_port.connection.parent

    # Start forward traversal from the components connected to the entry ports
    for port in start_ports:
        if port.connected:
            comp = port.connection.parent
            if comp.readable_id not in visited_ids:
                q.append(comp)
                visited_ids.add(comp.readable_id)

    while q:
        comp = q.popleft()
        subgraph_comps_set.add(comp)
        if comp == end_comp:
            continue
        for p_out in comp.out_ports:
            if p_out.connected:
                downstream_comp = p_out.connection.parent
                if isinstance(downstream_comp, ReduceComponent):
                    continue
                if downstream_comp.readable_id not in visited_ids:
                    q.append(downstream_comp)
                    visited_ids.add(downstream_comp.readable_id)

    subgraph_comps = list(subgraph_comps_set)
    subgraph_outputs = [end_port.connection]

    # Make the subgraph self-contained by replacing external inputs with placeholders
    subgraph_inputs = []

    # Identify the actual input ports of the subgraph components
    actual_input_ports = []
    for comp in subgraph_comps:
        for p_in in comp.in_ports:
            # An input port is an entry point if it's connected to something
            # OUTSIDE the subgraph component set.
            if p_in.connected and p_in.connection.parent not in subgraph_comps_set:
                actual_input_ports.append(p_in)

    # Replace each external connection with a placeholder
    for p_in in actual_input_ports:
        # Disconnect from the external ReduceComponent port
        p_in.disconnect()

        # Create and connect a placeholder
        placeholder = PlaceholderComponent(p_in.data_type)
        subgraph_comps.append(placeholder)
        placeholder.get_port("o_0").connect(p_in)

        # The placeholder's input is now the new official input for the collection
        subgraph_inputs.append(placeholder.get_port("i_0"))

    return ComponentCollection(components=subgraph_comps, inputs=subgraph_inputs, outputs=subgraph_outputs)


def _refactor_and_consolidate_reduce_subgraphs(
    reduce_comp: ReduceComponent, global_graph: GlobalGraph
) -> Dict[str, Any]:
    """
    Extracts, refactors, and consolidates the subgraphs of a ReduceComponent.

    This function performs the core refactoring on each of the three subgraphs
    ('key', 'transform', 'unit_reduce'), separates their memory access from
    computation, and returns the consolidated access patterns along with the
    new pure-computation FusedOpComponents.

    Returns:
        A dictionary containing the three FusedOpComponents and the unified
        set of memory access patterns required by all of them.
    """
    results = {}
    all_access_patterns = set()

    # 1. Refactor 'key' and 'transform' subgraphs
    for subgraph_type in ["key", "transform"]:
        subgraph_cc = _extract_subgraph_from_reduce(reduce_comp, subgraph_type)

        # Refactor the subgraph to separate memory access from computation
        refactored_cc = refactor_to_memread_fusedop(subgraph_cc, global_graph)

        # The refactored graph should contain a MemoryRead and a FusedOp component
        mem_read_comp = next(
            (c for c in refactored_cc.components if isinstance(c, MemoryReadComponent)), None
        )
        fused_op_comp = next((c for c in refactored_cc.components if isinstance(c, FusedOpComponent)), None)
        assert fused_op_comp is not None, f"FusedOpComponent not found after refactoring {subgraph_type}"

        # Store the pure-computation part
        results[f"fused_op_{subgraph_type}"] = fused_op_comp

        # Consolidate the memory access patterns
        if mem_read_comp:
            for pattern in mem_read_comp.access_pattern:
                # Convert path list to tuple to make it hashable for the set
                all_access_patterns.add((pattern[0], tuple(pattern[1])))

    # 2. Refactor 'unit_reduce' subgraph (no type dependency needed as per requirement)
    # The assumption is that the transform output type remains consistent.
    unit_subgraph_cc = _extract_subgraph_from_reduce(reduce_comp, "unit_reduce")
    refactored_unit_cc = refactor_to_memread_fusedop(unit_subgraph_cc, global_graph)

    mem_read_unit = next(
        (c for c in refactored_unit_cc.components if isinstance(c, MemoryReadComponent)), None
    )
    fused_op_unit = next((c for c in refactored_unit_cc.components if isinstance(c, FusedOpComponent)), None)
    assert fused_op_unit is not None, "FusedOpComponent not found after refactoring unit_reduce"
    results["fused_op_unit_reduce"] = fused_op_unit
    if mem_read_unit:
        for pattern in mem_read_unit.access_pattern:
            all_access_patterns.add((pattern[0], tuple(pattern[1])))

    # Convert set of tuples back to a list of lists for the MemoryReadComponent constructor
    results["unified_access_pattern"] = [(base, list(path)) for base, path in all_access_patterns]

    return results


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


# In: graphyflow/dataflow_ir_utils.py


def refactor_to_memread_fusedop(
    original_cc: ComponentCollection, global_graph: GlobalGraph
) -> ComponentCollection:
    """
    Refactors a ComponentCollection into a MemoryReadComponent and a FusedOpComponent.
    This is the final, robust implementation that handles complex Tuple inputs
    by normalizing access paths and restructuring the input data flow.
    """
    # print("Original ComponentCollection to Refactor:")
    # print(original_cc)
    # ======================================================================== #
    #                  PHASE 1: DATA-FLOW ANALYSIS                             #
    # ======================================================================== #
    port_origins: Dict[Port, DataOrigin] = {}
    compute_ops: List[Component] = []
    for p_in in original_cc.inputs:
        port_origins[p_in] = DataOrigin(source_port=p_in)

    def shrink_scatter_gather(access_path: Tuple[Union[str, int], ...]) -> Tuple[Union[str, int], ...]:
        """
        Simplifies access paths by removing redundant Scatter/Gather patterns.
        e.g., (0, 'g1', 1) -> (0,)
        """
        simplified_path = []
        skip_next = False
        while True:
            skipped_len = 0
            for i, elem in enumerate(access_path):
                if skip_next:
                    skip_next = False
                    continue
                if isinstance(elem, str) and elem.startswith("_g"):
                    if i + 1 < len(access_path) and isinstance(access_path[i + 1], int):
                        if access_path[i + 1] == int(elem[2:]):
                            skip_next = True
                            skipped_len += 2
                            continue
                simplified_path.append(elem)
            if skipped_len == 0:
                break
            access_path = tuple(simplified_path)
            simplified_path = []
        return access_path

    def get_origin(port: Port, access_index: Optional[int] = None) -> DataOrigin:
        if port.connected:
            origin = port_origins.get(port.connection)
        else:
            origin = port_origins.get(port)
        assert origin is not None, f"Missing origin for port {port}"
        if access_index is None or type(origin) is DataOrigin:
            assert type(origin) in [DataOrigin, list]
            return origin
        else:
            assert type(origin) is list, f"Expected list of origins for port {port}, got {origin}"
            assert isinstance(access_index, int), "Access index must be an integer"
            assert (
                0 <= access_index < len(origin)
            ), f"Access index {access_index} out of range for port {port}"
            assert origin[access_index] is not None, f"Missing origin for port {port} at index {access_index}"
            return origin[access_index]

    for comp in original_cc.topo_sort():
        origin = None
        if isinstance(comp, ConstantComponent):
            compute_ops.append(comp)
            for p_out in comp.out_ports:
                port_origins[p_out] = DataOrigin(source_port=p_out)
            continue
        assert comp.in_ports, f"Component {comp} has no input ports"
        p_in = comp.in_ports[0]

        if isinstance(comp, UnaryOpComponent) and comp.op in (UnaryOp.GET_ATTR, UnaryOp.SELECT):
            origin = get_origin(p_in, comp.select_index)
            assert origin is not None
            new_path = origin.access_path + (comp.select_index,)
            new_path = shrink_scatter_gather(new_path)
            new_origin = DataOrigin(source_port=origin.source_port, access_path=new_path)
            port_origins[comp.out_ports[0]] = new_origin
            continue
        elif isinstance(comp, ScatterComponent):
            for i, p_out in enumerate(comp.out_ports):
                origin = get_origin(p_in, i)
                assert origin is not None
                new_path = origin.access_path + (i,)
                new_path = shrink_scatter_gather(new_path)
                new_origin = DataOrigin(source_port=origin.source_port, access_path=new_path)
                port_origins[p_out] = new_origin
            continue
        elif isinstance(comp, GatherComponent):
            assert comp.out_ports[0] not in port_origins
            port_origins[comp.out_ports[0]] = [None for _ in comp.in_ports]
            for i, p_in in enumerate(comp.in_ports):
                assert p_in.connection is not None, f"Gather port {p_in} shouldn't be input."
                origin = get_origin(p_in)
                assert origin is not None
                new_path = origin.access_path + (f"_g{i}",)
                new_origin = DataOrigin(source_port=origin.source_port, access_path=new_path)
                port_origins[comp.out_ports[0]][i] = new_origin
            assert all(origin is not None for origin in port_origins[comp.out_ports[0]])
            continue
        elif isinstance(comp, (PlaceholderComponent, CopyComponent)):
            origin = get_origin(p_in)
            assert origin is not None
            for p_out in comp.out_ports:
                port_origins[p_out] = origin
            continue

        compute_ops.append(comp)
        for p_out in comp.out_ports:
            port_origins[p_out] = DataOrigin(source_port=p_out)

    # ======================================================================== #
    #                  PHASE 2                                                 #
    # ======================================================================== #

    scatter_paths = {}
    mem_paths = {}
    mem_patterns = set()

    for comp in compute_ops:
        print(f"Compute Component: {comp})")
        for p_in in comp.in_ports:
            origin = port_origins.get(p_in.connection) if p_in.connection else port_origins.get(p_in)
            assert origin is not None, f"Missing origin for port {p_in} in component {comp}"
            print(f"  Input Port: {p_in}, Origin: {origin}")
            if origin.source_port not in original_cc.inputs:
                assert len(origin.access_path) == 0
                continue
            if len(origin.access_path) == 0:
                continue
            current_type = origin.source_port.data_type
            if isinstance(current_type, ArrayType):
                current_type = current_type.type_
            cur_scatter_path = []
            cur_access_path = list(origin.access_path)
            is_simple_access = False
            while not isinstance(current_type, SpecialType):
                if isinstance(current_type, TupleType):
                    index = origin.access_path[0] if origin.access_path else 0
                    cur_access_path = cur_access_path[1:]
                    current_type = current_type.types[index]
                    cur_scatter_path.append(index)
                else:
                    is_simple_access = True
                    break
            scatter_paths[p_in] = (origin.source_port, cur_scatter_path)
            print(f"    Scatter Path: {cur_scatter_path}")
            if not is_simple_access and len(cur_access_path) > 0:
                mem_patterns.add((current_type.type_name, tuple(cur_access_path)))
                mem_paths[p_in] = (current_type.type_name, tuple(cur_access_path))
                print(
                    f"    Memory Access Pattern: Base={current_type.type_name}, Path={tuple(cur_access_path)}"
                )

    print("Analyzing Output Ports:")
    for p_out in original_cc.outputs:
        origin = port_origins.get(p_out.connection) if p_out.connection else port_origins.get(p_out)
        origins = origin if type(origin) is list else [origin]
        for origin in origins:
            print(f"Output Port: {p_out}, Origin: {origin}")
            assert origin is not None, f"Missing origin for output port {p_out}"
            if origin.source_port in original_cc.inputs:
                assert len(origin.access_path) > 0, f"Straight passing through is not allowed now."
                current_type = origin.source_port.data_type
                if isinstance(current_type, ArrayType):
                    current_type = current_type.type_
                cur_scatter_path = []
                cur_access_path = list(origin.access_path)
                is_simple_access = False
                while not isinstance(current_type, SpecialType):
                    if isinstance(current_type, TupleType):
                        index = origin.access_path[0] if origin.access_path else 0
                        cur_access_path = cur_access_path[1:]
                        current_type = current_type.types[index]
                        cur_scatter_path.append(index)
                    else:
                        is_simple_access = True
                        break
                scatter_paths[origin.source_port] = (origin.source_port, cur_scatter_path)
                print(f"    Scatter Path: {cur_scatter_path}")
                if not is_simple_access and len(cur_access_path) > 0:
                    mem_patterns.add((current_type.type_name, tuple(cur_access_path)))
                    mem_paths[origin.source_port] = (current_type.type_name, tuple(cur_access_path))
                    print(
                        f"    Memory Access Pattern: Base={current_type.type_name}, Path={tuple(cur_access_path)}"
                    )

    # ======================================================================== #
    #                  PHASE 3                                                 #
    # ======================================================================== #

    print(scatter_paths)
    print(mem_paths)

    fused_input_ports = []
    mem_targeting_ports = {}
    mem_targeting_scatter_paths = {}
    scatter_targeting_ports = {}
    required_scatter_paths = {}
    waiting_out_ports = []

    def copy_in_port(old_port: Port, new_port: Port) -> CopyComponent:
        print(f"Copying port {old_port} to new port {new_port}")
        copy_comp = CopyComponent(old_port.data_type)
        copy_comp.get_port("o_0").connect(old_port)
        copy_comp.get_port("o_1").connect(new_port)
        return copy_comp, copy_comp.get_port("i_0")

    added_comps = []
    for comp in compute_ops:
        for p_out in comp.out_ports:
            waiting_out_ports.extend(comp.out_ports)
        if isinstance(comp, ConstantComponent):
            continue
        for p_in in comp.in_ports:
            print(f"Processing input port {p_in} of component {comp}")
            origin = get_origin(p_in)
            assert origin is not None, f"Missing origin for port {p_in} in component {comp}"
            if p_in in mem_paths:
                mem_path = mem_paths[p_in]
                assert p_in in scatter_paths, f"Missing scatter path for port {p_in} in component {comp}"
                cur_scatter_path = tuple(scatter_paths[p_in][1])
                p_in.disconnect()
                if mem_path not in mem_targeting_ports:
                    mem_targeting_ports[mem_path] = p_in
                else:
                    copy_comp, new_in_port = copy_in_port(p_in, mem_targeting_ports[mem_path])
                    added_comps.append(copy_comp)
                    mem_targeting_ports[mem_path] = new_in_port
                if cur_scatter_path not in mem_targeting_scatter_paths:
                    mem_targeting_scatter_paths[cur_scatter_path] = set()
                mem_targeting_scatter_paths[cur_scatter_path].add(cur_scatter_path)
                assert origin.source_port in original_cc.inputs
                assert (
                    cur_scatter_path not in required_scatter_paths
                    or required_scatter_paths[cur_scatter_path] == "memory"
                ), f"Conflict scatter path usage for path {cur_scatter_path}"
                required_scatter_paths[cur_scatter_path] = "memory"
                continue
            if p_in in scatter_paths:
                assert origin.source_port in original_cc.inputs
                cur_scatter_path = tuple(scatter_paths[p_in][1])
                p_in.disconnect()
                assert (
                    cur_scatter_path not in required_scatter_paths
                    or required_scatter_paths[cur_scatter_path] == "through"
                ), f"Conflict scatter path usage for path {cur_scatter_path}"
                required_scatter_paths[cur_scatter_path] = "through"
                if cur_scatter_path not in scatter_targeting_ports:
                    scatter_targeting_ports[cur_scatter_path] = p_in
                else:
                    copy_comp, new_in_port = copy_in_port(p_in, scatter_targeting_ports[cur_scatter_path])
                    added_comps.append(copy_comp)
                    scatter_targeting_ports[cur_scatter_path] = new_in_port
                continue
            if origin.source_port.connected and origin.source_port in waiting_out_ports:
                waiting_out_ports.remove(origin.source_port)
                if p_in.connection != origin.source_port:
                    p_in.disconnect()
                    origin.source_port.disconnect()
                    origin.source_port.connect(p_in)
                continue
            assert False, f"Port {p_in} in component {comp} should have been handled."
    compute_ops.extend(added_comps)

    if waiting_out_ports == original_cc.outputs:
        print("No output ports need to be changed.")
    else:
        assert len(original_cc.outputs) == 1
        out_origins = get_origin(original_cc.outputs[0])
        assert type(out_origins) is list
        resorted_origins = [None for _ in out_origins]
        for origin in out_origins:
            index = int(origin.access_path[-1][2:])
            resorted_origins[index] = origin
        out_origins = resorted_origins
        gather_comp = GatherComponent([origin.source_port.data_type for origin in out_origins])
        for i, origin in enumerate(out_origins):
            if origin.source_port in waiting_out_ports:
                waiting_out_ports.remove(origin.source_port)
                gather_port = gather_comp.get_port(f"i_{i}")
                origin.source_port.disconnect()
                origin.source_port.connect(gather_port)
            else:
                assert origin.source_port in original_cc.inputs
                gather_port = gather_comp.get_port(f"i_{i}")
                if origin.source_port in mem_paths:
                    mem_path = mem_paths[origin.source_port]
                    assert (
                        origin.source_port in scatter_paths
                    ), f"Missing scatter path for port {origin.source_port} in gather."
                    cur_scatter_path = tuple(scatter_paths[origin.source_port][1])
                    if mem_path not in mem_targeting_ports:
                        mem_targeting_ports[mem_path] = gather_port
                    else:
                        copy_comp, new_in_port = copy_in_port(gather_port, mem_targeting_ports[mem_path])
                        compute_ops.append(copy_comp)
                        mem_targeting_ports[mem_path] = new_in_port
                    if cur_scatter_path not in mem_targeting_scatter_paths:
                        mem_targeting_scatter_paths[cur_scatter_path] = set()
                    mem_targeting_scatter_paths[cur_scatter_path].add(cur_scatter_path)
                    assert origin.source_port in original_cc.inputs
                    assert (
                        cur_scatter_path not in required_scatter_paths
                        or required_scatter_paths[cur_scatter_path] == "memory"
                    ), f"Conflict scatter path usage for path {cur_scatter_path}"
                    required_scatter_paths[cur_scatter_path] = "memory"
                else:
                    assert (
                        origin.source_port in scatter_paths
                    ), f"Missing scatter path for port {origin.source_port} in gather."
                    cur_scatter_path = tuple(scatter_paths[origin.source_port][1])
                    assert (
                        cur_scatter_path not in required_scatter_paths
                        or required_scatter_paths[cur_scatter_path] == "through"
                    ), f"Conflict scatter path usage for path {cur_scatter_path}"
                    required_scatter_paths[cur_scatter_path] = "through"
                    if cur_scatter_path not in scatter_targeting_ports:
                        scatter_targeting_ports[cur_scatter_path] = gather_port
                    else:
                        copy_comp, new_in_port = copy_in_port(
                            gather_port, scatter_targeting_ports[cur_scatter_path]
                        )
                        compute_ops.append(copy_comp)
                        scatter_targeting_ports[cur_scatter_path] = new_in_port
        assert len(waiting_out_ports) == 0
        waiting_out_ports.append(gather_comp.get_port("o_0"))
        compute_ops.append(gather_comp)

    print("Scatter Targeting Ports:")
    for scatter_path, port in scatter_targeting_ports.items():
        print(f"  Scatter Path: {scatter_path}, Port: {port}")
    print("Memory Targeting Ports:")
    for mem_path, port in mem_targeting_ports.items():
        print(f"  Memory Path: {mem_path}, Port: {port}")

    fused_comp_col = ComponentCollection(
        components=compute_ops,
        inputs=list(mem_targeting_ports.values()) + list(scatter_targeting_ports.values()),
        outputs=waiting_out_ports,
    )

    from pathlib import Path

    output_dir = Path("output")
    dot_orig = visualize_components(str(fused_comp_col))
    dot_orig.render(output_dir / "fused_comp_col", view=False, format="png")
    print("Generated graph for FusedOpComponent as fused_comp_col.png.")

    fused_comp_col = delete_placeholder_components_pass(fused_comp_col)
    fused_comp = FusedOpComponent("fused_op", fused_comp_col)

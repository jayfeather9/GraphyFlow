from __future__ import annotations
import copy
import collections
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple, Set

from torch import scatter
from graphyflow.dataflow_ir_datatype import *
from graphyflow.dataflow_ir import *
from graphyflow.global_graph import GlobalGraph
from graphyflow.visualize_ir import visualize_components
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataOrigin:
    """Stores the origin of a data stream for analysis."""

    # The ultimate source: either a primary input port or a computed port.
    source_port: Port
    # The accumulated memory access path from the source.
    access_path: Tuple[Union[str, int], ...] = field(default_factory=tuple)


@dataclass
class RefactorResult:
    """Holds the result of refactoring a ComponentCollection."""

    comp_col: ComponentCollection
    output_mapping: Dict[Port, Port]


def _extract_subgraph_from_reduce(reduce_comp: ReduceComponent, subgraph_type: str) -> ComponentCollection:
    """
    Extracts a subgraph (key, transform, or unit_reduce) from a ReduceComponent
    and returns it as a new, self-contained ComponentCollection where external
    inputs are replaced by PlaceholderComponents, disconnect the i/o ports.
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
    actual_input_ports = []

    # The end component is the one connected to the end_port
    end_comp = end_port.connection.parent

    # Start forward traversal from the components connected to the entry ports
    for port in start_ports:
        assert port.connected
        actual_input_ports.append(port.connection)
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
    actual_subgraph_outputs = [end_port.connection]

    # Make the subgraph self-contained by replacing external inputs with placeholders
    subgraph_inputs = []
    subgraph_outputs = []

    # Replace each external connection with a placeholder
    for p_in in actual_input_ports:
        p_in.disconnect()
        placeholder = PlaceholderComponent(p_in.data_type)
        subgraph_comps.append(placeholder)
        placeholder.get_port("o_0").connect(p_in)
        subgraph_inputs.append(placeholder.get_port("i_0"))

    for p_out in actual_subgraph_outputs:
        p_out.disconnect()
        placeholder = PlaceholderComponent(p_out.data_type)
        subgraph_comps.append(placeholder)
        p_out.connect(placeholder.get_port("i_0"))
        subgraph_outputs.append(placeholder.get_port("o_0"))

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
    for subgraph_type in ["key", "transform", "unit_reduce"]:
        subgraph_cc = _extract_subgraph_from_reduce(reduce_comp, subgraph_type)

        print(f"\n--- Refactoring '{subgraph_type}' subgraph ---")
        # print(subgraph_cc)

        # Refactor the subgraph to separate memory access from computation
        refactor_result = refactor_to_memread_fusedop(subgraph_cc, global_graph)
        refactored_cc = refactor_result.comp_col
        assert len(refactor_result.output_mapping) == 1

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
            assert subgraph_type != "unit_reduce", "unit_reduce should not have memory access"
            if subgraph_type != "unit_reduce":
                for in_idx, base_type, path in mem_read_comp.access_pattern:
                    # Convert path list to tuple to make it hashable for the set
                    all_access_patterns.add((in_idx, base_type, tuple(path)))

        print(f"\n--- Refactored '{subgraph_type}' subgraph ---")
        print(refactored_cc)
        dot_refactored = visualize_components(str(refactored_cc))
        dot_refactored.render(f"output/{subgraph_type}_graph", view=False, format="png")
        print(f"Refactored subgraph visualized to output/{subgraph_type}_graph.png")

    # Convert set of tuples back to a list of lists for the MemoryReadComponent constructor
    results["unified_access_pattern"] = [
        (in_idx, base, list(path)) for in_idx, base, path in all_access_patterns
    ]

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

    # add a unused end marker to each unconnected scatter port that's not one of outputs
    new_components = set()
    for comp in live_components:
        if isinstance(comp, ScatterComponent):
            for p_out in comp.out_ports:
                if not p_out.connected and p_out not in outputs:
                    unused_end = UnusedEndMarkerComponent(p_out.data_type)
                    p_out.connect(unused_end.get_port("i_0"))
                    new_components.add(unused_end)
    live_components.update(new_components)

    # Filter the component list and find the new set of necessary inputs
    final_components = list(live_components)
    final_inputs = []
    for comp in final_components:
        for p_in in comp.in_ports:
            if not p_in.connected:
                final_inputs.append(p_in)

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
) -> RefactorResult:
    """
    Refactors a ComponentCollection into a MemoryReadComponent and a FusedOpComponent.
    This is the final, robust implementation that handles complex Tuple inputs
    by normalizing access paths and restructuring the input data flow.
    """
    # print("Original ComponentCollection to Refactor:")
    # print(original_cc)
    # ======================================================================== #
    #               PHASE 0: CHECKING                                          #
    # ======================================================================== #
    # check if all in & out ports are not connected
    for p_in in original_cc.inputs:
        assert not p_in.connected, f"Input port {p_in} should not be connected."
    for p_out in original_cc.outputs:
        assert not p_out.connected, f"Output port {p_out} should not be connected."

    # ======================================================================== #
    #               PHASE 1: DATA-FLOW ANALYSIS & ORIGIN TRACING               #
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
        elif isinstance(comp, UnusedEndMarkerComponent):
            origin = get_origin(p_in)
            assert origin is not None
            continue

        assert isinstance(comp, (UnaryOpComponent, BinOpComponent))

        compute_ops.append(comp)
        for p_out in comp.out_ports:
            port_origins[p_out] = DataOrigin(source_port=p_out)

    # ======================================================================== #
    #         PHASE 2: IDENTIFYING MEMORY & SCATTER ACCESS PATTERNS            #
    # ======================================================================== #

    scatter_paths = {}
    mem_paths = {}
    mem_patterns = set()

    for comp in compute_ops:
        # print(f"Compute Component: {comp})")
        for p_in in comp.in_ports:
            origin = port_origins.get(p_in.connection) if p_in.connection else port_origins.get(p_in)
            assert origin is not None, f"Missing origin for port {p_in} in component {comp}"
            # print(f"  Input Port: {p_in}, Origin: {origin}")
            if origin.source_port not in original_cc.inputs:
                assert len(origin.access_path) == 0
                continue
            if len(origin.access_path) == 0:
                continue
            # get the in_idx
            in_idx = original_cc.inputs.index(origin.source_port)
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
            scatter_paths[p_in] = (in_idx, origin.source_port, cur_scatter_path)
            print(f"    Scatter Path: {cur_scatter_path}")
            print(f"    Access Path: {cur_access_path}")
            # print(f"    Scatter Path: {cur_scatter_path}")
            if not is_simple_access and len(cur_access_path) > 0:
                mem_patterns.add((current_type.type_name, tuple(cur_access_path)))
                mem_paths[p_in] = (current_type.type_name, tuple(cur_access_path))
                # print(
                #     f"    Memory Access Pattern: Base={current_type.type_name}, Path={tuple(cur_access_path)}"
                # )

    # print(f"Port origins: {port_origins}")
    temp_out_subports = {}
    # print("Analyzing Output Ports:")
    analyze_out_datas = []
    # assert len(original_cc.outputs) == 1, "Only single output is supported now."
    for out_idx, p_out in enumerate(original_cc.outputs):
        sub_type = p_out.data_type
        assert isinstance(sub_type, ArrayType)
        origin = port_origins.get(p_out.connection) if p_out.connection else port_origins.get(p_out)
        if type(origin) is not list:
            sub_type = sub_type.type_
            for in_idx, ori_in_port in enumerate(original_cc.inputs):
                if origin.source_port == ori_in_port:
                    analyze_out_datas.append((out_idx, in_idx, 0, origin, sub_type))
        else:
            origins = origin
            assert isinstance(sub_type.type_, TupleType)
            sub_type = sub_type.type_.types[i]
            for i, origin in enumerate(origins):
                # print(f"Output Port: {p_out}, Origin: {origin}")
                assert origin is not None, f"Missing origin for output port {p_out}"
                for in_idx, ori_in_port in enumerate(original_cc.inputs):
                    if origin.source_port == ori_in_port:
                        analyze_out_datas.append((out_idx, in_idx, i, origin, sub_type))

    for analyze_out_data in analyze_out_datas:
        out_idx, in_idx, i, origin, sub_type = analyze_out_data
        p_out = original_cc.outputs[out_idx]
        temp_out_subports[(out_idx, i)] = Port(f"o_final_out_{i}", p_out.parent)
        temp_out_subports[(out_idx, i)].data_type = sub_type
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
        scatter_paths[temp_out_subports[(out_idx, i)]] = (in_idx, origin.source_port, cur_scatter_path)
        print(f"    in_idx: {in_idx}, subport idx: {i}")
        print(f"    Scatter Path: {cur_scatter_path}")
        print(f"    Access Path: {cur_access_path}")
        access_len_without_g = len(cur_access_path)
        for path_idx, p in enumerate(cur_access_path):
            if isinstance(p, str) and p.startswith("_g"):
                access_len_without_g = path_idx
                break
        if not is_simple_access and access_len_without_g > 0:
            mem_patterns.add((current_type.type_name, tuple(cur_access_path)))
            mem_paths[temp_out_subports[(out_idx, i)]] = (current_type.type_name, tuple(cur_access_path))
            # print(
            #     f"    Memory Access Pattern: Base={current_type.type_name}, Path={tuple(cur_access_path)}"
            # )

    # ======================================================================== #
    #             PHASE 3: BUILDING THE FUSED OPERATION COMPONENT              #
    # ======================================================================== #

    # print(f"Identified Memory Access Paths: {mem_paths}")
    # print(f"Scatter Paths: {scatter_paths}")

    mem_targeting_ports = {}
    mem_source = {}
    mem_targeting_scatter_paths = {}
    scatter_targeting_ports = {}
    scatter_source = {}
    waiting_out_ports = []

    def copy_in_port(old_port: Port, new_port: Port) -> CopyComponent:
        # print(f"Copying port {old_port} to new port {new_port}")
        copy_comp = CopyComponent(old_port.data_type)
        copy_comp.get_port("o_0").connect(old_port)
        copy_comp.get_port("o_1").connect(new_port)
        return copy_comp, copy_comp.get_port("i_0")

    added_comps = []
    for comp in compute_ops:
        waiting_out_ports.extend(comp.out_ports)
        if isinstance(comp, ConstantComponent):
            continue
        for p_in in comp.in_ports:
            # print(f"Processing input port {p_in} of component {comp}")
            origin = get_origin(p_in)
            assert origin is not None, f"Missing origin for port {p_in} in component {comp}"
            # source is directly from input port and need memory access
            if p_in in mem_paths:
                mem_path = mem_paths[p_in]
                assert p_in in scatter_paths, f"Missing scatter path for port {p_in} in component {comp}"
                in_idx = scatter_paths[p_in][0]
                cur_scatter_path = tuple(scatter_paths[p_in][2])
                p_in.disconnect()
                if (in_idx, mem_path) not in mem_targeting_ports:
                    mem_targeting_ports[(in_idx, mem_path)] = p_in
                    mem_source[(in_idx, mem_path)] = origin.source_port
                else:
                    copy_comp, new_in_port = copy_in_port(p_in, mem_targeting_ports[(in_idx, mem_path)])
                    added_comps.append(copy_comp)
                    mem_targeting_ports[(in_idx, mem_path)] = new_in_port
                if (in_idx, mem_path) in mem_targeting_scatter_paths:
                    assert (
                        mem_targeting_scatter_paths[(in_idx, mem_path)] == cur_scatter_path
                    ), f"Conflict scatter paths for memory access pattern {mem_path}"
                mem_targeting_scatter_paths[(in_idx, mem_path)] = cur_scatter_path
                assert origin.source_port in original_cc.inputs
                continue
            # source is directly from input port but need scatter
            if p_in in scatter_paths:
                assert origin.source_port in original_cc.inputs
                in_idx = scatter_paths[p_in][0]
                cur_scatter_path = tuple(scatter_paths[p_in][2])
                p_in.disconnect()
                if (in_idx, cur_scatter_path) not in scatter_targeting_ports:
                    scatter_targeting_ports[(in_idx, cur_scatter_path)] = p_in
                    scatter_source[(in_idx, cur_scatter_path)] = origin.source_port
                else:
                    copy_comp, new_in_port = copy_in_port(
                        p_in, scatter_targeting_ports[(in_idx, cur_scatter_path)]
                    )
                    added_comps.append(copy_comp)
                    scatter_targeting_ports[(in_idx, cur_scatter_path)] = new_in_port
                continue
            # source is directly from input port but no need memory or scatter
            if origin.source_port in original_cc.inputs:
                if p_in != origin.source_port:
                    p_in.disconnect()
                continue
            # source is output of other comps
            if origin.source_port.connected and origin.source_port in waiting_out_ports:
                waiting_out_ports.remove(origin.source_port)
                if p_in.connection != origin.source_port:
                    p_in.disconnect()
                    origin.source_port.disconnect()
                    origin.source_port.connect(p_in)
                continue
            assert False, f"Port {p_in} in component {comp} should have been handled."
    compute_ops.extend(added_comps)

    # print(f"waiting out ports before gather: {waiting_out_ports}")

    def trans_spe(t: DfirType) -> DfirType:
        if isinstance(t, SpecialType):
            return SpecialIdType.from_spe(t)
        return t

    def manage_paths(query_port: Port, target_port: Port):
        nonlocal mem_paths, scatter_paths, mem_targeting_ports, mem_targeting_scatter_paths, scatter_targeting_ports, waiting_out_ports, compute_ops
        if query_port in mem_paths:
            mem_path = mem_paths[query_port]
            assert query_port in scatter_paths, f"Missing scatter path for port {query_port}."
            in_idx = scatter_paths[query_port][0]
            cur_scatter_path = tuple(scatter_paths[query_port][2])
            origin_port = scatter_paths[query_port][1]
            if (in_idx, mem_path) not in mem_targeting_ports:
                mem_targeting_ports[(in_idx, mem_path)] = target_port
                mem_source[(in_idx, mem_path)] = origin_port
            else:
                copy_comp, new_in_port = copy_in_port(target_port, mem_targeting_ports[(in_idx, mem_path)])
                compute_ops.append(copy_comp)
                mem_targeting_ports[(in_idx, mem_path)] = new_in_port
            if (in_idx, mem_path) in mem_targeting_scatter_paths:
                assert (
                    mem_targeting_scatter_paths[(in_idx, mem_path)] == cur_scatter_path
                ), f"Conflict scatter paths for memory access pattern {mem_path}"
            mem_targeting_scatter_paths[(in_idx, mem_path)] = cur_scatter_path
            assert origin.source_port in original_cc.inputs
        else:
            assert query_port in scatter_paths, f"Missing scatter path for port {query_port} in gather."
            cur_scatter_path = tuple(scatter_paths[query_port][2])
            origin_port = scatter_paths[query_port][1]
            in_idx = scatter_paths[query_port][0]
            if (in_idx, cur_scatter_path) not in scatter_targeting_ports:
                scatter_targeting_ports[(in_idx, cur_scatter_path)] = target_port
                scatter_source[(in_idx, cur_scatter_path)] = origin_port
            else:
                copy_comp, new_in_port = copy_in_port(
                    target_port, scatter_targeting_ports[(in_idx, cur_scatter_path)]
                )
                compute_ops.append(copy_comp)
                scatter_targeting_ports[(in_idx, cur_scatter_path)] = new_in_port

    output_port_map = {}
    # assert len(original_cc.outputs) == 1
    for out_idx, p_out in enumerate(original_cc.outputs):
        # print(f"Getting origin for output port {p_out}")
        out_origins = get_origin(p_out)
        if type(out_origins) is list:
            resorted_origins = [None for _ in out_origins]
            for origin in out_origins:
                index = int(origin.access_path[-1][2:])
                resorted_origins[index] = origin
            out_origins = resorted_origins
            gather_types = [ArrayType(trans_spe(g_type)) for g_type in p_out.data_type.type_.types]
            gather_comp = GatherComponent(gather_types)
            for i, origin in enumerate(out_origins):
                gather_port = gather_comp.get_port(f"i_{i}")
                if origin.source_port in waiting_out_ports:
                    waiting_out_ports.remove(origin.source_port)
                    origin.source_port.disconnect()
                    origin.source_port.connect(gather_port)
                else:
                    assert origin.source_port in original_cc.inputs
                    manage_paths(temp_out_subports[(out_idx, i)], gather_port)
            waiting_out_ports.append(gather_comp.get_port("o_0"))
            output_port_map[p_out] = gather_comp.get_port("o_0")
            compute_ops.append(gather_comp)
        else:
            origin = out_origins
            if origin.source_port not in waiting_out_ports:
                assert origin.source_port in original_cc.inputs
                placeholder = PlaceholderComponent(original_cc.outputs[0].data_type)
                target_port = placeholder.get_port("i_0")
                manage_paths(temp_out_subports[(out_idx, 0)], target_port)
                waiting_out_ports.append(placeholder.get_port("o_0"))
                compute_ops.append(placeholder)
                output_port_map[p_out] = placeholder.get_port("o_0")
            elif origin.source_port != p_out:
                origin.source_port.disconnect()
                output_port_map[p_out] = origin.source_port
            else:
                output_port_map[p_out] = p_out

    # print("Scatter Targeting Ports:")
    # for scatter_path, port in scatter_targeting_ports.items():
    #     print(f"  Scatter Path: {scatter_path}, Port: {port}")
    # print("Memory Targeting Ports:")
    # for mem_path, port in mem_targeting_ports.items():
    #     print(f"  Memory Path: {mem_path}, Port: {port}")

    # extract "_g" from mem_paths
    port_to_gather_paths = {}
    new_mem_targeting_ports = {}
    new_mem_targeting_scatter_paths = {}
    for fused_access_pattern, port in mem_targeting_ports.items():
        base_type, path = fused_access_pattern[1]
        in_idx = fused_access_pattern[0]
        gather_path = ()
        old_scatter_path = mem_targeting_scatter_paths[fused_access_pattern]
        # everything after "_g" is useless for memory accesss
        for i, p in enumerate(path):
            if isinstance(p, str) and p.startswith("_g"):
                gather_path = path[i:]
                path = path[:i]
                break
        if (in_idx, (base_type, path)) in new_mem_targeting_ports:
            copy_comp, new_in_port = copy_in_port(port, new_mem_targeting_ports[(in_idx, (base_type, path))])
            compute_ops.append(copy_comp)
            new_mem_targeting_ports[(in_idx, (base_type, path))] = new_in_port
        else:
            new_mem_targeting_ports[(in_idx, (base_type, path))] = port
        new_mem_targeting_scatter_paths[(in_idx, (base_type, path))] = old_scatter_path
        port_to_gather_paths[port] = gather_path
    mem_targeting_ports = new_mem_targeting_ports
    mem_targeting_scatter_paths = new_mem_targeting_scatter_paths

    compute_ops, input_ports = _dead_code_elimination(compute_ops, waiting_out_ports)
    compute_ops = _simplify_redundant_copies(compute_ops)

    # print(compute_ops, input_ports, waiting_out_ports)

    fused_comp_col = ComponentCollection(
        components=compute_ops,
        inputs=input_ports,
        outputs=waiting_out_ports,
    )
    # print(f"input ports: {input_ports}")
    from graphyflow.passes import delete_placeholder_components_pass

    fused_comp_col = delete_placeholder_components_pass(fused_comp_col)

    assert compute_ops, "No compute operations remain after dead code elimination."

    from pathlib import Path

    output_dir = Path("output")
    dot_orig = visualize_components(str(fused_comp_col))
    dot_orig.render(output_dir / "fused_comp_col", view=False, format="png")
    print("Generated graph for FusedOpComponent as fused_comp_col.png.")

    fused_comp = FusedOpComponent("fused_op", fused_comp_col)

    # update the actual output port map
    for orig_out_port in output_port_map:
        fused_out_port = output_port_map[orig_out_port]
        output_port_map[orig_out_port] = fused_comp.port_mapping[fused_out_port.readable_id]

    # update port to fused outer ports
    for fused_access_pattern, ori_port in mem_targeting_ports.items():
        new_port = fused_comp.port_mapping[ori_port.readable_id]
        mem_targeting_ports[fused_access_pattern] = new_port
    for fused_scatter_path, ori_port in scatter_targeting_ports.items():
        new_port = fused_comp.port_mapping[ori_port.readable_id]
        scatter_targeting_ports[fused_scatter_path] = new_port

    access_patterns = []
    output_types = {}
    for access_pattern, port in mem_targeting_ports.items():
        in_idx, type_and_path = access_pattern
        base_type, path = type_and_path
        # e.g., ("edge", ["weight_tuple", 1]) -> "o_edge_weight_tuple_1"
        path_str = "_".join(map(str, path))
        port_name = f"o_{in_idx}_{base_type}_{path_str}"
        access_patterns.append((in_idx, base_type, path))
        assert isinstance(port.data_type, ArrayType)
        output_types[port_name] = port.data_type

    mem_read_comp = MemoryReadComponent(
        access_pattern=access_patterns,
        output_types=output_types,
        parallel=True,
    )

    mem_read_comp.visualize_access_tree()

    # print(f"port_to_gather_paths: {port_to_gather_paths}")
    # print(f"mem_targeting_ports: {mem_targeting_ports}")
    # print(f"mem_targeting_scatter_paths: {mem_targeting_scatter_paths}")
    # print(f"scatter_targeting_ports: {scatter_targeting_ports}")

    # ======================================================================== #
    #             PHASE 4: ASSEMBLING THE FINAL REFACTORED GRAPH               #
    # ======================================================================== #

    def copy_out_port(out_port: Port, new_in_port: Port):
        original_connection = out_port.connection
        out_port.disconnect()
        copy_comp = CopyComponent(out_port.data_type)
        copy_comp.get_port("i_0").connect(out_port)
        new_in_port.connect(copy_comp.get_port("o_1"))
        original_connection.connect(copy_comp.get_port("o_0"))
        return copy_comp

    # print(f"Memory Read Component: {mem_read_comp}")
    # print(f"Fused Operation Component: {fused_comp}")

    # connect mem out ports to fused in ports
    for access_pattern, fused_in_port in mem_targeting_ports.items():
        in_idx, type_and_path = access_pattern
        base_type, path = type_and_path
        mem_out_pname = mem_read_comp.pattern_to_pname[access_pattern]
        mem_out_port = mem_read_comp.get_port(mem_out_pname)
        fused_in_port.connect(mem_out_port)

    mem_scatter_paths = set((fused[0], path) for fused, path in mem_targeting_scatter_paths.items())
    scatter_only_paths = set((in_idx, path) for in_idx, path in scatter_targeting_ports.keys())
    scatter_paths = mem_scatter_paths.union(scatter_only_paths)
    # print(scatter_paths, len(scatter_paths))
    # print(f"All required scatter paths: {scatter_paths}")
    components = [mem_read_comp, fused_comp]
    if len(scatter_paths) == 0 or (len(scatter_paths) == 1 and list(scatter_paths)[0] == (0, ())):
        assert all(
            len(p) == 0 for p in scatter_targeting_ports.keys()
        ), "No scatter_only ports should be present for empty scatter path."
        assert (
            len(mem_read_comp.in_ports) <= 1
        ), "Only one or no input should be present for empty scatter path."
        if len(mem_read_comp.in_ports) == 1:
            base_type = list(mem_targeting_ports.keys())[0][0]
            assert all(
                base_type == p[0] for p in mem_targeting_ports.keys()
            ), "All memory access patterns should share the same base type for empty scatter path."
    else:
        # assert all scatter paths depth=1
        assert all(len(p[1]) == 1 for p in scatter_paths), "Currently only support depth=1 scatter paths."
        # assert only one input of original_cc
        # assert len(original_cc.inputs) == 1, "Currently only support one input."
        scatter_comps_for_input = []
        for i, ori_in_port in enumerate(original_cc.inputs):
            input_datatype = ori_in_port.data_type
            assert isinstance(input_datatype, ArrayType)
            assert isinstance(input_datatype.type_, TupleType)
            input_datatype = input_datatype.type_.types
            input_datatype = [
                t if not isinstance(t, SpecialType) else SpecialIdType.from_spe(t) for t in input_datatype
            ]
            input_datatype = ArrayType(TupleType(input_datatype))
            scatter_comp = ScatterComponent(input_datatype)
            components.append(scatter_comp)
            scatter_comps_for_input.append(scatter_comp)

        # handle the scatter_only ports
        for fused_scatter_path, scatter_port in scatter_targeting_ports.items():
            in_idx, scatter_path = fused_scatter_path
            index = scatter_path[0]
            scatter_comp = scatter_comps_for_input[in_idx]
            scatter_out_port = scatter_comp.get_port(f"o_{index}")
            if scatter_out_port.connected:
                copy_comp = copy_out_port(scatter_out_port, scatter_port)
                components.append(copy_comp)
            else:
                scatter_port.connect(scatter_out_port)

        # handle the mem_targeting_ports
        connected_base_types = set()
        for fused_access_pattern, scatter_path in mem_targeting_scatter_paths.items():
            # print(f"Handling mem targeting scatter path: {fused_access_pattern} -> {scatter_path}")
            in_idx, access_pattern = fused_access_pattern
            base_type = access_pattern[0]
            if (in_idx, base_type) in connected_base_types:
                continue
            index = scatter_path[0]
            scatter_comp = scatter_comps_for_input[in_idx]
            scatter_out_port = scatter_comp.get_port(f"o_{index}")
            connected_base_types.add((in_idx, base_type))
            mem_port = mem_read_comp.get_port(f"i_{in_idx}_{base_type}_id")
            if scatter_out_port.connected:
                copy_comp = copy_out_port(scatter_out_port, mem_port)
                components.append(copy_comp)
            else:
                mem_port.connect(scatter_out_port)

    # print(scatter_comp)
    # print(fused_comp)
    # print(mem_read_comp)

    if len(mem_read_comp.in_ports) == 0:
        assert len(mem_read_comp.out_ports) == 0, "If no memory access, no output should be present."
        # no memory access, remove the mem_read_comp
        components.remove(mem_read_comp)

    out_ports = fused_comp.out_ports

    components, in_ports = _dead_code_elimination(components, out_ports)
    components = _simplify_redundant_copies(components)
    print(components, in_ports, out_ports)

    final_cc = ComponentCollection(
        components=components,
        inputs=in_ports,
        outputs=out_ports,
    )
    return RefactorResult(comp_col=final_cc, output_mapping=output_port_map)

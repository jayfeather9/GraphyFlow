import graphyflow.dataflow_ir as dfir
from typing import List, Tuple, Set, Dict, Any, Union
from dataclasses import dataclass
import collections
import copy
from graphyflow.dataflow_ir_utils import _extract_subgraph_from_reduce, refactor_to_memread_fusedop
from graphyflow.reduce_analysis import MemoryAccessInfo, SubgraphAnalysisResult, ReduceAnalysisResult
from graphyflow.global_graph import GlobalGraph
from graphyflow.visualize_ir import visualize_components


def delete_placeholder_components_pass(
    comp_col: dfir.ComponentCollection,
) -> dfir.ComponentCollection:
    # if all comp in comp_col are PlaceholderComponent, just return it
    # because we can't create an empty ComponentCollection
    if all(isinstance(comp, dfir.PlaceholderComponent) for comp in comp_col.components):
        print("Warning: All components are PlaceholderComponents. Returning the original collection.")
        return comp_col
    components_to_keep: List[dfir.Component] = []

    for comp in comp_col.components:
        if isinstance(comp, dfir.PlaceholderComponent):
            ph_input_port = comp.get_port("i_0")
            ph_output_port = comp.get_port("o_0")

            upstream_connected_port = ph_input_port.connection
            downstream_connected_port = ph_output_port.connection

            if upstream_connected_port is not None:
                upstream_connected_port.disconnect()
            else:
                assert ph_input_port in comp_col.inputs
                assert downstream_connected_port is not None, "Empty collection found."
                comp_col.inputs.remove(ph_input_port)
                comp_col.inputs.append(downstream_connected_port)

            if downstream_connected_port is not None:
                downstream_connected_port.disconnect()
            else:
                assert ph_output_port in comp_col.outputs
                assert upstream_connected_port is not None, "Empty collection found."
                comp_col.outputs.remove(ph_output_port)
                comp_col.outputs.append(upstream_connected_port)

            if upstream_connected_port is not None and downstream_connected_port is not None:
                upstream_connected_port.connect(downstream_connected_port)

        else:
            components_to_keep.append(comp)

    comp_col = dfir.ComponentCollection(components_to_keep, comp_col.inputs, comp_col.outputs)
    comp_col.update_ports()

    return comp_col


def remove_io_comp_pass(comp_col: dfir.ComponentCollection) -> dfir.ComponentCollection:
    """Remove IOComponents, and make corresponding ports inputs ports of the collection."""
    components_to_keep: List[dfir.Component] = []

    for comp in comp_col.components:
        if isinstance(comp, dfir.IOComponent):
            assert len(comp.in_ports) == 0 or len(comp.out_ports) == 1
            p_out = comp.out_ports[0]
            assert p_out not in comp_col.outputs
            comp_col.inputs.append(p_out.connection)
            p_out.disconnect()
        else:
            components_to_keep.append(comp)

    comp_col = dfir.ComponentCollection(components_to_keep, comp_col.inputs, comp_col.outputs)
    comp_col.update_ports()

    return comp_col


def _build_context_for_reduce_analysis(reduce_comp: dfir.ReduceComponent) -> dfir.ComponentCollection:
    """
    Constructs a temporary, minimal, and valid ComponentCollection containing
    a ReduceComponent and its subgraph, suitable for analysis passes.
    """
    # Deepcopy to avoid modifying the original component
    rc_copy = copy.deepcopy(reduce_comp)

    components = [rc_copy]
    q = collections.deque()

    # Find all connected subgraph components
    for port_name in [
        "o_reduce_key_in",
        "o_reduce_transform_in",
        "o_reduce_unit_start_0",
        "o_reduce_unit_start_1",
    ]:
        port = rc_copy.get_port(port_name)
        if port.connected:
            q.append(port.connection.parent)

    visited_ids = {c.readable_id for c in q}
    head = 0
    while head < len(q):
        comp = q[head]
        head += 1
        if comp not in components:
            components.append(comp)
        for p_out in comp.out_ports:
            if p_out.connected and p_out.connection.parent.readable_id not in visited_ids:
                if not isinstance(p_out.connection.parent, dfir.ReduceComponent):
                    q.append(p_out.connection.parent)
                    visited_ids.add(p_out.connection.parent.readable_id)

    # The collection's I/O are the main I/O of the ReduceComponent
    return dfir.ComponentCollection(
        components=components, inputs=[rc_copy.get_port("i_0")], outputs=[rc_copy.get_port("o_0")]
    )


def _analyze_refactored_cc(
    refactored_cc: dfir.ComponentCollection,
) -> Tuple[List[MemoryAccessInfo], SubgraphAnalysisResult]:
    """
    Analyzes a ComponentCollection that has already been processed by
    refactor_to_memread_fusedop to extract all necessary pathing information.
    This version correctly handles multiple ScatterComponents.
    """
    # --- PHASE 1: Find all key components in the refactored graph ---
    fused_op = next((c for c in refactored_cc.components if isinstance(c, dfir.FusedOpComponent)), None)
    mem_read_op = next((c for c in refactored_cc.components if isinstance(c, dfir.MemoryReadComponent)), None)
    # Find all scatter components, not just the first one.
    scatter_ops = [c for c in refactored_cc.components if isinstance(c, dfir.ScatterComponent)]
    assert fused_op is not None, "Refactored graph must contain a FusedOpComponent."

    mem_accesses: List[MemoryAccessInfo] = []
    provenance: Dict[dfir.Port, Union[MemoryAccessInfo, Tuple[int, tuple], str]] = {}

    # Map each scatter component to the index of the subgraph input that feeds it.
    scatter_to_input_idx_map = {
        sc: refactored_cc.inputs.index(sc.in_ports[0])
        for sc in scatter_ops
        if sc.in_ports[0] in refactored_cc.inputs
    }

    # --- PHASE 2: Analyze memory accesses by tracing from MemoryRead to Scatter ---
    if mem_read_op:
        for in_idx_mem, base_type, mem_path in mem_read_op.access_pattern:
            p_in_mem = mem_read_op.get_port(f"i_{in_idx_mem}_{base_type}_id")

            scatter_path_to_base = tuple()
            # The `in_idx` for the MemoryAccessInfo should be the *subgraph's* input index,
            # not the MemoryReadComponent's relative input index.
            origin_subgraph_in_idx = -1

            if p_in_mem.connected:
                upstream_scatter = p_in_mem.connection.parent
                if isinstance(upstream_scatter, dfir.ScatterComponent):
                    p_out_scatter = p_in_mem.connection
                    # The scatter path is the index of the output port on its Scatter component.
                    scatter_path_to_base = (upstream_scatter.out_ports.index(p_out_scatter),)
                    # Find the original subgraph input index that feeds this scatter.
                    origin_subgraph_in_idx = scatter_to_input_idx_map.get(upstream_scatter, -1)

            # Determine output type for this specific access
            p_out_name = mem_read_op.pattern_to_pname[(in_idx_mem, (base_type, tuple(mem_path)))]
            output_type = mem_read_op.get_port(p_out_name).data_type

            info = MemoryAccessInfo(
                in_idx=origin_subgraph_in_idx,
                base_type=base_type,
                mem_path=tuple(mem_path),
                scatter_path_to_base=scatter_path_to_base,
                output_type=output_type,
            )
            mem_accesses.append(info)

    # --- PHASE 3: Determine the provenance of each FusedOp input ---
    for p_in_fused in fused_op.in_ports:
        if not p_in_fused.connected:
            # Case: The FusedOp input is a main input of the refactored graph.
            in_idx = refactored_cc.inputs.index(p_in_fused)
            provenance[p_in_fused] = (in_idx, tuple())
            continue

        upstream_comp = p_in_fused.connection.parent

        if upstream_comp == mem_read_op:
            # Input comes from memory. Find the matching MemoryAccessInfo object.
            p_out_mem = p_in_fused.connection
            pattern_key = next(k for k, v in mem_read_op.pattern_to_pname.items() if v == p_out_mem.name)
            mem_info = next(info for info in mem_accesses if info.mem_path == tuple(pattern_key[1][1]))
            provenance[p_in_fused] = mem_info

        elif isinstance(upstream_comp, dfir.ScatterComponent):
            # Input is a direct passthrough from a scatter.
            p_out_scatter = p_in_fused.connection
            in_idx = scatter_to_input_idx_map.get(upstream_comp, -1)
            scatter_path = (upstream_comp.out_ports.index(p_out_scatter),)
            provenance[p_in_fused] = (in_idx, scatter_path)

        elif isinstance(upstream_comp, dfir.ConstantComponent):
            provenance[p_in_fused] = "constant"

        else:
            raise ValueError(f"Unhandled FusedOp input source: {type(upstream_comp)}")

    analysis = SubgraphAnalysisResult(
        refactored_fused_op=fused_op, input_provenance=provenance, full_subgraph=refactored_cc
    )
    return mem_accesses, analysis


def analyze_reduce_comp(
    comp_col: dfir.ComponentCollection, g: GlobalGraph
) -> Dict[str, ReduceAnalysisResult]:
    """
    Analyzes all ReduceComponents within a ComponentCollection and produces a
    detailed refactoring plan for each, without modifying the graph.

    This implementation analyzes the graph *after* it has been refactored
    to deduce all required pathing and memory access information.
    """
    comp_col_copy = copy.deepcopy(comp_col)
    analysis_results: Dict[str, ReduceAnalysisResult] = {}
    reduce_components = [c for c in comp_col_copy.components if isinstance(c, dfir.ReduceComponent)]

    for reduce_comp in reduce_components:
        # --- PHASE 1: Refactor each subgraph individually ---
        key_subgraph_orig = _extract_subgraph_from_reduce(reduce_comp, "key")
        key_refactored_cc = refactor_to_memread_fusedop(key_subgraph_orig, g).comp_col

        transform_subgraph_orig = _extract_subgraph_from_reduce(reduce_comp, "transform")
        transform_refactored_cc = refactor_to_memread_fusedop(transform_subgraph_orig, g).comp_col

        unit_subgraph_orig = _extract_subgraph_from_reduce(reduce_comp, "unit_reduce")
        unit_refactored_cc = refactor_to_memread_fusedop(unit_subgraph_orig, g).comp_col
        assert not any(isinstance(c, dfir.MemoryReadComponent) for c in unit_refactored_cc.components)

        for subgraph_type, refactored_cc in [
            ("key", key_refactored_cc),
            ("transform", transform_refactored_cc),
            ("unit_reduce", unit_refactored_cc),
        ]:
            print(f"\n--- Refactored '{subgraph_type}' subgraph ---")
            print(refactored_cc)
            dot_refactored = visualize_components(str(refactored_cc))
            dot_refactored.render(f"output/{subgraph_type}_graph", view=False, format="png")
            print(f"Refactored subgraph visualized to output/{subgraph_type}_graph.png")

        # --- PHASE 2: Analyze the refactored subgraphs to get paths and provenance ---
        key_mem_accesses, key_analysis = _analyze_refactored_cc(key_refactored_cc)
        transform_mem_accesses, transform_analysis = _analyze_refactored_cc(transform_refactored_cc)
        unit_mem_accesses, unit_analysis = _analyze_refactored_cc(unit_refactored_cc)
        assert not unit_mem_accesses, "unit_reduce subgraph should not perform memory access."

        # --- PHASE 3: Consolidate results and formulate final plan ---
        # Create a unique set of all memory accesses required
        consolidated_mem_access = list({info for info in key_mem_accesses + transform_mem_accesses})

        # The restructuring plan describes what the new, unified input stream should look like
        restructured_plan = {}
        for info in consolidated_mem_access:
            # Key describes the data, value describes its origin
            plan_key = f"mem_{info.base_type}_{'_'.join(map(str, info.mem_path))}"
            restructured_plan[plan_key] = info

        # A full implementation would also add passthrough paths to the plan
        # For now, this is sufficient to verify the memory analysis.

        analysis_results[reduce_comp.readable_id] = ReduceAnalysisResult(
            key_analysis=key_analysis,
            transform_analysis=transform_analysis,
            unit_analysis=unit_analysis,
            consolidated_mem_access=consolidated_mem_access,
            restructured_input_plan=restructured_plan,
        )

    print(f"Reduce analysis complete. Found and analyzed {len(analysis_results)} ReduceComponents.")
    return analysis_results


def optimize_reduce_comp(reduce_comp: dfir.ReduceComponent, g: GlobalGraph) -> dfir.ComponentCollection:
    """
    Takes a single ReduceComponent and returns a new, self-contained, and
    optimized ComponentCollection where memory accesses have been hoisted out.

    Args:
        reduce_comp: The specific ReduceComponent instance to optimize. Its i_0
                     and o_0 ports must be disconnected.
        g: The GlobalGraph context for type information.

    Returns:
        A new, fully-connected ComponentCollection representing the optimized logic.
    """
    # --- PHASE 0: Pre-condition check ---
    assert not reduce_comp.get_port("i_0").connected, "Input `reduce_comp` i_0 port must be disconnected."
    assert not reduce_comp.get_port("o_0").connected, "Input `reduce_comp` o_0 port must be disconnected."

    # --- PHASE 1: Analyze the ReduceComponent to get an optimization plan ---
    temp_analysis_context = _build_context_for_reduce_analysis(reduce_comp)
    analysis_results = analyze_reduce_comp(temp_analysis_context, g)
    analysis = analysis_results[reduce_comp.readable_id]

    final_components = []
    provenance_to_ext_port_map: Dict[Any, dfir.Port] = {}

    # --- PHASE 2: Create external MemoryRead and Scatter components ---
    mem_read_comp = None
    if analysis.consolidated_mem_access:
        mem_access_list = analysis.consolidated_mem_access
        access_pattern = [(info.in_idx, info.base_type, list(info.mem_path)) for info in mem_access_list]
        output_types = {}
        for info in mem_access_list:
            path_str = "_".join(map(str, info.mem_path))
            port_name = f"o_{info.in_idx}_{info.base_type}_{path_str}"
            output_types[port_name] = info.output_type

        mem_read_comp = dfir.MemoryReadComponent(
            access_pattern=access_pattern, output_types=output_types, parallel=True
        )
        final_components.append(mem_read_comp)

    scatter_comp = None
    needs_scatter = any(info.scatter_path_to_base for info in analysis.consolidated_mem_access)
    all_provenances = list(analysis.key_analysis.input_provenance.values()) + list(
        analysis.transform_analysis.input_provenance.values()
    )
    if not needs_scatter:
        for prov in all_provenances:
            if isinstance(prov, tuple) and prov[1]:
                needs_scatter = True
                break

    if needs_scatter:
        original_input_type = reduce_comp.get_port("i_0").data_type
        scatter_comp = dfir.ScatterComponent(original_input_type)
        final_components.insert(0, scatter_comp)

    # --- PHASE 3: Reconstruct the ReduceComponent and its subgraphs ---
    modified_reduce = copy.deepcopy(reduce_comp)
    final_components.append(modified_reduce)

    # 3a. Add and wire Key/Transform FusedOps
    key_fused_op = analysis.key_analysis.refactored_fused_op
    transform_fused_op = analysis.transform_analysis.refactored_fused_op
    final_components.extend([key_fused_op, transform_fused_op])

    for port in [
        modified_reduce.get_port("i_reduce_key_out"),
        modified_reduce.get_port("i_reduce_transform_out"),
        modified_reduce.get_port("o_reduce_key_in"),
        modified_reduce.get_port("o_reduce_transform_in"),
    ]:
        if port.connected:
            port.disconnect()

    if key_fused_op.get_port("o_0").connected:
        key_fused_op.get_port("o_0").disconnect()
    key_fused_op.get_port("o_0").connect(modified_reduce.get_port("i_reduce_key_out"))
    if transform_fused_op.get_port("o_0").connected:
        transform_fused_op.get_port("o_0").disconnect()
    transform_fused_op.get_port("o_0").connect(modified_reduce.get_port("i_reduce_transform_out"))

    data_requirements = collections.defaultdict(list)
    for p_in, prov in analysis.key_analysis.input_provenance.items():
        if prov != "constant":
            data_requirements[prov].append(p_in)
    for p_in, prov in analysis.transform_analysis.input_provenance.items():
        if prov != "constant":
            data_requirements[prov].append(p_in)

    for i, (provenance, dest_ports) in enumerate(data_requirements.items()):
        port_name_base = f"data_{i}"
        data_type = dest_ports[0].data_type

        # Determine the correct group for the internal output port based on
        # the FusedOp it connects to.
        first_dest_parent = dest_ports[0].parent
        if first_dest_parent == key_fused_op:
            out_port_group = "key"
        elif first_dest_parent == transform_fused_op:
            out_port_group = "transform"
        else:
            # This case should not be reached in this logic.
            raise ValueError("Destination of new port is not key or transform FusedOp.")

        ext_in_port, int_out_port = modified_reduce._add_io_port_pair(
            in_group="global", out_group=out_port_group, name_base=port_name_base, data_type=data_type
        )
        provenance_to_ext_port_map[provenance] = ext_in_port

        current_source_port = int_out_port
        for i_dest, dest_port in enumerate(dest_ports):
            if dest_port.connected:
                dest_port.disconnect()
            if i_dest < len(dest_ports) - 1:
                copy_comp = dfir.CopyComponent(data_type)
                final_components.append(copy_comp)
                current_source_port.connect(copy_comp.get_port("i_0"))
                copy_comp.get_port("o_0").connect(dest_port)
                current_source_port = copy_comp.get_port("o_1")
            else:
                current_source_port.connect(dest_port)

    # 3b. Inline and wire the Unit Reduce subgraph
    unit_refactored_cc = analysis.unit_analysis.full_subgraph
    assert len(unit_refactored_cc.inputs) == 2, "Unit reduce subgraph must have 2 inputs"
    assert len(unit_refactored_cc.outputs) == 1, "Unit reduce subgraph must have 1 output"
    final_components.extend(unit_refactored_cc.components)

    if modified_reduce.get_port("o_reduce_unit_start_0").connected:
        modified_reduce.get_port("o_reduce_unit_start_0").disconnect()
    modified_reduce.get_port("o_reduce_unit_start_0").connect(unit_refactored_cc.inputs[0])

    if modified_reduce.get_port("o_reduce_unit_start_1").connected:
        modified_reduce.get_port("o_reduce_unit_start_1").disconnect()
    modified_reduce.get_port("o_reduce_unit_start_1").connect(unit_refactored_cc.inputs[1])

    if modified_reduce.get_port("i_reduce_unit_end").connected:
        modified_reduce.get_port("i_reduce_unit_end").disconnect()
    unit_refactored_cc.outputs[0].connect(modified_reduce.get_port("i_reduce_unit_end"))

    # 3c. Clean up obsolete ports from ReduceComponent
    modified_reduce._remove_port_by_name("i_0")
    modified_reduce._remove_port_by_name("o_reduce_key_in")
    modified_reduce._remove_port_by_name("o_reduce_transform_in")

    # --- PHASE 4: Final External Wiring ---
    source_to_consumers_map = collections.defaultdict(list)

    for provenance, reduce_in_port in provenance_to_ext_port_map.items():
        if isinstance(provenance, MemoryAccessInfo):
            info = provenance
            path_str = "_".join(map(str, info.mem_path))
            port_name = f"o_{info.in_idx}_{info.base_type}_{path_str}"
            source_port = mem_read_comp.get_port(port_name)
            source_to_consumers_map[source_port].append(reduce_in_port)
        elif isinstance(provenance, tuple):
            in_idx, scatter_path = provenance
            source_port = scatter_comp.get_port(f"o_{scatter_path[0]}")
            source_to_consumers_map[source_port].append(reduce_in_port)

    if mem_read_comp:
        for info in analysis.consolidated_mem_access:
            if info.scatter_path_to_base:
                scatter_out_port_idx = info.scatter_path_to_base[0]
                source_port = scatter_comp.get_port(f"o_{scatter_out_port_idx}")
                dest_port_name = f"i_{info.in_idx}_{info.base_type}_id"
                dest_port = mem_read_comp.get_port(dest_port_name)
                if dest_port not in source_to_consumers_map[source_port]:
                    source_to_consumers_map[source_port].append(dest_port)

    print("Final wiring plan:")
    for source_port, consumer_ports in source_to_consumers_map.items():
        print(f"  Source Port: {source_port} from Component {source_port.parent.__class__.__name__}")
        for consumer_port in consumer_ports:
            print(
                f"    -> Consumer Port: {consumer_port} from Component {consumer_port.parent.__class__.__name__}"
            )

    for source_port, consumer_ports in source_to_consumers_map.items():
        current_source = source_port
        for i, consumer_port in enumerate(consumer_ports):
            assert not consumer_port.connected, f"Consumer port {consumer_port} is already connected."
            if i < len(consumer_ports) - 1:
                copy_comp = dfir.CopyComponent(source_port.data_type)
                final_components.append(copy_comp)
                assert not current_source.connected
                current_source.connect(copy_comp.get_port("i_0"))
                copy_comp.get_port("o_0").connect(consumer_port)
                current_source = copy_comp.get_port("o_1")
            else:
                assert not current_source.connected
                current_source.connect(consumer_port)

    # --- PHASE 5: Define final Collection I/O and construct it ---
    final_inputs = []
    if scatter_comp:
        final_inputs.append(scatter_comp.get_port("i_0"))
    elif mem_read_comp:
        final_inputs.extend(mem_read_comp.in_ports)

    final_outputs = [modified_reduce.get_port("o_0")]

    print(f"final_inputs: {final_inputs}")
    print(f"final_outputs: {final_outputs}")
    print(f"final_components: {final_components}")

    return dfir.ComponentCollection(components=final_components, inputs=final_inputs, outputs=final_outputs)


def refactor_other_comps(comp_col: dfir.ComponentCollection, g: GlobalGraph) -> dfir.ComponentCollection:
    """
    Refactors a component collection by partitioning it into "boundary" components
    (like Reduce, Collect, Conditional) and "compute islands" (pure computation subgraphs).

    It then optimizes each partition individually:
    - ReduceComponents are optimized to hoist out memory accesses.
    - Compute islands are refactored into MemoryRead and FusedOp components.
    Finally, it reassembles the optimized partitions into a new, more efficient graph.

    Args:
        comp_col: The original ComponentCollection to refactor.
        g: The GlobalGraph context for type information.

    Returns:
        A new, fully refactored and optimized ComponentCollection.
    """
    # --- PHASE 0: PRE-CHECKS AND DEFINITIONS ---
    io_components = [c for c in comp_col.components if isinstance(c, dfir.IOComponent)]
    # We will enforce at most one IO INPUT component in the final graph.

    def is_boundary(c: dfir.Component) -> bool:
        """Helper to identify components that act as partition boundaries."""
        return isinstance(
            c, (dfir.ReduceComponent, dfir.CollectComponent, dfir.ConditionalComponent, dfir.IOComponent)
        )

    def convert_spe_to_id_type(t: dfir.DfirType) -> dfir.DfirType:
        """Recursively convert SpecialType(node/edge) to SpecialIdType(node_id/edge_id)."""
        from graphyflow.dataflow_ir_datatype import (
            ArrayType,
            TupleType,
            OptionalType,
            SpecialType,
            SpecialIdType,
        )

        if isinstance(t, SpecialType):
            return SpecialIdType.from_spe(t)
        if isinstance(t, ArrayType):
            return ArrayType(convert_spe_to_id_type(t.type_))
        if isinstance(t, OptionalType):
            return OptionalType(convert_spe_to_id_type(t.type_))
        if isinstance(t, TupleType):
            return TupleType([convert_spe_to_id_type(tt) for tt in t.types])
        return t

    # --- PHASE 1: GRAPH PARTITIONING ---
    # Partition the graph into boundary components and compute islands once.
    boundary_components = {c for c in comp_col.components if is_boundary(c)}
    compute_components = {c for c in comp_col.components if not is_boundary(c)}

    comp_to_block_map: Dict[int, Union[dfir.Component, int]] = {}
    for bc in boundary_components:
        comp_to_block_map[bc.readable_id] = bc

    compute_islands: List[Set[dfir.Component]] = []
    visited_compute_ids = set()

    for comp in compute_components:
        if comp.readable_id in visited_compute_ids:
            continue

        # Start a new traversal to find a connected island
        new_island = set()
        q = collections.deque([comp])
        visited_compute_ids.add(comp.readable_id)

        while q:
            current_comp = q.popleft()
            new_island.add(current_comp)
            comp_to_block_map[current_comp.readable_id] = len(compute_islands)  # Assign island ID

            # Traverse to all connected compute components
            for port in current_comp.ports:
                if not port.connected:
                    continue
                neighbor = port.connection.parent
                if neighbor in compute_components and neighbor.readable_id not in visited_compute_ids:
                    visited_compute_ids.add(neighbor.readable_id)
                    q.append(neighbor)

        compute_islands.append(new_island)

    # --- PHASE 2: RECORD INTER-BLOCK CONNECTIONS ---
    # Record all connections between the identified blocks (boundaries and islands).
    @dataclass
    class Connection:
        src_block: Union[dfir.Component, int, str]
        src_port: dfir.Port
        dst_block: Union[dfir.Component, int, str]
        dst_port: dfir.Port

    connections: List[Connection] = []
    GRAPH_INPUT_MARKER = "GRAPH_INPUT"
    GRAPH_OUTPUT_MARKER = "GRAPH_OUTPUT"

    # Connections originating from graph inputs
    for p_in in comp_col.inputs:
        if p_in.connected:
            dst_comp = p_in.connection.parent
            dst_block = comp_to_block_map.get(dst_comp.readable_id)
            if dst_block is not None:
                connections.append(
                    Connection(
                        src_block=GRAPH_INPUT_MARKER,
                        src_port=p_in,
                        dst_block=dst_block,
                        dst_port=p_in.connection,
                    )
                )

    # Connections between internal blocks
    for comp in comp_col.components:
        src_block = comp_to_block_map.get(comp.readable_id)
        if src_block is None:
            continue

        for p_out in comp.out_ports:
            if p_out.connected:
                dst_comp = p_out.connection.parent
                dst_block = comp_to_block_map.get(dst_comp.readable_id)
                if dst_block is not None and src_block != dst_block:
                    connections.append(
                        Connection(
                            src_block=src_block,
                            src_port=p_out,
                            dst_block=dst_block,
                            dst_port=p_out.connection,
                        )
                    )

    # Connections terminating at graph outputs
    for p_out in comp_col.outputs:
        if p_out.connected:
            src_comp = p_out.connection.parent
            src_block = comp_to_block_map.get(src_comp.readable_id)
            if src_block is not None:
                connections.append(
                    Connection(
                        src_block=src_block,
                        src_port=p_out.connection,
                        dst_block=GRAPH_OUTPUT_MARKER,
                        dst_port=p_out,
                    )
                )

    # --- PHASE 3: OPTIMIZE ALL BLOCKS INDIVIDUALLY ---

    # Phase 3.1: Optimize ReduceComponents
    reduce_opt_blocks: Dict[int, dfir.ComponentCollection] = {}
    reduce_port_map: Dict[int, dfir.Port] = {}

    for rc in [c for c in boundary_components if isinstance(c, dfir.ReduceComponent)]:
        # Temporarily disconnect main I/O for optimization call
        p_in = rc.get_port("i_0")
        p_in_conn = p_in.connection
        p_out = rc.get_port("o_0")
        p_out_conn = p_out.connection
        if p_in.connected:
            p_in.disconnect()
        if p_out.connected:
            p_out.disconnect()

        opt_cc = optimize_reduce_comp(copy.deepcopy(rc), g)
        reduce_opt_blocks[rc.readable_id] = opt_cc

        # Map original i_0/o_0 ports to new optimized block's I/O
        if opt_cc.inputs:
            reduce_port_map[p_in.readable_id] = opt_cc.inputs[0]
        if opt_cc.outputs:
            reduce_port_map[p_out.readable_id] = opt_cc.outputs[0]

        # Reconnect for safety, though original comp_col is discarded
        if p_in_conn:
            p_in.connect(p_in_conn)
        if p_out_conn:
            p_out.connect(p_out_conn)

    # Phase 3.2: Optimize Compute Islands
    island_opt_blocks: Dict[int, dfir.ComponentCollection] = {}
    island_port_map: Dict[int, dfir.Port] = {}

    for i, island_comps in enumerate(compute_islands):
        # --- Create a robust copy of the island for refactoring ---
        # 1. Create clean copies of all components, stored in a map.
        orig_to_copy_map = {c.readable_id: copy.deepcopy(c) for c in island_comps}

        # 2. Disconnect all ports on the copies to prevent invalid, stale connections from deepcopy.
        for comp_copy in orig_to_copy_map.values():
            _disconnect_all_ports(comp_copy)

        temp_inputs, temp_outputs = [], []
        newly_added_comps = []

        # 3. Re-wire the copies based on the original island's topology.
        for original_comp in island_comps:
            comp_copy = orig_to_copy_map[original_comp.readable_id]

            # Re-wire internal connections and create placeholders for external ones.
            # We only process from the OUT port side to avoid creating connections twice.
            for p_out in original_comp.out_ports:
                if not p_out.connected:
                    continue

                dest_comp_orig = p_out.connection.parent
                p_in_orig = p_out.connection

                p_out_copy = comp_copy.get_port(p_out.name)

                if dest_comp_orig.readable_id in orig_to_copy_map:
                    # INTERNAL connection: Reconnect the copies.
                    dest_comp_copy = orig_to_copy_map[dest_comp_orig.readable_id]
                    p_in_copy = dest_comp_copy.get_port(p_in_orig.name)
                    p_out_copy.connect(p_in_copy)
                else:
                    # EXTERNAL connection: This is an island output. Use a placeholder.
                    ph = dfir.PlaceholderComponent(p_out.data_type)
                    p_out_copy.connect(ph.get_port("i_0"))
                    temp_outputs.append(ph.get_port("o_0"))
                    island_port_map[p_out.readable_id] = ph.get_port("o_0")
                    newly_added_comps.append(ph)

        # 4. Identify island inputs by finding unconnected input ports on the copies.
        for original_comp in island_comps:
            comp_copy = orig_to_copy_map[original_comp.readable_id]
            for p_in_orig in original_comp.in_ports:
                p_in_copy = comp_copy.get_port(p_in_orig.name)
                if not p_in_copy.connected:
                    # This must be an input from outside the island.
                    ph = dfir.PlaceholderComponent(p_in_orig.data_type)
                    ph.get_port("o_0").connect(p_in_copy)
                    temp_inputs.append(ph.get_port("i_0"))
                    island_port_map[p_in_orig.readable_id] = ph.get_port("i_0")
                    newly_added_comps.append(ph)

        # 5. Create the temporary ComponentCollection with a valid, self-contained graph.
        final_island_components = list(orig_to_copy_map.values()) + newly_added_comps
        temp_cc = dfir.ComponentCollection(final_island_components, temp_inputs, temp_outputs)

        refactored_cc = refactor_to_memread_fusedop(temp_cc, g).comp_col
        island_opt_blocks[i] = refactored_cc

        # Update port map to point to the new refactored ports
        for p_id in list(island_port_map.keys()):
            placeholder_port = island_port_map[p_id]
            if placeholder_port in temp_inputs:
                idx = temp_inputs.index(placeholder_port)
                if idx < len(refactored_cc.inputs):
                    island_port_map[p_id] = refactored_cc.inputs[idx]
            elif placeholder_port in temp_outputs:
                idx = temp_outputs.index(placeholder_port)
                if idx < len(refactored_cc.outputs):
                    island_port_map[p_id] = refactored_cc.outputs[idx]

    # --- PHASE 4: REASSEMBLE THE GRAPH ---
    final_components: List[dfir.Component] = []
    final_component_ids = set()

    # Helper to add components without duplicates
    def add_components_from_cc(cc: dfir.ComponentCollection):
        for comp in cc.components:
            if comp.readable_id not in final_component_ids:
                final_components.append(comp)
                final_component_ids.add(comp.readable_id)

    # Phase 4.1: Add all new optimized blocks and remaining boundaries
    for bc in boundary_components:
        if isinstance(bc, dfir.ReduceComponent):
            add_components_from_cc(reduce_opt_blocks[bc.readable_id])
        else:
            bc_copy = copy.deepcopy(bc)  # Use copies to ensure clean state
            _disconnect_all_ports(bc_copy)
            comp_to_block_map[bc.readable_id] = bc_copy  # Map original ID to the new copy
            if bc_copy.readable_id not in final_component_ids:
                final_components.append(bc_copy)
                final_component_ids.add(bc_copy.readable_id)

    for i, island_cc in island_opt_blocks.items():
        add_components_from_cc(island_cc)

    # Phase 4.2: Rewire the graph based on the recorded connections
    # Group consumers by source port to handle fan-out correctly
    source_to_consumers_map = collections.defaultdict(list)

    def get_new_port(block, old_port: dfir.Port) -> Optional[dfir.Port]:
        """Finds the corresponding port in the new, optimized graph."""
        if isinstance(block, dfir.ReduceComponent):
            return reduce_port_map.get(old_port.readable_id)
        elif isinstance(block, int):  # Island ID
            return island_port_map.get(old_port.readable_id)
        elif isinstance(block, dfir.Component):  # Other boundary component (now a copy)
            return block.get_port(old_port.name)
        return None

    final_outputs = []
    pending_inputs = []  # Inputs not connected to another block, likely from IO

    for conn in connections:
        new_src_port, new_dst_port = None, None

        # Handle source
        if conn.src_block == GRAPH_INPUT_MARKER:
            new_dst_port = get_new_port(conn.dst_block, conn.dst_port)
            if new_dst_port:
                pending_inputs.append(new_dst_port)
            continue
        else:
            new_src_port = get_new_port(conn.src_block, conn.src_port)

        # Handle destination
        if conn.dst_block == GRAPH_OUTPUT_MARKER:
            if new_src_port:
                final_outputs.append(new_src_port)
            continue
        else:
            new_dst_port = get_new_port(conn.dst_block, conn.dst_port)

        if new_src_port and new_dst_port:
            source_to_consumers_map[new_src_port].append(new_dst_port)

    # Execute connections using fanout for 1-to-many
    def fanout(source_out: dfir.Port, consumers: List[dfir.Port]):
        """Helper to fan-out a single source to multiple consumers using CopyComponents."""
        if not consumers:
            return
        current_src = source_out
        for i, dest in enumerate(consumers):
            if i < len(consumers) - 1:
                cp = dfir.CopyComponent(source_out.data_type)
                if cp.readable_id not in final_component_ids:
                    final_components.append(cp)
                    final_component_ids.add(cp.readable_id)
                current_src.connect(cp.get_port("i_0"))
                cp.get_port("o_0").connect(dest)
                current_src = cp.get_port("o_1")
            else:
                current_src.connect(dest)

    for src_port, consumers in source_to_consumers_map.items():
        fanout(src_port, consumers)

    # --- PHASE 5: FINALIZATION AND CLEANUP ---

    # Phase 5.1: Handle IO Component
    io_comp = next((c for c in io_components if c.io_type == dfir.IOComponent.IOType.INPUT), None)
    if not io_comp and comp_col.inputs:
        io_comp = dfir.IOComponent(dfir.IOComponent.IOType.INPUT, comp_col.inputs[0].data_type)

    if io_comp:
        io_copy = copy.deepcopy(io_comp)
        if io_copy.readable_id not in final_component_ids:
            final_components.insert(0, io_copy)
        fanout(io_copy.get_port("o_0"), [p for p in pending_inputs if not p.connected])

    # Phase 5.2: Type Conversion
    preserve_in_port_ids = {
        p.readable_id for p in reduce_port_map.values() if p.port_type == dfir.PortType.IN
    }
    for comp in final_components:
        if isinstance(comp, dfir.IOComponent):
            continue
        if hasattr(comp, "input_type") and comp.input_type is not None:
            if not any(p.readable_id in preserve_in_port_ids for p in getattr(comp, "in_ports", [])):
                comp.input_type = convert_spe_to_id_type(comp.input_type)
        if hasattr(comp, "output_type") and comp.output_type is not None:
            comp.output_type = convert_spe_to_id_type(comp.output_type)
        for p in comp.ports:
            if not (p.port_type == dfir.PortType.IN and p.readable_id in preserve_in_port_ids):
                p.data_type = convert_spe_to_id_type(p.data_type)

    # Phase 5.3: Terminate genuinely unused outputs
    for comp in list(final_components):
        for p_out in comp.out_ports:
            if not p_out.connected and p_out not in final_outputs:
                uem = dfir.UnusedEndMarkerComponent(p_out.data_type)
                final_components.append(uem)
                p_out.connect(uem.get_port("i_0"))

    # Phase 5.4: Construct final ComponentCollection
    final_cc = dfir.ComponentCollection(
        components=final_components,
        inputs=[],  # Inputs are now handled by the IOComponent
        outputs=final_outputs,
    )
    final_cc.update_ports()
    return final_cc


# Helper function that was part of the original code, needed by the new implementation
def _disconnect_all_ports(comp: dfir.Component):
    for p in list(comp.ports):
        if p.connected:
            p.disconnect()

import graphyflow.dataflow_ir as dfir
from typing import List, Tuple, Set, Dict, Any, Union, Optional
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
    global_block_cnt = 0

    @dataclass
    class Block:
        """A block is either a boundary component or a compute island."""

        is_island: bool
        comps: List[dfir.Component]
        is_input: bool = False
        is_output: bool = False
        input_ports: List[int] = None
        output_ports: List[int] = None

        def __post_init__(self):
            nonlocal global_block_cnt
            self.id = global_block_cnt
            global_block_cnt += 1

        def __hash__(self):
            return hash(self.id)

        def __repr__(self):
            type_str = "Island" if self.is_island else "Boundary"
            comp_ids = [f"[{c.readable_id}]{c.__class__.__name__}" for c in self.comps]
            return f"<{type_str} Block id={self.id}, is_input={self.is_input}, is_output={self.is_output}, comps={comp_ids}>"

    @dataclass
    class Connection:
        out: bool
        other: Block
        src_p: dfir.Port
        dst_p: dfir.Port

    # --- PHASE 0: PRE-CHECKS AND DEFINITIONS ---
    # io_components = [c for c in comp_col.components if isinstance(c, dfir.IOComponent)]
    assert all(
        not p.connected for p in comp_col.inputs + comp_col.outputs
    ), "All graph I/O ports must be disconnected."
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

    compute_islands: List[Set[dfir.Component]] = []
    visited_compute_ids = set()

    reduce_comps = [c for c in boundary_components if isinstance(c, dfir.ReduceComponent)]
    number_of_reduce = len(reduce_comps)
    number_of_re_subg = {name: number_of_reduce for name in ["key", "transform", "unit_reduce"]}
    victim_ports = []
    reduce_sub_out_names = {
        "o_reduce_key_in": "key",
        "o_reduce_transform_in": "transform",
        "o_reduce_unit_start_0": "unit_reduce",
        "o_reduce_unit_start_1": "unit_reduce",
    }
    ignored_reduce_sub_comps = []

    for reduce_comp in reduce_comps:
        for port_name, subgraph_name in reduce_sub_out_names.items():
            port = reduce_comp.get_port(port_name)
            assert (
                port.connected
            ), f"ReduceComponent {reduce_comp.readable_id} port {port_name} must be connected."
            victim_ports.append((port.connection, subgraph_name))

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

            # Traverse to all connected compute components
            for port in current_comp.ports:
                if not port.connected:
                    continue
                neighbor = port.connection.parent
                if neighbor in compute_components and neighbor.readable_id not in visited_compute_ids:
                    visited_compute_ids.add(neighbor.readable_id)
                    q.append(neighbor)

        # detect if any of the victim ports is in the island
        found_subgraph_name = None
        for vp, subgraph_name in victim_ports:
            if vp.parent in new_island:
                assert found_subgraph_name is None or found_subgraph_name == subgraph_name
                found_subgraph_name = subgraph_name
        if found_subgraph_name is not None:
            # This island is connected to a ReduceComponent subgraph
            # Mark it as part of that ReduceComponent's subgraph
            number_of_re_subg[found_subgraph_name] -= 1
            ignored_reduce_sub_comps.extend(new_island)
            if number_of_re_subg[found_subgraph_name] < 0:
                raise ValueError(
                    f"More than one ReduceComponent connected to the same {found_subgraph_name} subgraph."
                )
        else:
            compute_islands.append(new_island)

    print(
        f"Identified {len(boundary_components)} boundary components and {len(compute_islands)} compute islands."
    )
    # print("compute_islands:")
    # for i, island in enumerate(compute_islands):
    #     print(f"  Island {i}: {'\n'.join(str(c) for c in island)}\n\n")

    # --- PHASE 2: RECORD INTER-BLOCK CONNECTIONS ---
    # Record all connections between the identified blocks (boundaries and islands).

    boundary_blocks = [Block(is_island=False, comps=[bc]) for bc in boundary_components]
    island_blocks = [Block(is_island=True, comps=list(island)) for island in compute_islands]
    boundary_parent_block = {bc.readable_id: blk for blk in boundary_blocks for bc in blk.comps}
    island_parent_block = {comp.readable_id: blk for blk in island_blocks for comp in blk.comps}
    parent_block = {**boundary_parent_block, **island_parent_block}
    all_blocks = boundary_blocks + island_blocks
    connections = {blk: [] for blk in all_blocks}
    input_blocks = []
    output_blocks = []

    have_io = any(isinstance(c, dfir.IOComponent) for c in comp_col.components)
    assert (
        have_io or len(comp_col.inputs) == 1
    ), "Graphs without IOComponents must have exactly one input port."
    if have_io:
        assert len(comp_col.inputs) == 0
        # IO blk is input
        for blk in all_blocks:
            for c in blk.comps:
                if isinstance(c, dfir.IOComponent):
                    assert (
                        c.io_type == dfir.IOComponent.IOType.INPUT
                    ), "Only one IO INPUT component allowed in the graph."
                    blk.is_input = True
                    if blk.input_ports is None:
                        blk.input_ports = []
                    input_blocks.append(blk)
                    break
    else:
        # Connections originating from graph inputs
        for p_in in comp_col.inputs:
            found_target = False
            for blk in all_blocks:
                if p_in.parent in blk.comps:
                    assert not found_target, "Input port connected to multiple blocks, invalid graph."
                    blk.is_input = True
                    if blk.input_ports is None:
                        blk.input_ports = []
                    blk.input_ports.append(p_in.readable_id)
                    input_blocks.append(blk)
                    found_target = True

    # Connections terminating at graph outputs
    for p_out in comp_col.outputs:
        found_target = False
        for blk in all_blocks:
            if p_out.parent in blk.comps:
                assert not found_target, "Output port connected to multiple blocks, invalid graph."
                blk.is_output = True
                if blk.output_ports is None:
                    blk.output_ports = []
                blk.output_ports.append(p_out.readable_id)
                output_blocks.append(blk)
                found_target = True

    # Connections between internal blocks
    for comp in comp_col.components:
        if comp in ignored_reduce_sub_comps:
            continue
        print(f"Analyzing connections for component {comp.readable_id} ({comp.__class__.__name__})")
        src_block = parent_block.get(comp.readable_id)
        assert src_block is not None, "Component not assigned to any block."
        for p_out in comp.out_ports:
            assert p_out.connected or p_out in comp_col.outputs, "All non-output ports must be connected."
            if p_out.connected:
                dst_comp = p_out.connection.parent
                if dst_comp in ignored_reduce_sub_comps:
                    continue
                dst_block = parent_block.get(dst_comp.readable_id)
                assert dst_block is not None, "Destination component not assigned to any block."
                if src_block != dst_block:
                    connections[src_block].append(
                        Connection(out=True, other=dst_block, src_p=p_out, dst_p=p_out.connection)
                    )
                    connections[dst_block].append(
                        Connection(out=False, other=src_block, src_p=p_out, dst_p=p_out.connection)
                    )

    print("Block connections:")
    for blk, conns in connections.items():
        print(f"  Block {blk}:")
        for conn in conns:
            direction = "->" if conn.out else "<-"
            print(f"    {direction} Block {conn.other} via {conn.src_p} to {conn.dst_p}")

    # --- PHASE 3: OPTIMIZE ALL BLOCKS INDIVIDUALLY ---

    # Phase 3.1: Optimize ReduceComponents
    print("Optimizing ReduceComponents...")
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
        assert len(opt_cc.inputs) == 1, "Optimized ReduceComponent must have at most one input."
        assert len(opt_cc.outputs) == 1, "Optimized ReduceComponent must have at most one output."

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

    print("Optimizing Compute Islands...")
    # Phase 3.2: Optimize Compute Islands
    island_opt_blocks: Dict[int, dfir.ComponentCollection] = {}
    island_port_map: Dict[int, dfir.Port] = {}

    for i, island_blk in enumerate(island_blocks):
        island_comps = island_blk.comps
        print(f"Optimizing Island Block {i}")
        # --- Create a robust copy of the island for refactoring ---
        # 1. Create clean copies of all components, stored in a map.
        copied_island_comps = copy.deepcopy(island_comps)
        orig_to_copy_map = {c.readable_id: c for c in copied_island_comps}

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
                    if p_in_orig.connected:
                        # assert an incoming block connection is recorded
                        assert any(
                            (not conn.out and conn.dst_p == p_in_orig)
                            for conn in connections[parent_block[original_comp.readable_id]]
                        ), f"Island input port {p_in_orig} must have a recorded incoming connection."
                    else:
                        assert island_blk.is_input, "Unconnected input port must belong to an input block."
                        assert p_in_orig in comp_col.inputs, "Unconnected input port must be a graph input."

        # 5. Identify the output ports that are not connected internally.
        for original_comp in island_comps:
            comp_copy = orig_to_copy_map[original_comp.readable_id]
            for p_out_orig in original_comp.out_ports:
                p_out_copy = comp_copy.get_port(p_out_orig.name)
                if not p_out_copy.connected:
                    # This must be an output to outside the island.
                    ph = dfir.PlaceholderComponent(p_out_orig.data_type)
                    p_out_copy.connect(ph.get_port("i_0"))
                    temp_outputs.append(ph.get_port("o_0"))
                    island_port_map[p_out_orig.readable_id] = ph.get_port("o_0")
                    newly_added_comps.append(ph)
                    if p_out_orig.connected:
                        # assert an outgoing block connection is recorded
                        assert any(
                            (conn.out and conn.src_p == p_out_orig)
                            for conn in connections[parent_block[original_comp.readable_id]]
                        ), f"Island output port {p_out_orig} must have a recorded outgoing connection."
                    else:
                        assert island_blk.is_output, "Unconnected output port must belong to an output block."
                        assert (
                            p_out_orig in comp_col.outputs
                        ), "Unconnected output port must be a graph output."

        # 6. Create the temporary ComponentCollection with a valid, self-contained graph.
        final_island_components = list(orig_to_copy_map.values()) + newly_added_comps
        print(f"Creating cc with {final_island_components}, inputs={temp_inputs}, outputs={temp_outputs}")
        temp_cc = dfir.ComponentCollection(final_island_components, temp_inputs, temp_outputs)
        # print(f"Temporary island ComponentCollection:\n{temp_cc}")
        from graphyflow.visualize_ir import visualize_components

        dot_temp = visualize_components(str(temp_cc))
        dot_temp.render(f"output/island_{i}_temp_graph", view=False, format="png")
        print(f"Temporary island graph visualized to output/island_{i}_temp_graph.png")

        # Call the updated refactoring function which returns a result object
        refactor_result = refactor_to_memread_fusedop(temp_cc, g)
        refactored_cc = refactor_result.comp_col
        refactored_mapping = refactor_result.output_mapping
        print(f"{temp_inputs=}, {refactored_cc.inputs=}")
        print(f"{refactored_cc=}")
        assert (
            len(temp_inputs) == 1 and len(refactored_cc.inputs) == 1
        ), "Islands must have exactly one input."
        refactored_mapping[temp_inputs[0]] = refactored_cc.inputs[0]
        island_opt_blocks[i] = refactored_cc

        island_port_ids = [p.readable_id for p in island_port_map.values()]
        for old_port, new_port in refactored_mapping.items():
            print(f"Mapping old port ID {old_port} to new port {new_port}")
            old_port_id = old_port.readable_id
            assert old_port_id in island_port_ids, "Mapped port must be from the island ports."
            found_one = False
            for k, v in island_port_map.items():
                if v.readable_id == old_port_id:
                    assert not found_one, "Each old port ID should map to exactly one new port."
                    island_port_map[k] = new_port
                    print(f"Mapped old port ID {k} to new port {new_port} through {old_port_id}")
                    found_one = True
            assert found_one, "Old port ID must be found in island_port_map."

    # --- PHASE 4: REASSEMBLE THE GRAPH ---
    final_components: List[dfir.Component] = []
    final_component_ids = set()

    # Helper to add components without duplicates
    def add_components_from_cc(cc: dfir.ComponentCollection):
        for comp in cc.components:
            if comp.readable_id not in final_component_ids:
                final_components.append(comp)
                final_component_ids.add(comp.readable_id)
            else:
                print(f"Skipping duplicate component {comp} during reassembly.")

    other_bc_port_map: Dict[int, dfir.Port] = {}

    # Phase 4.1: Add all new optimized blocks and remaining boundaries
    for bc in boundary_components:
        if isinstance(bc, dfir.ReduceComponent):
            add_components_from_cc(reduce_opt_blocks[bc.readable_id])
        else:
            bc_copy = copy.deepcopy(bc)  # Use copies to ensure clean state
            _disconnect_all_ports(bc_copy)
            assert bc_copy.readable_id not in final_component_ids, "Component IDs must be unique."
            final_components.append(bc_copy)
            final_component_ids.add(bc_copy.readable_id)
            for p in bc_copy.ports:
                other_bc_port_map[p.readable_id] = p

    for i, island_cc in island_opt_blocks.items():
        add_components_from_cc(island_cc)

    # Phase 4.2: Rewire the graph based on the recorded connections
    # Group consumers by source port to handle fan-out correctly
    source_to_consumers_map = collections.defaultdict(list)

    final_outputs = []
    pending_inputs = []  # Inputs not connected to another block, likely from IO

    def get_block_port(block: Block, port_id: int) -> dfir.Port:
        """Helper to get the corresponding port in the refactored graph."""
        if block.is_island:
            assert port_id in island_port_map, "Island port must be mapped."
            return island_port_map[port_id]
        else:
            if isinstance(block.comps[0], dfir.ReduceComponent):
                assert port_id in reduce_port_map
                return reduce_port_map[port_id]
            else:
                # For other boundary components, ports remain unchanged
                assert port_id in other_bc_port_map, "Boundary component port must be mapped."
                return other_bc_port_map[port_id]

    for block, conns in connections.items():
        for conn in conns:
            if conn.out:
                # Outgoing connection from this block to another
                src_port = get_block_port(block, conn.src_p.readable_id)
                dst_port = get_block_port(conn.other, conn.dst_p.readable_id)
                print(f"Connecting {src_port} to {dst_port}")
                source_to_consumers_map[src_port].append(dst_port)

        if block.is_input:
            # This block has inputs from outside the graph
            for p_in_id in block.input_ports:
                pending_inputs.append(get_block_port(block, p_in_id))

        if block.is_output:
            # This block has outputs to outside the graph
            for p_out_id in block.output_ports:
                final_outputs.append(get_block_port(block, p_out_id))

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
    io_comps = [c for c in comp_col.components if isinstance(c, dfir.IOComponent)]
    assert len(io_comps) <= 1, "At most one IOComponent allowed in the original graph."
    io_comp = io_comps[0] if io_comps else None
    if io_comp is None:
        assert len(comp_col.inputs) == 1, "Graph must have exactly one input if no IOComponent is present."
        io_comp = dfir.IOComponent(dfir.IOComponent.IOType.INPUT, comp_col.inputs[0].data_type)
        final_components.insert(0, io_comp)
        assert len(pending_inputs) == 1, "There must be exactly one pending input to connect to IOComponent."
        fanout(io_comp.get_port("o_0"), pending_inputs)
    else:
        assert len(pending_inputs) == 0, "All inputs must be connected if IOComponent is present."

    # Phase 5.2: Type Conversion
    for comp in final_components:
        for p in comp.ports:
            p.data_type = convert_spe_to_id_type(p.data_type)

    # Phase 5.3: Terminate genuinely unused outputs
    for comp in list(final_components):
        for p_out in comp.out_ports:
            if not p_out.connected and p_out not in final_outputs:
                raise ValueError(f"Output port {p_out} of component {comp} is unused and not a graph output.")
                # uem = dfir.UnusedEndMarkerComponent(p_out.data_type)
                # final_components.append(uem)
                # p_out.connect(uem.get_port("i_0"))

    print(f"\n\nfinal_components: {final_components}")

    # Phase 5.4: Validate if all port connections' type is matched
    for comp in final_components:
        for p in comp.ports:
            if p.connected:
                assert p.data_type == p.connection.data_type, (
                    f"Type mismatch on connection from {p.connection} to {p}: "
                    f"{p.connection.data_type} -> {p.data_type}"
                )

    # Phase 5.5: Construct final ComponentCollection
    final_cc = dfir.ComponentCollection(
        components=final_components,
        inputs=[],  # Inputs are handled by the IOComponent
        outputs=final_outputs,
    )
    final_cc.update_ports()
    return final_cc


# Helper function that was part of the original code, needed by the new implementation
def _disconnect_all_ports(comp: dfir.Component):
    for p in list(comp.ports):
        if p.connected:
            p.disconnect()

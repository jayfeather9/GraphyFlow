import graphyflow.dataflow_ir as dfir
from typing import List, Tuple, Set, Dict
import collections
from graphyflow.dataflow_ir_utils import _extract_subgraph_from_reduce, refactor_to_memread_fusedop
from graphyflow.reduce_analysis import MemoryAccessInfo, SubgraphAnalysisResult, ReduceAnalysisResult
from graphyflow.global_graph import GlobalGraph


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

    analysis = SubgraphAnalysisResult(refactored_fused_op=fused_op, input_provenance=provenance)
    return mem_accesses, analysis


def simplify_reduce_comp_pass(
    comp_col: dfir.ComponentCollection, g: GlobalGraph
) -> Dict[str, ReduceAnalysisResult]:
    """
    Analyzes all ReduceComponents within a ComponentCollection and produces a
    detailed refactoring plan for each, without modifying the graph.

    This implementation analyzes the graph *after* it has been refactored
    to deduce all required pathing and memory access information.
    """
    analysis_results: Dict[str, ReduceAnalysisResult] = {}
    reduce_components = [c for c in comp_col.components if isinstance(c, dfir.ReduceComponent)]

    for reduce_comp in reduce_components:
        # --- PHASE 1: Refactor each subgraph individually ---
        key_subgraph_orig = _extract_subgraph_from_reduce(reduce_comp, "key")
        key_refactored_cc = refactor_to_memread_fusedop(key_subgraph_orig, g)

        transform_subgraph_orig = _extract_subgraph_from_reduce(reduce_comp, "transform")
        transform_refactored_cc = refactor_to_memread_fusedop(transform_subgraph_orig, g)

        unit_subgraph_orig = _extract_subgraph_from_reduce(reduce_comp, "unit_reduce")
        unit_refactored_cc = refactor_to_memread_fusedop(unit_subgraph_orig, g)
        assert not any(isinstance(c, dfir.MemoryReadComponent) for c in unit_refactored_cc.components)

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


if __name__ == "__main__":
    from graphyflow.global_graph import *
    from graphyflow.visualize_ir import visualize_components

    g = GlobalGraph(
        properties={
            "node": {"weight": dfir.IntType()},
            "edge": {"e_id": dfir.IntType()},
        }
    )
    nodes = g.add_graph_input("edge")
    src_dst_weight = nodes.map_(map_func=lambda edge: (edge.src.weight, edge.dst.weight, edge))
    # first_reduce = src_dst_weight.reduce_by(
    #     reduce_key=lambda sw, dw, e: sw,
    #     reduce_transform=lambda sw, dw, e: (sw, dw, e),
    #     reduce_method=lambda x, y: (x[0], x[1], x[2]),
    # )
    filtered = src_dst_weight.filter(filter_func=lambda sw, dw, e: sw > dw)
    reduced_result = filtered.reduce_by(
        reduce_key=lambda sw, dw, e: e.dst,
        reduce_transform=lambda sw, dw, e: (sw, e.dst),
        reduce_method=lambda x, y: (x[0] + y[0], x[1]),
    )
    result = reduced_result.map_(map_func=lambda w, dst: (w, dst.weight, dst))

    dfirs = g.to_dfir()
    dfirs[0] = delete_placeholder_components_pass(dfirs[0])
    dot = visualize_components(str(dfirs[0]))
    dot.render("component_graph", view=False, format="png")
    # print(dfirs[0].topo_sort())
    # import graphyflow.hls_utils as hls

    # header, source = hls.global_hls_config.generate_hls_code(g, dfirs[0])
    # import os

    # if not os.path.exists("output"):
    #     os.makedirs("output")
    # with open("output/graphyflow.h", "w") as f:
    #     f.write(header)
    # with open("output/graphyflow.cpp", "w") as f:
    #     f.write(source)

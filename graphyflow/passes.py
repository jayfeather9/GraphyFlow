import dataflow_ir as dfir
from typing import List, Tuple, Set


def delete_placeholder_components_pass(
    comp_col: dfir.ComponentCollection,
) -> dfir.ComponentCollection:
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

            if (
                upstream_connected_port is not None
                and downstream_connected_port is not None
            ):
                upstream_connected_port.connect(downstream_connected_port)

        else:
            components_to_keep.append(comp)

    comp_col = dfir.ComponentCollection(
        components_to_keep, comp_col.inputs, comp_col.outputs
    )
    comp_col.update_ports()

    return comp_col


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
    src_dst_weight = nodes.map_(
        map_func=lambda edge: (edge.src.weight, edge.dst.weight, edge)
    )
    filtered = src_dst_weight.filter(filter_func=lambda sw, dw, e: sw > dw)
    result = filtered.map_(map_func=lambda sw, dw, e: e.e_id)

    dfirs = g.to_dfir()
    dfirs[0] = delete_placeholder_components_pass(dfirs[0])
    dot = visualize_components(str(dfirs[0]))
    dot.render("component_graph", view=False, format="png")

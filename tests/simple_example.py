from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.visualize_ir import visualize_components
from graphyflow.lambda_func import lambda_min, lambda_max

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
# filtered = src_dst_weight.filter(filter_func=lambda sw, dw, e: sw > dw)
# result = filtered.map_(map_func=lambda sw, dw, e: e.e_id)

dfir = g.to_dfir()
dot = visualize_components(str(dfir[0]))
dot.render("component_graph", view=False, format="png")

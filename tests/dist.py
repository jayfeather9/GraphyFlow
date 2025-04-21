from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.visualize_ir import visualize_components
from graphyflow.lambda_func import lambda_min, lambda_max

g = GlobalGraph(
    properties={
        "node": {"distance": dfir.FloatType()},
        "edge": {"weight": dfir.FloatType()},
    }
)

edges = g.add_input("edge")

aaa = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
bbb = aaa.reduce_by(
    reduce_key=lambda src_dist, dst, edge_w: dst,
    reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
    reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
)
ccc = bbb.map_(map_func=lambda dist, node: (lambda_min(dist, node.distance), node))

dfir = g.to_dfir()
dot = visualize_components(str(dfir[0]))
dot.render("component_graph", view=False, format="png")

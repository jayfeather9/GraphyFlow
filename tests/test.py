from graphyflow.structures import *
from graphyflow.tmp_gas import *

g = GlobalGraph(properties={"node": {"out_degree": int}, "edge": {"pr": float}})
# node: node {out_degree: int}
# edge: edge {src: node, dst: node, pr: float}
nodes = g.add_input("node")
edges = g.add_input("edge")
# scatter
scattered = edges.map_(map_func=lambda edge: (edge.pr, edge.dst))
# gather
gathered_node_property = scattered.reduce_by(
    reduce_key=lambda src_pr, dst: dst,
    reduce_method=lambda ori, update: (ori[0] + update[0], ori[1]),
)
# apply
apply = gathered_node_property.map_(
    map_func=lambda pr, node: (
        (0.15 / nodes.length().to_tracer() + 0.85 * pr) / node.out_degree,
        node,
    )
)
g.apply_all_edges(apply, "pr")
print(g)
translate_graph(g)

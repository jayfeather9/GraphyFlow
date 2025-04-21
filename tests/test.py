from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.visualize_ir import visualize_components
from graphyflow.lambda_func import lambda_min, lambda_max

# g = GlobalGraph(
#     properties={
#         "node": {"out_degree": (int, "out_degree")},
#         "edge": {"pr": (float, "1.0f / node_num / edge.src.out_degree")},
#     }
# )
# g = GlobalGraph(
#     properties={
#         "node": {"out_degree": int},
#         "edge": {"pr": float},
#     }
# )
g = GlobalGraph(
    properties={
        "node": {"weight": dfir.IntType()},
        "edge": {"e_id": dfir.IntType()},
    }
)
nodes = g.add_input("edge")
# apply = nodes.filter(filter_func=lambda node: node.id + 5 > 10)
src_dst_weight = nodes.map_(
    map_func=lambda edge: (edge.src.weight, edge.dst.weight, edge)
)
test = src_dst_weight.reduce_by(
    reduce_key=lambda data: data[2].dst,
    reduce_transform=lambda data: (data[0], data[1], data[2].dst),
    reduce_method=lambda data1, data2: (
        data1[0] + data2[0],
        data1[1] + data2[1],
        data1[2],
    ),
)
# filtered = src_dst_weight.filter(filter_func=lambda sw, dw, e: sw > dw)
# result = filtered.map_(map_func=lambda sw, dw, e: e.e_id)

# node: node {out_degree: int}
# edge: edge {src: node, dst: node, pr: float}
# nodes = g.add_input("node")
# edges = g.add_input("edge")
# # scatter
# scattered = edges.map_(map_func=lambda edge: (edge.pr, edge.dst))
# # gather
# gathered_node_property = scattered.reduce_by(
#     reduce_transform=lambda src_pr, dst: dst,
#     reduce_method=lambda ori, update: (ori[0] + update[0], ori[1]),
# )
# # apply
# apply = gathered_node_property.map_(
#     map_func=lambda pr, node: (
#         (0.15 / nodes.length().to_tracer() + 0.85 * pr) / node.out_degree,
#         node,
#     )
# )
# g.apply_all_edges(apply, "pr")
# print(g)
# print(g.to_dfir())
dfir = g.to_dfir()
dot = visualize_components(str(dfir[0]))
dot.render("component_graph", view=False, format="png")
# i = 0
# for node in g.nodes.values():
#     print("=" * 20)
#     i += 1
# if i == 2:
#     print(node.to_dfir(dfir.ArrayType(dfir.SpecialType("node"))))
# elif i == 3:
#     print(
#         node.to_dfir(
#             dfir.ArrayType(
#                 dfir.TupleType(
#                     [dfir.IntType(), dfir.IntType(), dfir.SpecialType("node")]
#                 )
#             )
#         )
#     )
# elif i == 4:
#     print(
#         node.to_dfir(
#             dfir.ArrayType(
#                 dfir.TupleType(
#                     [dfir.IntType(), dfir.IntType(), dfir.SpecialType("node")]
#                 )
#             )
#         )
#     )
# if i == 5:
#     print(
#         node.to_dfir(
#             dfir.ArrayType(
#                 dfir.TupleType(
#                     [dfir.IntType(), dfir.IntType(), dfir.SpecialType("node")]
#                 )
#             )
#         )
#     )
# if i == 6:
#     print(
#         node.to_dfir(
#             dfir.ArrayType(
#                 dfir.TupleType(
#                     [dfir.IntType(), dfir.IntType(), dfir.SpecialType("node")]
#                 )
#             )
#         )
#     )

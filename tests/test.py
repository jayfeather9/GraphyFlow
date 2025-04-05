from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir

# g = GlobalGraph(
#     properties={
#         "node": {"out_degree": (int, "out_degree")},
#         "edge": {"pr": (float, "1.0f / node_num / edge.src.out_degree")},
#     }
# )
g = GlobalGraph(
    properties={
        "node": {"out_degree": int},
        "edge": {"pr": float},
    }
)
nodes = g.add_input("node")
# apply = nodes.filter(filter_func=lambda node: node.id + 5 > 10)
apply = nodes.map_(map_func=lambda node: (node.id + 5, 6, node))
apply2 = apply.map_(map_func=lambda a, b, c: (a + b, a + c.id, a < c.id, c))
apply3 = apply2.filter(filter_func=lambda a, b, c: (a > 10 + b * a))
apply4 = apply3.reduce_by(
    reduce_key=lambda a, b, c: b,
    reduce_method=lambda a, b: a + b,
)
apply5 = apply3.reduce_by(
    reduce_key=lambda x: x[2],
    reduce_method=lambda a, b: a + b,
)
# node: node {out_degree: int}
# edge: edge {src: node, dst: node, pr: float}
# nodes = g.add_input("node")
# edges = g.add_input("edge")
# # scatter
# scattered = edges.map_(map_func=lambda edge: (edge.pr, edge.dst))
# # gather
# gathered_node_property = scattered.reduce_by(
#     reduce_key=lambda src_pr, dst: dst,
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
print(g)
i = 0
for node in g.nodes.values():
    print("=" * 20)
    i += 1
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
    if i == 5:
        print(
            node.to_dfir(
                dfir.ArrayType(
                    dfir.TupleType(
                        [dfir.IntType(), dfir.IntType(), dfir.SpecialType("node")]
                    )
                )
            )
        )
    if i == 6:
        print(
            node.to_dfir(
                dfir.ArrayType(
                    dfir.TupleType(
                        [dfir.IntType(), dfir.IntType(), dfir.SpecialType("node")]
                    )
                )
            )
        )

from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.visualize_ir import visualize_components
from graphyflow.lambda_func import lambda_min, lambda_max

# ================图初始化================
g = GlobalGraph(  # 创建一个全局图对象 g
    properties={  # 定义图的属性
        "node": {
            "distance": dfir.FloatType()
        },  # 节点 (node) 具有 "distance" 属性，存储源点到该节点的最短距离
        "edge": {
            "weight": dfir.FloatType()
        },  # 边 (edge) 具有 "weight" 属性，存储边的权重
    }
)

# ================节点距离初始化================
src = g.add_input(
    "src", dfir.SpecialType("node")
)  # 添加一个名为 "src" 的特殊输入，代表源节点
# 将源节点的距离初始化为 0，生成 (节点对象, 0) 的元组
src_dist = src.map_(map_func=lambda node: (node, 0))
others = g.add_graph_input("node")  # 添加图的常规节点输入，名为 "node"
# 将其他所有节点的距离初始化为无穷大，生成 (节点对象, 无穷大) 的元组
others_dist = others.map_(map_func=lambda node: (node, float("inf")))
# 从 "others" 集合中过滤掉源节点 (基于节点 ID)
others_without_src = others.filter(filter_func=lambda node, dist: node.id != src.id)
# 将源节点(距离为0)和其他节点(距离为无穷)的数据流合并
all_nodes_dist = others_without_src.append(src_dist)
# 完成节点 "distance" 属性的初始化设置，使用 all_nodes_dist 作为初始值
g.finish_init(all_nodes_dist, {"node": ["distance"]})

# ================边权重初始化================
# 添加一个名为 "weight" 的输入，它是一个数组，包含所有边的权重信息
weight = g.add_input(
    "weight", dfir.ArrayType(dfir.FloatType(), dfir.SpecialType("edge"))
)
# 完成边 "weight" 属性的初始化设置，使用 weight 输入作为初始值
g.finish_init(weight, {"edge": ["weight"]})

# ================迭代计算最短路径================
edges = g.add_graph_input("edge")  # 获取图的所有边作为数据流输入
# 准备松弛操作：对每条边，生成一个包含 (源节点当前距离, 目标节点对象, 边权重) 的元组
potential_dst_updates = edges.map_(
    map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight)
)
# 按目标节点聚合，计算到达每个目标节点的所有路径中的最小潜在距离
min_potential_distances = potential_dst_updates.reduce_by(
    reduce_key=lambda src_dist, dst, edge_w: dst,  # 使用目标节点 (dst) 作为聚合的键
    # 对每个目标节点，转换数据为 (通过这条边计算出的潜在新距离, 目标节点对象)
    reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
    # 聚合方法：对于同一个目标节点，比较所有潜在新距离，保留最小值
    reduce_method=lambda x, y: (
        lambda_min(x[0], y[0]),
        x[1],
    ),  # x 和 y 是 (距离, 节点) 元组
)
# 筛选出那些计算出的最小潜在距离小于节点当前记录距离的更新 (即发生了松弛的节点)
update_len = min_potential_distances.filter(
    filter_func=lambda dist, node: dist < node.distance
).length()
# 创建迭代终止标记：如果没有节点的距离被更新 (update_len 为 0)，则 end_marker 为 True
end_marker = update_len.map_(map_func=lambda length: length == 0)
# 更新节点距离：对每个节点，取其当前距离和计算出的最小潜在距离中的较小值
updated_node_distances = min_potential_distances.map_(
    map_func=lambda dist, node: (lambda_min(dist, node.distance), node)
)
# 完成一次迭代的定义：指定使用 updated_node_distances 更新节点的 "distance" 属性，并提供 end_marker 作为终止条件
g.finish_iter(updated_node_distances, {"node": ["distance"]}, end_marker)

dfirs = g.to_dfir()
dot = visualize_components(str(dfirs[0]))
dot.render("component_graph", view=False, format="png")

import graphyflow.hls_utils as hls

header, source = hls.global_hls_config.generate_hls_code(g, dfirs[0])
import os

if not os.path.exists("output"):
    os.makedirs("output")
with open("output/graphyflow.h", "w") as f:
    f.write(header)
with open("output/graphyflow.cpp", "w") as f:
    f.write(source)

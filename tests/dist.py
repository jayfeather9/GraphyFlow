from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.visualize_ir import visualize_components
from graphyflow.lambda_func import lambda_min, lambda_max

# 定义全局图结构（假设已预置distance属性和weight属性）
g = GlobalGraph(
    properties={
        "node": {"distance": float, "out_degree": int},
        "edge": {"weight": float},
    }
)

# 添加输入数据源
nodes = g.add_input("node")
edges = g.add_input("edge")

source_id = 0

# 初始化节点距离：源节点为0，其他为无穷大
init_nodes = nodes.map_(
    map_func=lambda node: (0.0 if node.id == source_id else float("inf"), node)
)
g.apply_all_nodes(init_nodes, "distance")  # 将初始化结果应用到节点属性

# 迭代松弛操作（假设迭代n-1次）
for _ in range(nodes.length().to_tracer() - 1):
    # Scatter阶段：从边传播距离
    scattered = edges.map_(
        map_func=lambda edge: (edge.src.distance + edge.weight, edge.dst)
    )

    # Gather阶段：聚合到目标节点，取最小值
    gathered = scattered.reduce_by(
        reduce_key=lambda new_dist, dst: dst,
        reduce_method=lambda a, b: lambda_min(a, b),
    )

    # Apply更新：仅当新距离更小时更新节点属性
    apply_update = gathered.map_(
        map_func=lambda new_dist, node: (lambda_min(new_dist, node.distance), node)
    )
    g.apply_all_nodes(apply_update, "distance")

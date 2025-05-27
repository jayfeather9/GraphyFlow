from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.lambda_func import lambda_abs
from graphyflow.visualize_ir import visualize_components

# ================图初始化================
g = GlobalGraph(
    properties={
        "node": {
            "features": dfir.TensorType(dfir.FloatType()),  # 节点特征向量
            "hidden": dfir.TensorType(dfir.FloatType()),  # 隐藏状态向量
        },
        "edge": {"weight": dfir.FloatType()},  # 边权重
    }
)

# ================节点特征初始化================
# 添加初始节点特征输入
node_features = g.add_input(
    "node_features",
    dfir.TupleType(dfir.TensorType(dfir.FloatType()), dfir.SpecialType("node")),
)
# 初始化节点的隐藏状态，初始时与特征相同
initial_hidden = node_features.map_(map_func=lambda node_feat: node_feat)

# 完成节点属性初始化
g.finish_init(node_features, {"node": ["features"]})
g.finish_init(initial_hidden, {"node": ["hidden"]})

# ================边权重初始化================
# 添加边权重输入
edge_weights = g.add_input(
    "edge_weights",
    dfir.TupleType(dfir.TensorType(dfir.FloatType()), dfir.SpecialType("edge")),
)
# 完成边权重初始化
g.finish_init(edge_weights, {"edge": ["weight"]})

# ================消息传递和聚合================
# 获取所有节点和边
nodes = g.add_graph_input("node")
edges = g.add_graph_input("edge")

# 消息传递：每个节点向邻居发送其隐藏状态
messages = edges.map_(map_func=lambda edge: (edge.dst, edge.src.hidden * edge.weight))

# 按目标节点聚合所有收到的消息
aggregated_messages = messages.reduce_by(
    reduce_key=lambda dst_node, msg: dst_node,
    reduce_transform=lambda dst_node, msg: (msg, dst_node),
    reduce_method=lambda x, y: (x[0] + y[0], x[1]),  # 简单相加聚合
)

# 更新节点隐藏状态：结合当前隐藏状态和聚合消息
# 简化实现：新隐藏状态 = 0.5 * 当前隐藏状态 + 0.5 * 聚合消息
updated_hidden = aggregated_messages.map_(
    map_func=lambda agg_msg, node: (node, 0.5 * node.hidden + 0.5 * agg_msg)
)

# ================迭代终止条件================
# 计算新旧隐藏状态的差异（简单实现为差的绝对值）
state_diff = updated_hidden.map_(
    map_func=lambda node, new_hidden: lambda_abs(new_hidden - node.hidden).tensor_sum()
)

# 检查是否所有节点的状态变化都小于阈值
EPSILON = 1e-4  # 收敛阈值
not_converged = state_diff.filter(filter_func=lambda diff: diff > EPSILON).length()

# 如果没有节点变化超过阈值，则迭代结束
end_marker = not_converged.map_(map_func=lambda count: count == 0)

# 完成迭代定义：更新节点的隐藏状态，并提供终止条件
g.finish_iter(updated_hidden, {"node": ["hidden"]}, end_marker)

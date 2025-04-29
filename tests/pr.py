from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.lambda_func import lambda_abs

# ================ 图初始化 ================
# 定义超参数
DAMPING_FACTOR = 0.85
EPSILON = 1e-6  # 收敛阈值

g_pr = GlobalGraph(
    properties={
        # 节点属性：pagerank 分数，out_degree 出度
        "node": {"pagerank": dfir.FloatType(), "out_degree": dfir.IntType()},
        "edge": {},  # 边不需要额外属性
    }
)

# ================ 节点属性初始化 ================
nodes_pr = g_pr.add_graph_input("node")  # 获取所有节点
total_nodes_flow = nodes_pr.length()  # 获取节点总数 (假设 length() 返回一个可用的流)

# 初始化 PageRank: 通常为 1/N，这里先设为 1.0 (后续迭代公式处理)
# 初始化 Out-degree: 需要计算每个节点的出度

# ================ 迭代计算 PageRank ================
edges_pr = g_pr.add_graph_input("edge")  # 获取图的边
total_nodes_val = g_pr.add_graph_input("node").length()

# 准备 PageRank贡献 消息：
# 对于每条边，源节点将其当前的 pagerank / out_degree 传递给目标节点
# 简化：假设出度不为 0
pr_contributions = edges_pr.filter(
    filter_func=lambda edge: edge.src.out_degree > 0
).map_(map_func=lambda edge: (edge.dst, edge.src.pagerank / edge.src.out_degree))

# 按目标节点聚合，累加收到的所有 PageRank 贡献
summed_contributions = pr_contributions.reduce_by(
    reduce_key=lambda dst_node, contrib: dst_node,
    reduce_transform=lambda dst_node, contrib: (contrib, dst_node),
    reduce_method=lambda x, y: (x[0] + y[0], x[1]),  # (累加的贡献, 节点)
)

# 应用 PageRank 公式计算新得分: new_pr = (1 - DAMPING_FACTOR) / TOTAL_NODES + DAMPING_FACTOR * summed_contrib
updated_pagerank_data = summed_contributions.map_(
    map_func=lambda summed_contrib, node: (
        node,
        (1.0 - DAMPING_FACTOR) / total_nodes_val + DAMPING_FACTOR * summed_contrib,
    )
)

# 检查收敛性：计算新旧 PageRank 的差值绝对值是否小于阈值
# 这可能需要将 updated_pagerank_data 与节点的当前 pagerank 结合比较
# 概念：假设 filter 可以访问 node.pagerank
converged_check = updated_pagerank_data.filter(
    # 使用假设的 lambda_abs_diff_gt 比较差值绝对值是否 > EPSILON
    filter_func=lambda node, new_rank: lambda_abs(new_rank - node.pagerank)
    > EPSILON
)

# 计算未收敛的节点数量
not_converged_count = converged_check.length()
# 如果没有节点变化超过阈值，则 end_marker 为 True
end_marker_pr = not_converged_count.map_(map_func=lambda length: length == 0)

# 完成迭代：更新节点的 pagerank 属性
g_pr.finish_iter(updated_pagerank_data, {"node": ["pagerank"]}, end_marker_pr)

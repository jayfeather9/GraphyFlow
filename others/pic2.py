from graphviz import Digraph

# 创建主图
g = Digraph("DFG_IR_Optimization", format="png")
g.attr(fontname="SimHei", fontsize="16", nodesep="0.6", ranksep="0.6", splines="spline")

# 添加主标题节点
g.node(
    "DFGIR",
    label="DFG-IR优化策略体系",
    shape="doubleoctagon",
    color="gold",
    fontsize="18",
    style="filled",
    width="2",
    height="1",
)

# 添加三大类优化策略节点
g.node(
    "Basic",
    label="基础编译优化策略\n(通用优化)",
    shape="box3d",
    color="lightgreen",
    style="filled",
    fontsize="14",
)
g.node(
    "Differential",
    label="差分优化策略\n(动态图计算优化)",
    shape="box3d",
    color="lightpink",
    style="filled",
    fontsize="14",
)
g.node(
    "Parallel",
    label="并行优化策略\n(硬件友好优化)",
    shape="box3d",
    color="lightskyblue",
    style="filled",
    fontsize="14",
)

# 连接主节点到三大类
g.edge("DFGIR", "Basic", label="第一阶段")
g.edge("DFGIR", "Differential", label="第二阶段")
g.edge("DFGIR", "Parallel", label="第三阶段")

# 添加优化目标节点
g.node(
    "Target",
    label="优化目标：提高动态图算法执行效率，降低硬件资源消耗",
    shape="note",
    color="lightyellow",
    fontsize="14",
    style="filled",
)
g.edge("DFGIR", "Target", style="dotted")

# 添加策略流程连接
g.edge("Basic", "Differential", label="通用→专用", style="dashed")
g.edge("Differential", "Parallel", label="软件→硬件", style="dashed")

# 设置三大类策略节点的同级排列
with g.subgraph() as s:
    s.attr(rank="same")
    s.node("Basic")
    s.node("Differential")
    s.node("Parallel")

# 创建子图1：基础编译优化策略（竖向排列）
with g.subgraph(name="cluster_basic") as c:
    c.attr(
        label="基础编译优化策略",
        style="rounded,filled",
        color="lightgreen",
        fontsize="16",
    )
    c.attr(rankdir="TB")

    c.node("B1", label="常量传播", style="filled", color="lightblue")
    c.node("B2", label="死代码消除", style="filled", color="lightblue")
    c.node("B3", label="算子融合", style="filled", color="lightblue")
    c.node("B4", label="代数优化", style="filled", color="lightblue")
    c.node("B5", label="条件分支提升", style="filled", color="lightblue")
    c.node("B6", label="公共子表达式消除", style="filled", color="lightblue")

    c.edge("Basic", "B1")
    c.edge("Basic", "B2")
    c.edge("Basic", "B3")
    c.edge("Basic", "B4")
    c.edge("Basic", "B5")
    c.edge("Basic", "B6")

# 创建子图2：差分优化策略（横向排列）
with g.subgraph(name="cluster_differential") as c:
    c.attr(label="差分优化策略", style="rounded,filled", color="lightpink", fontsize="16")
    c.attr(rankdir="LR")

    c.node("D1", label="增量计算识别", style="filled", color="lightblue")
    c.node("D2", label="差分传播路径优化", style="filled", color="lightblue")
    c.node("D3", label="变化量阈值过滤", style="filled", color="lightblue")
    c.node("D4", label="缓存感知差分处理", style="filled", color="lightblue")
    c.node("D5", label="差分计算结果重用", style="filled", color="lightblue")

    c.edge("Differential", "D1")
    c.edge("Differential", "D2")
    c.edge("Differential", "D3")
    c.edge("Differential", "D4")
    c.edge("Differential", "D5")

    # 设置横向排列
    with c.subgraph() as s:
        s.attr(rank="same")
        s.node("D1")
        s.node("D2")
        s.node("D3")
        s.node("D4")
        s.node("D5")

# 创建子图3：并行优化策略（竖向排列）
with g.subgraph(name="cluster_parallel") as c:
    c.attr(
        label="并行优化策略",
        style="rounded,filled",
        color="lightskyblue",
        fontsize="16",
    )
    c.attr(rankdir="TB")

    c.node("P1", label="元组分解并行", style="filled", color="lightblue")
    c.node("P2", label="归约操作优化\nO(n)→O(log n)", style="filled", color="lightblue")
    c.node("P3", label="数据划分与负载均衡", style="filled", color="lightblue")
    c.node("P4", label="流水线并行", style="filled", color="lightblue")
    c.node("P5", label="数据块大小自适应", style="filled", color="lightblue")
    c.node("P6", label="依赖感知任务调度", style="filled", color="lightblue")

    c.edge("Parallel", "P1")
    c.edge("Parallel", "P2")
    c.edge("Parallel", "P3")
    c.edge("Parallel", "P4")
    c.edge("Parallel", "P5")
    c.edge("Parallel", "P6")

# 保存图像
g.render("optimization_for_ppt", format="png", cleanup=True)

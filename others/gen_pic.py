from graphviz import Digraph

# 创建Digraph对象
dot = Digraph(comment="硬件描述级代码生成路径")
dot.attr(rankdir="LR")  # 从左到右的方向
dot.attr("node", shape="box", style="filled", color="lightblue")

# 添加节点
dot.node("GraphyFlow", "GraphyFlow")
dot.node("DFG_IR", "DFG-IR")
# dot.node('Optimized_DFG_IR', '优化后的DFG-IR')
dot.node("HW_Units", "硬件功能\n单元映射")
dot.node("Resource_Scheduling", "资源分配\n与时序调度")
dot.node("Calyx_IR", "Calyx IR")
dot.node("Verilog", "Verilog\nHDL")

# 添加配置文件节点
dot.node("Config", "目标平台约束配置文件", shape="note", color="lightyellow")

# 添加边
dot.edge("GraphyFlow", "DFG_IR", label="转换")
dot.edge("DFG_IR", "HW_Units", label="优化后映射")
# dot.edge('Optimized_DFG_IR', 'HW_Units', label='模块化映射')
dot.edge("HW_Units", "Resource_Scheduling", label="分配调度")
dot.edge("Config", "Resource_Scheduling", label="提供约束", style="dashed")
dot.edge("Resource_Scheduling", "Calyx_IR", label="生成")
dot.edge("Calyx_IR", "Verilog", label="转换")

# 保存图片
dot.render("hardware_code_generation_path", format="png", cleanup=True)
print("图片已保存为: hardware_code_generation_path.png")

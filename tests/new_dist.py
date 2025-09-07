# tests/dist.py
from pathlib import Path
from graphyflow.global_graph import GlobalGraph
import graphyflow.dataflow_ir as dfir
from graphyflow.lambda_func import lambda_min
from graphyflow.passes import delete_placeholder_components_pass

# 导入我们最终的生成器 API
from graphyflow.project_generator import generate_project

# ==================== 配置 =======================
KERNEL_NAME = "graphyflow"
EXECUTABLE_NAME = "graphyflow_host"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "generated_project"

# ==================== 1. 定义图算法 =======================
print("--- Defining Graph Algorithm using GraphyFlow ---")
g = GlobalGraph(
    properties={
        "node": {"distance": dfir.FloatType()},
        "edge": {"weight": dfir.FloatType()},
    }
)
edges = g.add_graph_input("edge")
pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
pdu = pdu.filter(filter_func=lambda x, y, z: z >= 0.0)
min_dist = pdu.reduce_by(
    reduce_key=lambda src_dist, dst, edge_w: dst.id,
    reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
    reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
)
updated_nodes = min_dist.map_(map_func=lambda dist, node: (lambda_min(dist, node.distance), node))

# ==================== 2. 前端处理 =======================
print("\n--- Frontend Processing ---")
dfirs = g.to_dfir()
comp_col = delete_placeholder_components_pass(dfirs[0])
print("DFG-IR generated and optimized.")

# ==================== 3. 一键生成项目！ =======================
print("\n--- Generating Full Vitis Project ---")
generate_project(
    comp_col=comp_col,
    global_graph=g,
    kernel_name=KERNEL_NAME,
    output_dir=OUTPUT_DIR,
    executable_name=EXECUTABLE_NAME,
)

print("\n========================================================")
print("     Final Showcase Completed Successfully!     ")
print(f"   Project files are located in: {OUTPUT_DIR}   ")
print("========================================================")

# tests/dist.py

import os
import shutil
from pathlib import Path

# 确保所有需要的模块都被导入
from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.visualize_ir import visualize_components
from graphyflow.lambda_func import lambda_min
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.backend_manager import BackendManager

# ==================== 配置 =======================
# 定义内核和可执行文件的基础名称
KERNEL_NAME = "graphyflow"
EXECUTABLE_NAME = "graphyflow_host"

# 定义项目路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
# 定义最终生成项目的目录
OUTPUT_DIR = PROJECT_ROOT / "generated_project"
# 将 tmp_work 作为静态文件的模板源
TEMPLATE_DIR = PROJECT_ROOT / "tmp_work"

# ==================== 1. 定义图算法 (与之前相同) =======================
print("--- Step 1: Defining Graph Algorithm using GraphyFlow ---")
g = GlobalGraph(
    properties={
        "node": {"distance": dfir.FloatType()},
        "edge": {"weight": dfir.FloatType()},
    }
)
edges = g.add_graph_input("edge")
potential_dst_updates = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
potential_dst_updates = potential_dst_updates.filter(filter_func=lambda x, y, z: z >= 0.0)
min_potential_distances = potential_dst_updates.reduce_by(
    reduce_key=lambda src_dist, dst, edge_w: dst.id,
    reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
    reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
)
updated_node_distances = min_potential_distances.map_(
    map_func=lambda dist, node: (lambda_min(dist, node.distance), node)
)

# ==================== 2. 前端处理和IR优化 (与之前相同) =======================
print("\n--- Step 2: Frontend Processing & IR Optimization ---")
dfirs = g.to_dfir()
comp_col = delete_placeholder_components_pass(dfirs[0])
print("DFG-IR generated and optimized.")


# ==================== 3. 创建输出目录结构 =======================
print(f"\n--- Step 3: Setting up Output Directory: '{OUTPUT_DIR}' ---")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
    print(f"Removed existing directory: {OUTPUT_DIR}")

HOST_SCRIPT_DIR = OUTPUT_DIR / "scripts" / "host"
KERNEL_SCRIPT_DIR = OUTPUT_DIR / "scripts" / "kernel"
XCLBIN_DIR = OUTPUT_DIR / "xclbin"

HOST_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
KERNEL_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
XCLBIN_DIR.mkdir(exist_ok=True)
print("Created project directory structure.")


# ==================== 4. 后端代码生成 =======================
print("\n--- Step 4: Backend C++/Build System Code Generation ---")
bkd_mng = BackendManager()

# 生成内核代码 (graphyflow.h, graphyflow.cpp)
kernel_h, kernel_cpp = bkd_mng.generate_backend(comp_col, g, KERNEL_NAME)
print(f"Generated kernel HLS code for '{KERNEL_NAME}'.")

# 生成共享头文件 (common.h)
common_h = bkd_mng.generate_common_header(KERNEL_NAME)
print("Generated shared header file 'common.h'.")

# 注意：主机代码和构建文件生成将在后续阶段完善，目前可能只是占位符
host_h, host_cpp = bkd_mng.generate_host_codes(KERNEL_NAME)
print("Generated host-side interface code (placeholders for now).")
build_files = bkd_mng.generate_build_system_files(KERNEL_NAME, EXECUTABLE_NAME)
print("Generated build system files (Makefiles, etc.).")


# ==================== 5. 部署生成的文件 =======================
print(f"\n--- Step 5: Deploying Generated Files to '{OUTPUT_DIR}' ---")

# 写入动态生成的内核文件
with open(KERNEL_SCRIPT_DIR / f"{KERNEL_NAME}.h", "w") as f:
    f.write(kernel_h)
with open(KERNEL_SCRIPT_DIR / f"{KERNEL_NAME}.cpp", "w") as f:
    f.write(kernel_cpp)
print(f"Deployed: {KERNEL_NAME}.h, {KERNEL_NAME}.cpp")

# 写入动态生成的共享头文件
with open(HOST_SCRIPT_DIR / "common.h", "w") as f:
    f.write(common_h)
print(f"Deployed: common.h")

# 写入动态生成的主机文件 (当前为占位符)
# with open(HOST_SCRIPT_DIR / "generated_host.h", "w") as f: f.write(host_h)
# with open(HOST_SCRIPT_DIR / "generated_host.cpp", "w") as f: f.write(host_cpp)
# print(f"Deployed: generated_host.h, generated_host.cpp (placeholders)")


# ==================== 6. 复制所有静态文件 =======================
print(f"\n--- Step 6: Copying Static Files from '{TEMPLATE_DIR}' ---")

# 定义需要复制的静态文件列表
static_root_files = ["Makefile", "run.sh", "system.cfg", "global_para.mk", "gen_random_graph.py"]
static_script_files = ["clean.mk", "help.mk", "main.mk", "utils.mk"]
static_host_files = [
    "fpga_executor.cpp",
    "fpga_executor.h",
    "graph_loader.cpp",
    "graph_loader.h",
    "host.cpp",
    "host.mk",
    "host_verifier.cpp",
    "host_verifier.h",
    "xcl2.cpp",
    "xcl2.h",
    # 注意: 我们暂时从模板复制这些文件，后续会动态生成
    "generated_host.cpp",
    "generated_host.h",
]
static_kernel_files = ["kernel.mk"]

# 执行复制
try:
    for f in static_root_files:
        shutil.copy(TEMPLATE_DIR / f, OUTPUT_DIR / f)

    for f in static_script_files:
        shutil.copy(TEMPLATE_DIR / "scripts" / f, OUTPUT_DIR / "scripts" / f)

    for f in static_host_files:
        shutil.copy(TEMPLATE_DIR / "scripts" / "host" / f, HOST_SCRIPT_DIR / f)

    for f in static_kernel_files:
        shutil.copy(TEMPLATE_DIR / "scripts" / "kernel" / f, KERNEL_SCRIPT_DIR / f)

    print("All static files copied successfully.")

except FileNotFoundError as e:
    print(f"\n[ERROR] Could not copy static files. Make sure the '{TEMPLATE_DIR}' directory is complete.")
    print(f"File not found: {e.filename}")
    exit(1)


print("\n========================================================")
print("Generation process completed successfully!")
print(f"Project files are located in: {OUTPUT_DIR}")
print("========================================================")

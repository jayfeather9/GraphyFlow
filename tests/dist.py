import os
import shutil
import subprocess
from pathlib import Path

from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.visualize_ir import visualize_components
from graphyflow.lambda_func import lambda_min, lambda_max
from graphyflow.passes import delete_placeholder_components_pass
import graphyflow.backend_manager as bkd

# ==================== 配置 =======================
# 定义内核和可执行文件的基础名称
KERNEL_NAME = "graphyflow"
EXECUTABLE_NAME = "graphyflow_host"

# 定义项目路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
TMP_WORK_DIR = PROJECT_ROOT / "tmp_work"
HOST_SCRIPT_DIR = TMP_WORK_DIR / "scripts" / "host"
KERNEL_SCRIPT_DIR = TMP_WORK_DIR / "scripts" / "kernel"


# ==================== 1. 定义图算法 =======================
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
# g.finish_iter(updated_node_distances, {"node": ["distance"]}, None) # 假设单次执行，无终止条件


# ==================== 2. 前端处理和IR优化 =======================
print("\n--- Step 2: Frontend Processing & IR Optimization ---")
dfirs = g.to_dfir()
dfirs[0] = delete_placeholder_components_pass(dfirs[0])
dot = visualize_components(str(dfirs[0]))
OUTPUT_DIR.mkdir(exist_ok=True)
dot.render(str(OUTPUT_DIR / "component_graph"), view=False, format="png")
print(f"Component graph visualization saved to {OUTPUT_DIR / 'component_graph.png'}")


# ==================== 3. 后端代码生成 =======================
print("\n--- Step 3: Backend C++/Build System Code Generation ---")
bkd_mng = bkd.BackendManager()

# 生成内核代码
kernel_h, kernel_cpp = bkd_mng.generate_backend(dfirs[0], g, KERNEL_NAME)
print(f"Generated kernel code for '{KERNEL_NAME}'.")

# 生成Host代码
host_h, host_cpp = bkd_mng.generate_host_codes(KERNEL_NAME)
print("Generated host-side interface code (generated_host.h/.cpp).")

# 生成构建系统文件
build_files = bkd_mng.generate_build_system_files(KERNEL_NAME, EXECUTABLE_NAME)
print("Generated parameterized build system files (Makefiles, run.sh, etc.).")


# ==================== 4. 将生成的文件部署到编译目录 =======================
print(f"\n--- Step 4: Deploying Generated Files to '{TMP_WORK_DIR}' ---")

# 确保目标目录存在
KERNEL_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
HOST_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# 写入内核文件
with open(KERNEL_SCRIPT_DIR / f"{KERNEL_NAME}.h", "w") as f:
    f.write(kernel_h)
with open(KERNEL_SCRIPT_DIR / f"{KERNEL_NAME}.cpp", "w") as f:
    f.write(kernel_cpp)
print(f"Copied kernel files to {KERNEL_SCRIPT_DIR}")

# 写入Host文件
with open(HOST_SCRIPT_DIR / "generated_host.h", "w") as f:
    f.write(host_h)
with open(HOST_SCRIPT_DIR / "generated_host.cpp", "w") as f:
    f.write(host_cpp)
print(f"Copied host files to {HOST_SCRIPT_DIR}")

# 写入构建系统文件
for filename, content in build_files.items():
    if "kernel.mk" in filename:
        target_path = KERNEL_SCRIPT_DIR / filename
    elif "fpga_executor.cpp" in filename:  # fpga_executor.cpp 也在 host 目录
        target_path = HOST_SCRIPT_DIR / filename
    else:
        target_path = TMP_WORK_DIR / filename

    with open(target_path, "w") as f:
        f.write(content)
print(f"Updated build system files in '{TMP_WORK_DIR}'.")


# ==================== 5. 执行编译和仿真 =======================
print(f"\n--- Step 5: Running Compilation and Simulation in '{TMP_WORK_DIR}' ---")

try:
    # 执行 make clean all
    print("\n[CMD] make cleanall all")
    # check=True 会在命令失败时抛出异常
    subprocess.run(["make", "cleanall", "all"], cwd=TMP_WORK_DIR, check=True, capture_output=True, text=True)
    print("Build successful.")

    # 执行 run.sh
    print("\n[CMD] ./run.sh")
    result = subprocess.run(["./run.sh"], cwd=TMP_WORK_DIR, check=True, capture_output=True, text=True)

    # 打印仿真输出
    print("\n--- Simulation Output ---")
    print(result.stdout)
    if result.stderr:
        print("\n--- Simulation Stderr ---")
        print(result.stderr)

except subprocess.CalledProcessError as e:
    print("\n--- AN ERROR OCCURRED ---")
    print(f"Command '{' '.join(e.cmd)}' failed with exit code {e.returncode}.")
    print("\n--- Stdout ---")
    print(e.stdout)
    print("\n--- Stderr ---")
    print(e.stderr)
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("Could not execute command. Is 'make' or 'bash' installed and in your PATH?")

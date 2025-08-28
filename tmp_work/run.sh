#!/bin/bash

# 加载 Vitis 和 XRT 的标准环境设置
source /opt/Xilinx2024/Vitis/2024.1/settings64.sh
source /opt/xilinx/xrt/setup.sh

# ------------------- 关键修复 -------------------
# 使用 LD_PRELOAD 强制为所有子进程（包括 XRT 的仿真进程）加载 OpenCL 库。
# 这是解决深层环境问题的终极方法。
export LD_PRELOAD=/lib/x86_64-linux-gnu/libOpenCL.so.1
# -----------------------------------------------

# 设置仿真模式
export XCL_EMULATION_MODE=sw_emu
DATASET="./graph.txt"

# echo "--- Running with LD_PRELOAD fix ---"
# echo "LD_PRELOAD is set to: $LD_PRELOAD"
# echo "-----------------------------------"

# 运行 host 程序
./graphyflow_host ./xclbin/graphyflow.sw_emu.xclbin $DATASET
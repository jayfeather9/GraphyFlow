#!/bin/bash

# Setup Vitis environment
# 你可能需要根据你的安装路径修改这一行
source /home/feiyang/set_env.sh

# Set emulation mode
export XCL_EMULATION_MODE=sw_emu
# export XCL_EMULATION_MODE=hw_emu

DATASET="./graph.txt"
# DATASET="/home/feiyang/ReGraph/dataset/rmat-19-32.txt"

# Run the host application
# Usage: ./<executable> <xclbin_file> <graph_data_file>
./bellman_ford_host ./xclbin/bellman_ford.sw_emu.xclbin $DATASET
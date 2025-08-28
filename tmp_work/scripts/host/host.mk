# Makefile for the Host Application

# Define the C++ compiler
CXX := g++

# Executable name
EXECUTABLE := bellman_ford_host

# Source files - 更新源文件列表
# 移除了 fpga_bellman_ford.cpp 和 fpga_handler.cpp
# 添加了 generated_host.cpp 和 fpga_executor.cpp
HOST_SRCS := scripts/host/host.cpp \
             scripts/host/graph_loader.cpp \
             scripts/host/fpga_executor.cpp \
             scripts/host/generated_host.cpp \
             scripts/host/host_verifier.cpp \
             scripts/host/host_bellman_ford.cpp \
             scripts/host/xcl2.cpp

# Include directories
CXXFLAGS := -Iscripts/host
# 新增: 将生成的 kernel 目录也加入 include 路径
CXXFLAGS += -Iscripts/kernel
CXXFLAGS += -I$(XILINX_XRT)/include
CXXFLAGS += -I$(XILINX_VITIS)/include

# Compiler flags
CXXFLAGS += -std=c++14 -Wall

# Linker flags
LDFLAGS := -L$(XILINX_XRT)/lib
LDFLAGS += -lxilinxopencl -lxrt_coreutil -lstdc++ -lrt -pthread

# Build rule for the executable
$(EXECUTABLE): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) -o $(EXECUTABLE) $(LDFLAGS)

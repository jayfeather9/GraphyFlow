# Makefile for the Host Application
CXX := g++

# Executable name (由顶层 Makefile 传入)
# EXECUTABLE := graphyflow_host

# Source files
HOST_SRCS := scripts/host/host.cpp \
             scripts/host/graph_loader.cpp \
             scripts/host/fpga_executor.cpp \
             scripts/host/generated_host.cpp \
             scripts/host/host_verifier.cpp \
             scripts/host/host_bellman_ford.cpp \
             scripts/host/xcl2.cpp

# Include directories
# 新增: 将生成的 kernel 目录也加入 include 路径, 内核头文件名与内核名相同
CXXFLAGS := -Iscripts/host -Iscripts/kernel
CXXFLAGS += -I$(XILINX_XRT)/include
CXXFLAGS += -I$(XILINX_VITIS)/include
CXXFLAGS += -I$(XILINX_HLS)/include

# Compiler flags
CXXFLAGS += -std=c++17 -Wall

# Linker flags
LDFLAGS := -L$(XILINX_XRT)/lib
LDFLAGS += -lOpenCL -lxrt_coreutil -lstdc++ -lrt -pthread -Wl,--export-dynamic

# Build rule for the executable
$(EXECUTABLE): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) -o $(EXECUTABLE) $(LDFLAGS)
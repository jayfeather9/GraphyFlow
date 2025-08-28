# Makefile for the Bellman-Ford Host Application

# Define the C++ compiler
CXX := g++

# Executable name
EXECUTABLE := bellman_ford_host

# Source files - Correct the path
HOST_SRCS := $(wildcard scripts/host/*.cpp)

# Include directories
# Correct the include path for common.h
CXXFLAGS := -Iscripts/host
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
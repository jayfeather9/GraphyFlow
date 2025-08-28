#TARGET := hw
TARGET := sw_emu

DEVICE := /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm

.PHONY: all clean cleanall exe

# Remove hwemuprepare, it's not needed for this flow
# include scripts/main.mk
include scripts/main.mk

# Add a 'check' target for easy execution
check: all
	./run.sh
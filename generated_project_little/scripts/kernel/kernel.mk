
# Makefile for the Vitis Kernel
VPP := v++
# KERNEL_NAME is passed from the top Makefile
KERNEL_SRC := scripts/kernel/$(KERNEL_NAME).cpp
XCLBIN_DIR := ./xclbin
XCLBIN_FILE := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xclbin
KERNEL_XO := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xo
EMCONFIG_FILE := ./emconfig.json
CLFLAGS += --kernel $(KERNEL_NAME)
CLFLAGS += -Iscripts/kernel
CLFLAGS += -Iscripts/host
CLFLAGS += -I$(XILINX_XRT)/include
CLFLAGS += -I$(XILINX_VITIS)/include
LDFLAGS_VPP += --config ./system.cfg
LDFLAGS_VPP += -Iscripts/kernel
LDFLAGS_VPP += -Iscripts/host
LDFLAGS_VPP += -I$(XILINX_XRT)/include
LDFLAGS_VPP += -I$(XILINX_VITIS)/include

$(KERNEL_XO): $(KERNEL_SRC)
	@mkdir -p $(XCLBIN_DIR)
	$(VPP) -c -t $(TARGET) --platform $(DEVICE) $(CLFLAGS) -o $@ $<

$(XCLBIN_FILE): $(KERNEL_XO)
	$(VPP) -l -t $(TARGET) --platform $(DEVICE) $(LDFLAGS_VPP) -o $@ $<

emconfig:
	emconfigutil --platform $(DEVICE) --od .

.PHONY: emconfig

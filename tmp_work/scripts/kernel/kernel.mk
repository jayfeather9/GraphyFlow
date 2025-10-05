
# Makefile for the Vitis Kernel
VPP := v++
# KERNEL_NAMES is a list of all kernels to be compiled
KERNEL_NAMES := graphyflow_kernel global_controller

XCLBIN_DIR := ./xclbin
# Generate .xo file paths for each kernel
KERNEL_XOS := $(patsubst %,$(XCLBIN_DIR)/%.$(TARGET).xo,$(KERNEL_NAMES))
# Define the final .xclbin file
XCLBIN_FILES := $(XCLBIN_DIR)/graphyflow.$(TARGET).xclbin

EMCONFIG_FILE := ./emconfig.json

# Common flags
COMMON_FLAGS += -Iscripts/kernel
COMMON_FLAGS += -Iscripts/host
COMMON_FLAGS += -I$(XILINX_XRT)/include
COMMON_FLAGS += -I$(XILINX_VITIS)/include

# Rule to compile .cpp to .xo for each kernel
$(XCLBIN_DIR)/%.$(TARGET).xo: scripts/kernel/%.cpp
	@mkdir -p $(XCLBIN_DIR)
	$(VPP) -c -t $(TARGET) --platform $(DEVICE) --freqhz $(FREQ_HZ) --kernel $* $(COMMON_FLAGS) -o $@ $<

# Rule to link all .xo files to a single .xclbin
$(XCLBIN_FILES): $(KERNEL_XOS)
	$(VPP) -l -t $(TARGET) --platform $(DEVICE) --config ./system.cfg --include ./scripts/kernel/graphyflow_kernel.h $(COMMON_FLAGS) -o $@ $^

emconfig:
	emconfigutil --platform $(DEVICE) --od .

.PHONY: emconfig

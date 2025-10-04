
# Makefile for the Vitis Kernel
VPP := v++
# KERNEL_NAMES is a list of all kernels to be compiled
KERNEL_NAMES := graphyflow glb_controller

XCLBIN_DIR := ./xclbin
# Generate .xo and .xclbin file paths for each kernel
KERNEL_XOS := $(patsubst %,$(XCLBIN_DIR)/%.$(TARGET).xo,$(KERNEL_NAMES))
XCLBIN_FILES := $(patsubst %,$(XCLBIN_DIR)/%.$(TARGET).xclbin,$(KERNEL_NAMES))

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

# Rule to link .xo to .xclbin for each kernel
$(XCLBIN_DIR)/%.$(TARGET).xclbin: $(XCLBIN_DIR)/%.$(TARGET).xo
	$(VPP) -l -t $(TARGET) --platform $(DEVICE) --config ./system.cfg $(COMMON_FLAGS) -o $@ $<

emconfig:
	emconfigutil --platform $(DEVICE) --od .

.PHONY: emconfig

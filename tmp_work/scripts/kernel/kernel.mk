# Makefile for the Vitis Kernel

VPP := v++
KERNEL_NAME := bellman_ford_kernel
KERNEL_SRC := scripts/kernel/$(KERNEL_NAME).cpp
XCLBIN_DIR := ./xclbin
XCLBIN_FILE := $(XCLBIN_DIR)/bellman_ford.$(TARGET).xclbin
KERNEL_XO := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xo
EMCONFIG_FILE := ./emconfig.json

# V++ Compiler Flags
CLFLAGS += --kernel $(KERNEL_NAME)

# V++ Linker Flags
LDFLAGS_VPP += --config ./system.cfg

# Build rule for kernel object (.xo)
$(KERNEL_XO): $(KERNEL_SRC)
	@mkdir -p $(XCLBIN_DIR)
	$(VPP) -c -t $(TARGET) --platform $(DEVICE) $(CLFLAGS) -o $@ $<

# Build rule for binary container (.xclbin)
$(XCLBIN_FILE): $(KERNEL_XO)
	$(VPP) -l -t $(TARGET) --platform $(DEVICE) $(LDFLAGS_VPP) -o $@ $<

# Rule to generate emulation configuration
emconfig:
	emconfigutil --platform $(DEVICE) --od .

.PHONY: emconfig
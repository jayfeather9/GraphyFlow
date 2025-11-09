# GraphyFlow Multi-Merger Configuration Generator

This script automatically generates a multi-merger configuration for the GraphyFlow FPGA graph processing system.

## Overview

The current system has:
- 11 little kernels → 1 little merger → apply kernel
- 3 big kernels → 1 big merger → apply kernel
- apply kernel → hbm_writer

The generator allows you to define multiple mergers, where each merger can have a different number of pipelines and SLR assignments.

## Configuration Format

Define your configuration as a Python list:

```python
config = [
    {"kernel_type": "big", "pipeline_num": 3, "merger_slr": 1, "pipeline_slr": [2, 1, 2]},
    {"kernel_type": "little", "pipeline_num": 5, "merger_slr": 1, "pipeline_slr": [0, 1, 2, 0, 1]},
    {"kernel_type": "little", "pipeline_num": 6, "merger_slr": 1, "pipeline_slr": [0, 1, 2, 0, 1, 2]},
]
```

Each element defines:
- `kernel_type`: "big" or "little"
- `pipeline_num`: Number of pipelines connecting to this merger
- `merger_slr`: SLR assignment for the merger kernel
- `pipeline_slr`: List of SLR assignments for each pipeline (length must equal `pipeline_num`)

## What Gets Generated

1. **Template Files Copied**: All required source files are automatically copied from `templates/kernel/`
   - `graphyflow_big.cpp` and `graphyflow_big.h`
   - `graphyflow_little.cpp` and `graphyflow_little.h`
   - `hbm_writer.cpp`

2. **Merger Kernels**: One merger kernel per config element
   - `little_merger_0.cpp`, `little_merger_1.cpp`, etc.
   - `big_merger_0.cpp`, etc.

3. **Apply Kernel**: Modified to accept multiple merger streams
   - `apply_kernel.cpp` with multi-stream merge logic (generated from template)

4. **Shared Parameters**: Updated header with all merger declarations
   - `shared_kernel_params.h` (generated from template)

5. **System Configuration**: Complete connectivity and SLR assignments
   - `system.cfg` with all stream connections and SLR placements

6. **Kernel Makefile**: Updated with all kernel names
   - `scripts/kernel/kernel.mk`

7. **Host Configuration**: Kernel counts and HBM mappings
   - `scripts/host/host_config.h`

## Usage

### Option 1: Use example configuration (hardcoded in script)

```bash
cd /data/feiyang/tmp5_GraphyFlow/tmp_work2
python3 generate_config.py
```

### Option 2: Use JSON configuration file

Create a JSON file with your configuration:

```json
[
    {"kernel_type": "big", "pipeline_num": 3, "merger_slr": 1, "pipeline_slr": [2, 1, 2]},
    {"kernel_type": "little", "pipeline_num": 5, "merger_slr": 1, "pipeline_slr": [0, 1, 2, 0, 1]},
    {"kernel_type": "little", "pipeline_num": 6, "merger_slr": 1, "pipeline_slr": [0, 1, 2, 0, 1, 2]}
]
```

Then run:

```bash
python3 generate_config.py config.json
```

## Output Structure

After generation, the `tmp_work2` directory will contain:

```
tmp_work2/
├── templates/
│   └── kernel/
│       ├── graphyflow_big.cpp
│       ├── graphyflow_big.h
│       ├── graphyflow_little.cpp
│       ├── graphyflow_little.h
│       ├── hbm_writer.cpp
│       ├── shared_kernel_params.h.template
│       ├── apply_kernel.cpp.template
│       ├── little_merger.cpp.template
│       └── big_merger.cpp.template
├── scripts/
│   ├── kernel/
│   │   ├── graphyflow_big.cpp (copied from template)
│   │   ├── graphyflow_big.h (copied from template)
│   │   ├── graphyflow_little.cpp (copied from template)
│   │   ├── graphyflow_little.h (copied from template)
│   │   ├── hbm_writer.cpp (copied from template)
│   │   ├── little_merger_0.cpp (generated)
│   │   ├── little_merger_1.cpp (generated)
│   │   ├── big_merger_0.cpp (generated)
│   │   ├── apply_kernel.cpp (generated from template)
│   │   ├── shared_kernel_params.h (generated from template)
│   │   └── kernel.mk (generated)
│   └── host/
│       └── host_config.h (generated)
├── system.cfg (generated)
└── generate_config.py
```

**Note**: All template files are automatically copied during generation, so you don't need to manually copy them. The script ensures all required files are present for compilation.

## Host Code Updates

**IMPORTANT**: The host code templates need manual updates for multiple SP/DP arrays. The graph partitioning logic in `graph_preprocess.cpp` needs to be modified to:

1. Create separate `SPs_1`, `SPs_2`, `DPs_1`, `DPs_2` arrays based on the merger configuration
2. Partition the graph nodes based on the number of destination nodes for each merger
3. Update `generated_host.cpp` to handle multiple partition arrays

For the example configuration:
- Merger 0 (big, 3 pipelines): 524288 dst nodes → `SPs_0`
- Merger 1 (little, 5 pipelines): 65536 dst nodes → `DPs_0`
- Merger 2 (little, 6 pipelines): 65536 dst nodes → `DPs_1`

The host code should partition the graph into these separate arrays and initialize each merger accordingly.

## Notes

- The script automatically assigns kernel instance IDs sequentially
- Stream connections are automatically generated based on the configuration
- HBM port mappings are automatically calculated
- SLR assignments follow the `pipeline_slr` and `merger_slr` specifications
- The apply_kernel merge logic uses round-robin reading from all merger streams

## Verification

After generation, verify:
1. All kernel files compile correctly
2. `system.cfg` has correct stream connections
3. SLR assignments match your requirements
4. Host code is updated to handle multiple partition arrays


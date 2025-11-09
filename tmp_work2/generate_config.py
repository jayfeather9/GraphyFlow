#!/usr/bin/env python3
"""
Automatic generation script for GraphyFlow multi-merger configuration.

This script generates:
1. Multiple merger kernels (one per config element)
2. Apply kernel that accepts multiple merger streams
3. system.cfg with proper connections and SLR assignments
4. kernel.mk with all generated kernels
5. Host code templates for multiple SP/DP arrays
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Configuration example
CONFIG_EXAMPLE = [
    {"kernel_type": "big", "pipeline_num": 3, "merger_slr": 1, "pipeline_slr": [2, 1, 2]},
    {"kernel_type": "little", "pipeline_num": 5, "merger_slr": 1, "pipeline_slr": [0, 1, 2, 0, 1]},
    {"kernel_type": "little", "pipeline_num": 6, "merger_slr": 1, "pipeline_slr": [0, 1, 2, 0, 1, 2]},
]


class ConfigGenerator:
    def __init__(self, config: List[Dict[str, Any]], base_dir: Path, output_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.templates_dir = base_dir / "templates"
        self.scripts_dir = output_dir / "scripts"

        # Track kernel instances
        self.little_kernel_count = 0
        self.big_kernel_count = 0
        self.merger_count = 0
        self.merger_info = []  # List of (merger_id, kernel_type, pipeline_num, merger_slr, pipeline_slr)

        # Process config to assign kernel IDs
        self._process_config()

    def _process_config(self):
        """Process config to assign kernel instance IDs and track mergers."""
        for idx, elem in enumerate(self.config):
            kernel_type = elem["kernel_type"]
            pipeline_num = elem["pipeline_num"]
            merger_slr = elem["merger_slr"]
            pipeline_slr = elem["pipeline_slr"]

            if len(pipeline_slr) != pipeline_num:
                raise ValueError(
                    f"Config element {idx}: pipeline_slr length ({len(pipeline_slr)}) != pipeline_num ({pipeline_num})"
                )

            merger_id = self.merger_count
            self.merger_count += 1

            # Track which kernels belong to this merger
            kernel_start = {}
            if kernel_type == "little":
                kernel_start = self.little_kernel_count
                self.little_kernel_count += pipeline_num
            else:
                kernel_start = self.big_kernel_count
                self.big_kernel_count += pipeline_num

            self.merger_info.append(
                {
                    "merger_id": merger_id,
                    "kernel_type": kernel_type,
                    "pipeline_num": pipeline_num,
                    "merger_slr": merger_slr,
                    "pipeline_slr": pipeline_slr,
                    "kernel_start": kernel_start,
                }
            )

    def _get_kernel_slr(self, kernel_instance: str) -> int:
        """
        Get the SLR number for a kernel instance name.

        Args:
            kernel_instance: Kernel instance name (e.g., 'graphyflow_little_1', 'little_merger_0_1', 'hbm_writer_1')

        Returns:
            SLR number (0, 1, or 2)
        """
        # Handle graphyflow_little_N and graphyflow_big_N
        if kernel_instance.startswith("graphyflow_little_"):
            kernel_num = int(kernel_instance.split("_")[-1])
            # Find which merger this kernel belongs to and get its SLR
            little_kernel_idx = 0
            for merger in self.merger_info:
                if merger["kernel_type"] == "little":
                    pipeline_slr = merger["pipeline_slr"]
                    pipeline_num = merger["pipeline_num"]
                    # Check if this kernel belongs to this merger
                    if little_kernel_idx < kernel_num <= little_kernel_idx + pipeline_num:
                        # Get the position within this merger's pipeline
                        pos_in_merger = kernel_num - little_kernel_idx - 1
                        return pipeline_slr[pos_in_merger]
                    little_kernel_idx += pipeline_num
        elif kernel_instance.startswith("graphyflow_big_"):
            kernel_num = int(kernel_instance.split("_")[-1])
            # Find which merger this kernel belongs to and get its SLR
            big_kernel_idx = 0
            for merger in self.merger_info:
                if merger["kernel_type"] == "big":
                    pipeline_slr = merger["pipeline_slr"]
                    pipeline_num = merger["pipeline_num"]
                    # Check if this kernel belongs to this merger
                    if big_kernel_idx < kernel_num <= big_kernel_idx + pipeline_num:
                        # Get the position within this merger's pipeline
                        pos_in_merger = kernel_num - big_kernel_idx - 1
                        return pipeline_slr[pos_in_merger]
                    big_kernel_idx += pipeline_num
        # Handle merger instances (little_merger_N_1 or big_merger_N_1)
        elif kernel_instance.startswith("little_merger_") or kernel_instance.startswith("big_merger_"):
            parts = kernel_instance.split("_")
            merger_id = int(parts[2])
            for merger in self.merger_info:
                if merger["merger_id"] == merger_id:
                    return merger["merger_slr"]
        # Handle fixed kernels
        elif kernel_instance.startswith("hbm_writer_"):
            return 0  # Default SLR0
        elif kernel_instance.startswith("apply_kernel_"):
            return 1  # Default SLR1

        # Default fallback
        return 0

    def copy_template_files(self):
        """Copy all required template files to output directory."""
        import shutil

        kernel_templates = [
            "graphyflow_big.cpp",
            "graphyflow_big.h",
            "graphyflow_little.cpp",
            "graphyflow_little.h",
        ]

        for template_file in kernel_templates:
            src = self.templates_dir / "kernel" / template_file
            dst = self.scripts_dir / "kernel" / template_file

            if src.exists():
                shutil.copy2(src, dst)
                print(f"Copied template: {template_file}")
            else:
                print(f"Warning: Template file not found: {src}")

    def generate_all(self):
        """Generate all necessary files."""
        print(f"Generating configuration with {len(self.config)} mergers...")
        print(f"  Total little kernels: {self.little_kernel_count}")
        print(f"  Total big kernels: {self.big_kernel_count}")

        # Create directories
        (self.scripts_dir / "kernel").mkdir(parents=True, exist_ok=True)
        (self.scripts_dir / "host").mkdir(parents=True, exist_ok=True)

        # Copy template files first
        print("\nCopying template files...")
        self.copy_template_files()

        # Generate kernel files
        self.generate_merger_kernels()
        self.generate_apply_kernel()
        self.generate_hbm_writer()
        self.generate_shared_params()

        # Generate system.cfg
        self.generate_system_cfg()

        # Generate kernel.mk
        self.generate_kernel_mk()

        # Generate host files
        self.generate_host_config()
        self.generate_host_files()

        print("\nGeneration complete!")
        print(f"Output directory: {self.output_dir}")

    def generate_merger_kernels(self):
        """Generate merger kernel files for each config element."""
        for merger in self.merger_info:
            kernel_type = merger["kernel_type"]
            pipeline_num = merger["pipeline_num"]
            merger_id = merger["merger_id"]

            if kernel_type == "little":
                self._generate_little_merger(merger_id, pipeline_num)
            else:
                self._generate_big_merger(merger_id, pipeline_num)

    def _generate_little_merger(self, merger_id: int, pipeline_num: int):
        """Generate a little merger kernel."""
        template_path = self.templates_dir / "kernel" / "little_merger.cpp.template"
        output_path = self.scripts_dir / "kernel" / f"little_merger_{merger_id}.cpp"

        with open(template_path, "r") as f:
            content = f.read()

        # Generate stream parameter declarations for merge function
        merge_stream_params = []
        for i in range(pipeline_num):
            merge_stream_params.append(f"    hls::stream<little_out_pkt_t> &little_kernel_{i+1}_out_stream")

        # Generate stream parameter declarations for extern function
        extern_stream_params = []
        for i in range(pipeline_num):
            extern_stream_params.append(
                f"              hls::stream<little_out_pkt_t> &little_kernel_{i+1}_out_stream"
            )

        # Generate stream reads
        stream_reads = []
        for i in range(pipeline_num):
            stream_reads.append(
                f"""        if (!process_flag[{i}])
            process_flag[{i}] =
                little_kernel_{i+1}_out_stream.read_nb(tmp_prop_pkt[{i}]);"""
            )

        # Generate merge flag terms
        merge_flag_terms = [f"process_flag[{i}]" for i in range(pipeline_num)]

        # Generate function call parameters
        call_params = [f"little_kernel_{i+1}_out_stream" for i in range(pipeline_num)]

        # Replace placeholders
        content = content.replace("{{LITTLE_MERGER_LENGTH}}", str(pipeline_num))
        content = content.replace("{{LITTLE_MERGER_STREAM_PARAMS}}", ",\n".join(merge_stream_params) + ",")
        content = content.replace("{{STREAM_READS}}", "\n".join(stream_reads))
        content = content.replace("{{MERGE_FLAG_TERMS}}", " & ".join(merge_flag_terms))
        content = content.replace(
            "{{LITTLE_MERGER_MERGE_FUNCTION_NAME}}", f"merge_little_kernels_{merger_id}"
        )
        content = content.replace("{{LITTLE_MERGER_FUNCTION_NAME}}", f"little_merger_{merger_id}")
        content = content.replace("{{LITTLE_MERGER_EXTERN_PARAMS}}", ",\n".join(extern_stream_params) + ",")
        content = content.replace(
            "{{MERGE_FUNCTION_CALL}}",
            f"merge_little_kernels_{merger_id}(" + ", ".join(call_params) + ", kernel_out_stream);",
        )

        with open(output_path, "w") as f:
            f.write(content)

        print(f"Generated: {output_path}")

    def _generate_big_merger(self, merger_id: int, pipeline_num: int):
        """Generate a big merger kernel."""
        template_path = self.templates_dir / "kernel" / "big_merger.cpp.template"
        output_path = self.scripts_dir / "kernel" / f"big_merger_{merger_id}.cpp"

        with open(template_path, "r") as f:
            content = f.read()

        # Generate stream parameter declarations for merge function
        merge_stream_params = []
        for i in range(pipeline_num):
            merge_stream_params.append(f"    hls::stream<write_burst_pkt_t> &big_kernel_{i+1}_out_stream")

        # Generate stream parameter declarations for extern function
        extern_stream_params = []
        for i in range(pipeline_num):
            extern_stream_params.append(
                f"           hls::stream<write_burst_pkt_t> &big_kernel_{i+1}_out_stream"
            )

        # Generate stream reads
        stream_reads = []
        for i in range(pipeline_num):
            stream_reads.append(
                f"""        if (!process_flag[{i}])
            process_flag[{i}] = big_kernel_{i+1}_out_stream.read_nb(tmp_prop_pkt[{i}]);"""
            )

        # Generate merge flag terms
        merge_flag_terms = [f"process_flag[{i}]" for i in range(pipeline_num)]

        # Generate function call parameters
        call_params = [f"big_kernel_{i+1}_out_stream" for i in range(pipeline_num)]

        # Replace placeholders
        content = content.replace("{{MERGER_ID}}", str(merger_id))
        content = content.replace("{{BIG_MERGER_LENGTH}}", str(pipeline_num))
        content = content.replace("{{BIG_MERGER_STREAM_PARAMS}}", ",\n".join(merge_stream_params) + ",")
        content = content.replace("{{STREAM_READS}}", "\n".join(stream_reads))
        content = content.replace("{{MERGE_FLAG_TERMS}}", " & ".join(merge_flag_terms))
        content = content.replace("{{BIG_MERGER_FUNCTION_NAME}}", f"big_merger_{merger_id}")
        content = content.replace("{{BIG_MERGER_EXTERN_PARAMS}}", ",\n".join(extern_stream_params) + ",")
        content = content.replace(
            "{{MERGE_FUNCTION_CALL}}",
            f"merge_big_kernels_{merger_id}(" + ", ".join(call_params) + ", kernel_out_stream);",
        )

        with open(output_path, "w") as f:
            f.write(content)

        print(f"Generated: {output_path}")

    def generate_apply_kernel(self):
        """Generate apply kernel that accepts multiple merger streams."""
        template_path = self.templates_dir / "kernel" / "apply_kernel.cpp.template"
        output_path = self.scripts_dir / "kernel" / "apply_kernel.cpp"

        with open(template_path, "r") as f:
            content = f.read()

        # Count little and big mergers
        little_mergers = [m for m in self.merger_info if m["kernel_type"] == "little"]
        big_mergers = [m for m in self.merger_info if m["kernel_type"] == "big"]

        # Generate stream parameters for little mergers (ordered by merger_id)
        little_stream_params = []
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            little_stream_params.append(
                f"    hls::stream<write_burst_pkt_t> &little_merger_{merger_id}_out_stream"
            )

        # Generate stream parameters for big mergers (ordered by merger_id)
        big_stream_params = []
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            big_stream_params.append(f"    hls::stream<write_burst_pkt_t> &big_merger_{merger_id}_out_stream")

        # Generate length parameters
        little_length_params = []
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            little_length_params.append(f"    uint32_t little_merger_{merger_id}_length,")

        big_length_params = []
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            big_length_params.append(f"    uint32_t big_merger_{merger_id}_length,")

        # Generate offset parameters
        little_offset_params = []
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            little_offset_params.append(f"    uint32_t little_merger_{merger_id}_st_offset,")

        big_offset_params = []
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            big_offset_params.append(f"    uint32_t big_merger_{merger_id}_st_offset,")

        # Generate total length calculation
        total_length_lines = []
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            total_length_lines.append(f"    total_length += little_merger_{merger_id}_length;")
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            total_length_lines.append(f"    total_length += big_merger_{merger_id}_length;")

        # Generate index declarations (no remaining variables)
        index_declarations = []
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            index_declarations.append(
                f"    uint32_t little_merger_{merger_id}_idx = little_merger_{merger_id}_st_offset;"
            )
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            index_declarations.append(
                f"    uint32_t big_merger_{merger_id}_idx = big_merger_{merger_id}_st_offset;"
            )

        # Generate read blocks (only check read_nb, no remaining check)
        read_blocks = []
        sorted_little = sorted(little_mergers, key=lambda x: x["merger_id"])
        for i, merger in enumerate(sorted_little):
            merger_id = merger["merger_id"]
            if i == 0:
                read_blocks.append(
                    f"        if (little_merger_{merger_id}_out_stream.read_nb(tmp_prop_pkt)) {{"
                )
            else:
                read_blocks.append(
                    f"        else if (little_merger_{merger_id}_out_stream.read_nb(tmp_prop_pkt)) {{"
                )
            read_blocks.append(f"            in_write_burst_w_dst_pkt_t write_burst;")
            read_blocks.append(f"            write_burst.data = tmp_prop_pkt.data;")
            read_blocks.append(f"            write_burst.dest_addr = little_merger_{merger_id}_idx;")
            read_blocks.append(f"            write_burst.end_flag = false;")
            read_blocks.append(f"            kernel_out_stream.write(write_burst);")
            read_blocks.append(f"            little_merger_{merger_id}_idx++;")
            read_blocks.append(f"            total_length--;")
            read_blocks.append(f"        }}")

        sorted_big = sorted(big_mergers, key=lambda x: x["merger_id"])
        for i, merger in enumerate(sorted_big):
            merger_id = merger["merger_id"]
            if len(little_mergers) == 0 and i == 0:
                read_blocks.append(f"        if (big_merger_{merger_id}_out_stream.read_nb(tmp_prop_pkt)) {{")
            else:
                read_blocks.append(
                    f"        else if (big_merger_{merger_id}_out_stream.read_nb(tmp_prop_pkt)) {{"
                )
            read_blocks.append(f"            in_write_burst_w_dst_pkt_t write_burst;")
            read_blocks.append(f"            write_burst.data = tmp_prop_pkt.data;")
            read_blocks.append(f"            write_burst.dest_addr = big_merger_{merger_id}_idx;")
            read_blocks.append(f"            write_burst.end_flag = false;")
            read_blocks.append(f"            kernel_out_stream.write(write_burst);")
            read_blocks.append(f"            big_merger_{merger_id}_idx++;")
            read_blocks.append(f"            total_length--;")
            read_blocks.append(f"        }}")

        # Generate apply_kernel parameters
        apply_params = []
        apply_params.append("             uint32_t num_little_mergers,")
        apply_params.append("             uint32_t num_big_mergers,")
        apply_params.extend([f"             {p.rstrip(',')}," for p in little_length_params])
        apply_params.extend([f"             {p.rstrip(',')}," for p in big_length_params])
        apply_params.extend([f"             {p.rstrip(',')}," for p in little_offset_params])
        apply_params.extend([f"             {p.rstrip(',')}," for p in big_offset_params])
        for i, param in enumerate(little_stream_params):
            # Always add comma since either big stream params or kernel_out_stream follows
            apply_params.append(f"             {param.rstrip(',')},")
        for i, param in enumerate(big_stream_params):
            # Always add comma since kernel_out_stream follows
            apply_params.append(f"             {param.rstrip(',')},")
        apply_params.append("             hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream")

        # Generate interface pragmas
        interface_pragmas = []
        interface_pragmas.append("#pragma HLS INTERFACE s_axilite port = num_little_mergers bundle = control")
        interface_pragmas.append("#pragma HLS INTERFACE s_axilite port = num_big_mergers bundle = control")
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            interface_pragmas.append(
                f"#pragma HLS INTERFACE s_axilite port = little_merger_{merger_id}_length bundle = control"
            )
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            interface_pragmas.append(
                f"#pragma HLS INTERFACE s_axilite port = big_merger_{merger_id}_length bundle = control"
            )
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            interface_pragmas.append(
                f"#pragma HLS INTERFACE s_axilite port = little_merger_{merger_id}_st_offset bundle = control"
            )
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            interface_pragmas.append(
                f"#pragma HLS INTERFACE s_axilite port = big_merger_{merger_id}_st_offset bundle = control"
            )

        # Generate merge function call parameters
        call_params = []
        call_params.extend([f"little_merger_{m['merger_id']}_out_stream" for m in sorted_little])
        call_params.extend([f"big_merger_{m['merger_id']}_out_stream" for m in sorted_big])
        call_params.append("write_burst_stream")
        call_params.append("num_little_mergers")
        call_params.append("num_big_mergers")
        call_params.extend([f"little_merger_{m['merger_id']}_length" for m in sorted_little])
        call_params.extend([f"big_merger_{m['merger_id']}_length" for m in sorted_big])
        call_params.extend([f"little_merger_{m['merger_id']}_st_offset" for m in sorted_little])
        call_params.extend([f"big_merger_{m['merger_id']}_st_offset" for m in sorted_big])

        # Replace placeholders
        content = content.replace(
            "{{LITTLE_MERGER_STREAM_PARAMS}}",
            ",\n".join(little_stream_params) + "," if little_stream_params else "",
        )
        content = content.replace(
            "{{BIG_MERGER_STREAM_PARAMS}}", ",\n".join(big_stream_params) + "," if big_stream_params else ""
        )
        content = content.replace(
            "{{LITTLE_MERGER_LENGTH_PARAMS}}", "\n".join(little_length_params) if little_length_params else ""
        )
        content = content.replace(
            "{{BIG_MERGER_LENGTH_PARAMS}}", "\n".join(big_length_params) if big_length_params else ""
        )
        # Remove trailing comma from last offset param
        # If both little and big offset params exist, keep comma on last little param
        # but remove comma from last big param (it's the last param before closing paren)
        if little_offset_params and big_offset_params:
            # Keep comma on last little param since big params follow
            # Remove comma from last big param since it's the last param
            big_offset_params[-1] = big_offset_params[-1].rstrip(",")
        elif little_offset_params:
            little_offset_params[-1] = little_offset_params[-1].rstrip(",")
        elif big_offset_params:
            big_offset_params[-1] = big_offset_params[-1].rstrip(",")
        content = content.replace(
            "{{LITTLE_MERGER_OFFSET_PARAMS}}", "\n".join(little_offset_params) if little_offset_params else ""
        )
        content = content.replace(
            "{{BIG_MERGER_OFFSET_PARAMS}}", "\n".join(big_offset_params) if big_offset_params else ""
        )
        content = content.replace("{{TOTAL_LENGTH_CALCULATION}}", "\n".join(total_length_lines))
        content = content.replace("{{MERGER_INDEX_DECLARATIONS}}", "\n".join(index_declarations))
        content = content.replace("{{MERGER_READ_BLOCKS}}", "\n".join(read_blocks))
        content = content.replace("{{APPLY_KERNEL_PARAMS}}", "\n".join(apply_params))
        content = content.replace("{{INTERFACE_PRAGMAS}}", "\n".join(interface_pragmas))
        content = content.replace(
            "{{MERGE_FUNCTION_CALL_PARAMS}}", ",\n                            ".join(call_params)
        )

        with open(output_path, "w") as f:
            f.write(content)

        print(f"Generated: {output_path}")

    def generate_hbm_writer(self):
        """Generate hbm_writer.cpp from template based on kernel counts."""
        template_path = self.templates_dir / "kernel" / "hbm_writer.cpp.template"
        output_path = self.scripts_dir / "kernel" / "hbm_writer.cpp"

        with open(template_path, "r") as f:
            content = f.read()

        total_kernels = self.little_kernel_count + self.big_kernel_count

        # Generate src_prop parameters
        # Always add comma because output parameter comes after
        src_prop_params = []
        for i in range(1, total_kernels + 1):
            src_prop_params.append(f"    bus_word_t *src_prop_{i},")

        # Generate src_prop m_axi pragmas
        src_prop_m_axi_pragmas = []
        for i in range(1, total_kernels + 1):
            bundle_idx = (i - 1) % 14  # Cycle through gmem0-gmem13
            src_prop_m_axi_pragmas.append(
                f"#pragma HLS INTERFACE m_axi port = src_prop_{i} offset = slave bundle = gmem{bundle_idx}"
            )

        # Generate src_prop s_axilite pragmas
        src_prop_s_axilite_pragmas = []
        for i in range(1, total_kernels + 1):
            src_prop_s_axilite_pragmas.append(
                f"#pragma HLS INTERFACE s_axilite port = src_prop_{i} bundle = control"
            )

        # Generate PPB stream parameters
        # Add comma to last param only if there are big kernels (cacheline streams come after)
        ppb_stream_params = []
        for i in range(1, self.little_kernel_count + 1):
            ppb_stream_params.append(f"    hls::stream<ppb_request_pkt_t> &ppb_req_stream_{i},")
            ppb_stream_params.append(f"    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_{i},")
        # Remove comma from last param only if there are no big kernels (write_burst_stream comes after)
        if ppb_stream_params and self.big_kernel_count == 0:
            ppb_stream_params[-1] = ppb_stream_params[-1].rstrip(",")

        # Generate cacheline stream parameters
        # Always keep comma because write_burst_stream comes after
        cacheline_stream_params = []
        for i in range(1, self.big_kernel_count + 1):
            cacheline_stream_params.append(
                f"    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_{i},"
            )
            cacheline_stream_params.append(
                f"    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_{i},"
            )

        # Generate little_prop_loader_out stream declarations
        little_prop_loader_out_streams = []
        for i in range(1, self.little_kernel_count + 1):
            little_prop_loader_out_streams.append(
                f"    hls::stream<little_ppb_resp_t> little_prop_loader_out_{i};"
            )
            little_prop_loader_out_streams.append(
                f"#pragma HLS STREAM variable = little_prop_loader_out_{i} depth = 16"
            )
            little_prop_loader_out_streams.append(
                "    // #pragma HLS BIND_STORAGE variable = little_prop_loader_out_" + str(i) + " type = FIFO"
            )
            little_prop_loader_out_streams.append("    // impl = BRAM")
            if i < self.little_kernel_count:
                little_prop_loader_out_streams.append("")

        # Generate little_node_prop_loader and little_response_packer calls
        little_node_prop_loader_calls = []
        for i in range(self.little_kernel_count):
            idx = i + 1
            little_node_prop_loader_calls.append(
                f"    little_node_prop_loader({i}, src_prop_{idx}, num_partitions_little,"
            )
            little_node_prop_loader_calls.append(
                f"                            ppb_req_stream_{idx}, little_prop_loader_out_{idx});"
            )
            little_node_prop_loader_calls.append(
                f"    little_response_packer({i}, little_prop_loader_out_{idx}, ppb_resp_stream_{idx},"
            )
            little_node_prop_loader_calls.append("                           num_partitions_little);")
            if i < self.little_kernel_count - 1:
                little_node_prop_loader_calls.append("")

        # Generate big_node_prop_loader calls
        big_node_prop_loader_calls = []
        for i in range(self.big_kernel_count):
            idx = self.little_kernel_count + i + 1
            big_node_prop_loader_calls.append(
                f"    big_node_prop_loader({i}, src_prop_{idx}, num_partitions_big,"
            )
            big_node_prop_loader_calls.append(
                f"                         cacheline_req_stream_{i+1}, cacheline_resp_stream_{i+1});"
            )
            if i < self.big_kernel_count - 1:
                big_node_prop_loader_calls.append("")

        # Replace placeholders
        content = content.replace("{{SRC_PROP_PARAMS}}", "\n".join(src_prop_params))
        content = content.replace("{{SRC_PROP_M_AXI_PRAGMAS}}", "\n".join(src_prop_m_axi_pragmas))
        content = content.replace("{{SRC_PROP_S_AXILITE_PRAGMAS}}", "\n".join(src_prop_s_axilite_pragmas))
        content = content.replace(
            "{{PPB_STREAM_PARAMS}}", "\n".join(ppb_stream_params) if ppb_stream_params else ""
        )
        content = content.replace(
            "{{CACHELINE_STREAM_PARAMS}}",
            "\n".join(cacheline_stream_params) if cacheline_stream_params else "",
        )
        content = content.replace(
            "{{LITTLE_PROP_LOADER_OUT_STREAMS}}",
            "\n".join(little_prop_loader_out_streams) if little_prop_loader_out_streams else "",
        )
        content = content.replace(
            "{{LITTLE_NODE_PROP_LOADER_CALLS}}",
            "\n".join(little_node_prop_loader_calls) if little_node_prop_loader_calls else "",
        )
        content = content.replace(
            "{{BIG_NODE_PROP_LOADER_CALLS}}",
            "\n".join(big_node_prop_loader_calls) if big_node_prop_loader_calls else "",
        )

        with open(output_path, "w") as f:
            f.write(content)

        print(f"Generated: {output_path}")

    def generate_shared_params(self):
        """Generate shared_kernel_params.h with all merger declarations."""
        template_path = self.templates_dir / "kernel" / "shared_kernel_params.h.template"
        output_path = self.scripts_dir / "kernel" / "shared_kernel_params.h"

        # Read template file
        if template_path.exists():
            with open(template_path, "r") as f:
                content = f.read()
        else:
            # Fallback to original location if template doesn't exist
            original_path = (
                self.base_dir.parent / "tmp_work" / "scripts" / "kernel" / "shared_kernel_params.h"
            )
            with open(original_path, "r") as f:
                content = f.read()

        # Remove old merger declarations
        import re

        content = re.sub(r"#define BIG_MERGER_LENGTH \d+", "", content)
        content = re.sub(r"#define LITTLE_MERGER_LENGTH \d+", "", content)

        # Add merger length definitions (max values)
        max_little = max(
            [m["pipeline_num"] for m in self.merger_info if m["kernel_type"] == "little"], default=0
        )
        max_big = max([m["pipeline_num"] for m in self.merger_info if m["kernel_type"] == "big"], default=0)

        content = content.replace(
            "#define REDUCE_MEM_WIDTH 64",
            f"#define MAX_LITTLE_MERGER_LENGTH {max_little}\n#define MAX_BIG_MERGER_LENGTH {max_big}\n#define REDUCE_MEM_WIDTH 64",
        )

        # Remove old extern declarations
        content = re.sub(r'extern "C" void\s+big_merger\([^)]+\);', "", content)
        content = re.sub(r'extern "C" void\s+little_merger\([^)]+\);', "", content)

        # Count little and big mergers
        little_mergers = [m for m in self.merger_info if m["kernel_type"] == "little"]
        big_mergers = [m for m in self.merger_info if m["kernel_type"] == "big"]

        # Add new extern declarations for all mergers
        merger_decls = []
        for merger in self.merger_info:
            merger_id = merger["merger_id"]
            kernel_type = merger["kernel_type"]
            pipeline_num = merger["pipeline_num"]

            if kernel_type == "little":
                params = ", ".join(
                    [
                        f"hls::stream<little_out_pkt_t> &little_kernel_{i+1}_out_stream"
                        for i in range(pipeline_num)
                    ]
                )
                merger_decls.append(
                    f'extern "C" void\nlittle_merger_{merger_id}({params},\n              hls::stream<write_burst_pkt_t> &kernel_out_stream);'
                )
            else:
                params = ", ".join(
                    [
                        f"hls::stream<write_burst_pkt_t> &big_kernel_{i+1}_out_stream"
                        for i in range(pipeline_num)
                    ]
                )
                merger_decls.append(
                    f'extern "C" void\nbig_merger_{merger_id}({params},\n           hls::stream<write_burst_pkt_t> &kernel_out_stream);'
                )

        # Generate stream parameters for little mergers (ordered by merger_id)
        little_stream_params = []
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            little_stream_params.append(
                f"hls::stream<write_burst_pkt_t> &little_merger_{merger_id}_out_stream"
            )

        # Generate stream parameters for big mergers (ordered by merger_id)
        big_stream_params = []
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            big_stream_params.append(f"hls::stream<write_burst_pkt_t> &big_merger_{merger_id}_out_stream")

        # Remove old apply_kernel declaration (multi-line)
        old_apply_pattern = r'extern "C" void\s+apply_kernel\([^;]+\);'
        content = re.sub(old_apply_pattern, "", content, flags=re.DOTALL)

        # Generate new apply_kernel declaration
        apply_param_lines = [
            'extern "C" void',
            "apply_kernel(bus_word_t *node_props,",
            f"             uint32_t num_little_mergers,",
            f"             uint32_t num_big_mergers,",
        ]

        # Add length and offset parameters (using merger_id)
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            apply_param_lines.append(f"             uint32_t little_merger_{merger_id}_length,")
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            apply_param_lines.append(f"             uint32_t big_merger_{merger_id}_length,")
        for merger in sorted(little_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            apply_param_lines.append(f"             uint32_t little_merger_{merger_id}_st_offset,")
        for merger in sorted(big_mergers, key=lambda x: x["merger_id"]):
            merger_id = merger["merger_id"]
            apply_param_lines.append(f"             uint32_t big_merger_{merger_id}_st_offset,")

        # Add stream parameters
        for i, param in enumerate(little_stream_params):
            apply_param_lines.append(f"             {param},")
        for i, param in enumerate(big_stream_params):
            apply_param_lines.append(f"             {param},")
        apply_param_lines.append("             hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream);")

        apply_decl = "\n".join(apply_param_lines)

        # Generate hbm_writer declaration to match the implementation
        total_kernels = self.little_kernel_count + self.big_kernel_count

        # Generate src_prop parameters for declaration (all with commas)
        hbm_src_prop_params = []
        for i in range(1, total_kernels + 1):
            hbm_src_prop_params.append(f"    bus_word_t *src_prop_{i},")

        # Generate PPB stream parameters
        hbm_ppb_stream_params = []
        for i in range(1, self.little_kernel_count + 1):
            hbm_ppb_stream_params.append(f"    hls::stream<ppb_request_pkt_t> &ppb_req_stream_{i},")
            hbm_ppb_stream_params.append(f"    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_{i},")
        # Remove comma from last param only if there are no big kernels
        if hbm_ppb_stream_params and self.big_kernel_count == 0:
            hbm_ppb_stream_params[-1] = hbm_ppb_stream_params[-1].rstrip(",")

        # Generate cacheline stream parameters
        hbm_cacheline_stream_params = []
        for i in range(1, self.big_kernel_count + 1):
            hbm_cacheline_stream_params.append(
                f"    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_{i},"
            )
            hbm_cacheline_stream_params.append(
                f"    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_{i},"
            )

        # Build hbm_writer declaration
        hbm_writer_param_lines = ['extern "C" void hbm_writer(']
        hbm_writer_param_lines.extend(hbm_src_prop_params)
        hbm_writer_param_lines.append("    bus_word_t *output,")
        hbm_writer_param_lines.append("    uint32_t num_partitions_little, uint32_t num_partitions_big,")
        hbm_writer_param_lines.extend(hbm_ppb_stream_params)
        hbm_writer_param_lines.extend(hbm_cacheline_stream_params)
        hbm_writer_param_lines.append("    hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream);")

        hbm_writer_decl = "\n".join(hbm_writer_param_lines)

        # Find position of old hbm_writer declaration before removing it
        hbm_writer_pos = content.find('extern "C" void hbm_writer')

        # Remove old hbm_writer declaration (multi-line)
        old_hbm_writer_pattern = r'extern "C" void\s+hbm_writer\([^;]+\);'
        content = re.sub(old_hbm_writer_pattern, "", content, flags=re.DOTALL)

        # Insert merger declarations, apply_kernel declaration, and hbm_writer declaration
        if hbm_writer_pos > 0:
            # Insert at the position where old declaration was
            content = (
                content[:hbm_writer_pos]
                + "\n".join(merger_decls)
                + "\n\n"
                + apply_decl
                + "\n\n"
                + hbm_writer_decl
                + "\n"
                + content[hbm_writer_pos:]
            )
        else:
            # Fallback: insert at end before #endif
            endif_pos = content.rfind("#endif")
            if endif_pos > 0:
                content = (
                    content[:endif_pos]
                    + "\n".join(merger_decls)
                    + "\n\n"
                    + apply_decl
                    + "\n\n"
                    + hbm_writer_decl
                    + "\n"
                    + content[endif_pos:]
                )

        with open(output_path, "w") as f:
            f.write(content)

        print(f"Generated: {output_path}")

    def generate_system_cfg(self):
        """Generate system.cfg with all connections."""
        output_path = self.output_dir / "system.cfg"

        lines = ["[connectivity]"]
        lines.append("# --- 1. Kernel Instantiation (nk) ---")

        # Count kernels
        lines.append(f"nk=graphyflow_little:{self.little_kernel_count}")
        lines.append(f"nk=graphyflow_big:{self.big_kernel_count}")
        lines.append("nk=hbm_writer:1")
        lines.append("nk=apply_kernel:1")

        # Add merger instantiations
        for merger in self.merger_info:
            kernel_type = merger["kernel_type"]
            merger_id = merger["merger_id"]
            lines.append(f"nk={kernel_type}_merger_{merger_id}:1")

        lines.append("")
        lines.append("# --- 2. HBM Port Mapping (sp) ---")
        lines.append("")

        # HBM mapping for little kernels
        hbm_idx = 0
        for i in range(self.little_kernel_count):
            lines.append(f"# -- Mapping for instance: graphyflow_little_{i+1} --")
            lines.append(f"sp=graphyflow_little_{i+1}.edge_props:HBM[{hbm_idx}]")
            hbm_idx += 2

        # HBM mapping for big kernels
        for i in range(self.big_kernel_count):
            lines.append(f"# -- Mapping for instance: graphyflow_big_{i+1} --")
            lines.append(f"sp=graphyflow_big_{i+1}.edge_props:HBM[{hbm_idx}]")
            hbm_idx += 2

        # HBM mapping for hbm_writer (needs to be calculated based on total kernels)
        lines.append("")
        lines.append("# -- Mapping for instance: hbm_writer_1 --")
        total_kernels = self.little_kernel_count + self.big_kernel_count
        for i in range(1, total_kernels + 1):
            lines.append(f"sp=hbm_writer_1.src_prop_{i}:HBM[{2*i-1}]")
        lines.append(f"sp=hbm_writer_1.output:HBM[1]")

        # HBM mapping for apply_kernel
        lines.append("")
        lines.append("# -- Mapping for instance: apply_kernel_1 --")
        lines.append("sp=apply_kernel_1.node_props:HBM[30]")

        lines.append("")
        lines.append("# --- 3. Stream Connections ---")
        lines.append("")

        # Stream connections for kernels -> mergers
        little_kernel_idx = 0
        big_kernel_idx = 0

        for merger in self.merger_info:
            merger_id = merger["merger_id"]
            kernel_type = merger["kernel_type"]
            pipeline_num = merger["pipeline_num"]
            kernel_start = merger["kernel_start"]

            for i in range(pipeline_num):
                if kernel_type == "little":
                    kernel_num = little_kernel_idx + 1
                    little_kernel_idx += 1

                    # PPB streams to hbm_writer
                    lines.append(f"# -- Stream connections for graphyflow_little_{kernel_num} --")
                    # Check if same SLR
                    src_slr = self._get_kernel_slr(f"graphyflow_little_{kernel_num}")
                    dst_slr = self._get_kernel_slr("hbm_writer_1")
                    depth_suffix = ":16" if src_slr != dst_slr else ""
                    lines.append(
                        f"stream_connect=graphyflow_little_{kernel_num}.ppb_req_stream:hbm_writer_1.ppb_req_stream_{kernel_num}{depth_suffix}"
                    )
                    lines.append(
                        f"stream_connect=hbm_writer_1.ppb_resp_stream_{kernel_num}:graphyflow_little_{kernel_num}.ppb_resp_stream{depth_suffix}"
                    )

                    # Kernel output to merger
                    # Note: Since mergers are instantiated with :1, they get _1 suffix in compute unit names
                    dst_slr = self._get_kernel_slr(f"{kernel_type}_merger_{merger_id}_1")
                    depth_suffix = ":16" if src_slr != dst_slr else ""
                    lines.append(
                        f"stream_connect=graphyflow_little_{kernel_num}.kernel_out_stream:{kernel_type}_merger_{merger_id}_1.little_kernel_{i+1}_out_stream{depth_suffix}"
                    )
                else:
                    kernel_num = big_kernel_idx + 1
                    big_kernel_idx += 1

                    # Cacheline streams to hbm_writer
                    lines.append(f"# -- Stream connections for graphyflow_big_{kernel_num} --")
                    # Check if same SLR
                    src_slr = self._get_kernel_slr(f"graphyflow_big_{kernel_num}")
                    dst_slr = self._get_kernel_slr("hbm_writer_1")
                    depth_suffix = ":16" if src_slr != dst_slr else ""
                    lines.append(
                        f"stream_connect=graphyflow_big_{kernel_num}.cacheline_req_stream:hbm_writer_1.cacheline_req_stream_{kernel_num}{depth_suffix}"
                    )
                    lines.append(
                        f"stream_connect=hbm_writer_1.cacheline_resp_stream_{kernel_num}:graphyflow_big_{kernel_num}.cacheline_resp_stream{depth_suffix}"
                    )

                    # Kernel output to merger
                    # Note: Since mergers are instantiated with :1, they get _1 suffix in compute unit names
                    dst_slr = self._get_kernel_slr(f"{kernel_type}_merger_{merger_id}_1")
                    depth_suffix = ":16" if src_slr != dst_slr else ""
                    lines.append(
                        f"stream_connect=graphyflow_big_{kernel_num}.kernel_out_stream:{kernel_type}_merger_{merger_id}_1.big_kernel_{i+1}_out_stream{depth_suffix}"
                    )

        # Stream connections for mergers -> apply_kernel
        lines.append("")
        for merger in self.merger_info:
            merger_id = merger["merger_id"]
            kernel_type = merger["kernel_type"]

            if kernel_type == "little":
                lines.append(f"# -- Stream connections for little_merger_{merger_id} --")
                # Note: Since mergers are instantiated with :1, they get _1 suffix in compute unit names
                src_slr = self._get_kernel_slr(f"little_merger_{merger_id}_1")
                dst_slr = self._get_kernel_slr("apply_kernel_1")
                depth_suffix = ":16" if src_slr != dst_slr else ""
                lines.append(
                    f"stream_connect=little_merger_{merger_id}_1.kernel_out_stream:apply_kernel_1.little_merger_{merger_id}_out_stream{depth_suffix}"
                )
            else:
                lines.append(f"# -- Stream connections for big_merger_{merger_id} --")
                # Note: Since mergers are instantiated with :1, they get _1 suffix in compute unit names
                src_slr = self._get_kernel_slr(f"big_merger_{merger_id}_1")
                dst_slr = self._get_kernel_slr("apply_kernel_1")
                depth_suffix = ":16" if src_slr != dst_slr else ""
                lines.append(
                    f"stream_connect=big_merger_{merger_id}_1.kernel_out_stream:apply_kernel_1.big_merger_{merger_id}_out_stream{depth_suffix}"
                )

        # Stream connection for apply_kernel -> hbm_writer
        lines.append("")
        lines.append("# -- Stream connections for apply_kernel_1 --")
        src_slr = self._get_kernel_slr("apply_kernel_1")
        dst_slr = self._get_kernel_slr("hbm_writer_1")
        depth_suffix = ":16" if src_slr != dst_slr else ""
        lines.append(
            f"stream_connect=apply_kernel_1.kernel_out_stream:hbm_writer_1.write_burst_stream{depth_suffix}"
        )

        # SLR assignments
        lines.append("")
        lines.append("# --- 4. SLR Placement ---")

        # Assign SLRs to kernels based on pipeline_slr
        little_kernel_idx = 0
        big_kernel_idx = 0

        for merger in self.merger_info:
            kernel_type = merger["kernel_type"]
            pipeline_slr = merger["pipeline_slr"]

            for slr in pipeline_slr:
                if kernel_type == "little":
                    kernel_num = little_kernel_idx + 1
                    little_kernel_idx += 1
                    lines.append(f"slr=graphyflow_little_{kernel_num}:SLR{slr}")
                else:
                    kernel_num = big_kernel_idx + 1
                    big_kernel_idx += 1
                    lines.append(f"slr=graphyflow_big_{kernel_num}:SLR{slr}")

        # Assign SLRs to mergers
        # Note: Since all mergers are instantiated with :1, they get _1 suffix in compute unit names
        for merger in self.merger_info:
            merger_id = merger["merger_id"]
            kernel_type = merger["kernel_type"]
            merger_slr = merger["merger_slr"]
            lines.append(f"slr={kernel_type}_merger_{merger_id}_1:SLR{merger_slr}")

        # Assign SLRs to other kernels (defaults)
        lines.append("slr=hbm_writer_1:SLR0")
        lines.append("slr=apply_kernel_1:SLR1")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Generated: {output_path}")

    def generate_kernel_mk(self):
        """Generate kernel.mk with all kernel names."""
        template_path = self.templates_dir / "kernel" / "kernel.mk.template"
        output_path = self.scripts_dir / "kernel" / "kernel.mk"

        with open(template_path, "r") as f:
            content = f.read()

        # Build kernel names list
        kernel_names = ["graphyflow_little", "graphyflow_big", "apply_kernel", "hbm_writer"]

        # Add merger kernels
        for merger in self.merger_info:
            kernel_type = merger["kernel_type"]
            merger_id = merger["merger_id"]
            kernel_names.append(f"{kernel_type}_merger_{merger_id}")

        kernel_names_str = " ".join(kernel_names)
        content = content.replace(
            "KERNEL_NAMES := graphyflow_little graphyflow_big apply_kernel hbm_writer big_merger little_merger",
            f"KERNEL_NAMES := {kernel_names_str}",
        )

        with open(output_path, "w") as f:
            f.write(content)

        print(f"Generated: {output_path}")

    def generate_host_config(self):
        """Generate host_config.h with kernel counts."""
        output_path = self.scripts_dir / "host" / "host_config.h"

        lines = [
            "#ifndef __HOST_CONFIG_H__",
            "#define __HOST_CONFIG_H__",
            "",
            "#include <stdint.h>",
            "",
            f"#define BIG_KERNEL_NUM {self.big_kernel_count}",
            f"#define LITTLE_KERNEL_NUM {self.little_kernel_count}",
            "",
            "#define NUM_KERNEL (BIG_KERNEL_NUM + LITTLE_KERNEL_NUM)",
            "",
        ]

        # Generate HBM ID arrays
        little_hbm_edge = []
        little_hbm_node = []
        big_hbm_edge = []
        big_hbm_node = []

        hbm_idx = 0
        for i in range(self.little_kernel_count):
            little_hbm_edge.append(str(hbm_idx))
            little_hbm_node.append(str(hbm_idx + 1))
            hbm_idx += 2

        for i in range(self.big_kernel_count):
            big_hbm_edge.append(str(hbm_idx))
            big_hbm_node.append(str(hbm_idx + 1))
            hbm_idx += 2

        lines.append(f"#define LITTLE_KERNEL_HBM_EDGE_ID {{{', '.join(little_hbm_edge)}}}")
        lines.append(f"#define LITTLE_KERNEL_HBM_NODE_ID {{{', '.join(little_hbm_node)}}}")
        lines.append(f"#define BIG_KERNEL_HBM_EDGE_ID {{{', '.join(big_hbm_edge)}}}")
        lines.append(f"#define BIG_KERNEL_HBM_NODE_ID {{{', '.join(big_hbm_node)}}}")
        lines.append("")
        lines.append("#endif /* __HOST_CONFIG_H__ */")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Generated: {output_path}")

    def generate_host_files(self):
        """Generate host code files (templates need manual update for multiple SP/DP arrays)."""
        print("\nHost code generation:")
        print("  NOTE: Host code templates need manual updates for multiple SP/DP arrays.")
        print("  The graph partitioning logic needs to be updated to create separate")
        print("  SPs_1, SPs_2, DPs_1, DPs_2 arrays based on the merger configuration.")
        print("  This requires understanding the graph partitioning algorithm.")
        print("  Please review and update:")
        print("    - templates/host/generated_host.cpp.template")
        print("    - templates/host/generated_host.h.template")
        print("    - graph_preprocess logic")


def main():
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        # Use example config
        config = CONFIG_EXAMPLE
        print("Using example configuration. Provide JSON file as argument to use custom config.")

    base_dir = Path(__file__).parent
    output_dir = base_dir

    generator = ConfigGenerator(config, base_dir, output_dir)
    generator.generate_all()


if __name__ == "__main__":
    main()

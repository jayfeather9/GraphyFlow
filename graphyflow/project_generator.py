import shutil
from pathlib import Path
from typing import Any, Dict,List, Optional
import copy
import sys
import textwrap
from .global_graph import GlobalGraph
from .dataflow_ir import ComponentCollection
# from .backend_manager import BackendManager
from .newbackend_manager import BackendManager


# --- CONFIGURATION SECTION ---
INIT_VALUE = "16384.0"
# You can change the number of kernels and their HBM mapping here.

# 
from .kernel_numbers import NUM_BIG_KERNELS, NUM_LITTLE_KERNELS

# HBM channel IDs for Big Kernels. The list length must match NUM_BIG_KERNELS.
# big_kernel_hbm_edge_id = [22,24,26]
# big_kernel_hbm_node_id = [23,25,27] # for hbm writer

# little_kernel_hbm_edge_id = [0,2,4,6,8,10,12,14,16,18,20]
# little_kernel_hbm_node_id = [1,3,5,7,9,11,13,15,17,19,21] # for hbm writer

hbm_edge_ids = [0,2,4,6,8,10,12,14,16,18,20,22,24,26]
hbm_node_ids = [1,3,5,7,9,11,13,15,17,19,21,23,25,27]

little_kernel_hbm_edge_id = hbm_edge_ids[:NUM_LITTLE_KERNELS]
little_kernel_hbm_node_id = hbm_node_ids[:NUM_LITTLE_KERNELS]

big_kernel_hbm_edge_id = hbm_edge_ids[NUM_LITTLE_KERNELS:NUM_LITTLE_KERNELS+NUM_BIG_KERNELS]
big_kernel_hbm_node_id = hbm_node_ids[NUM_LITTLE_KERNELS:NUM_LITTLE_KERNELS+NUM_BIG_KERNELS]

apply_kernel_hbm_node_id = [30]
hbm_writer_output_id = [1]
# big_kernel_slr = ["SLR2","SLR1", "SLR2"]
# little_kernel_slr = ["SLR0","SLR1", "SLR2","SLR0","SLR1", "SLR2","SLR0","SLR1", "SLR2","SLR0","SLR1"]

slr_list = ["SLR0","SLR1", "SLR2","SLR0","SLR1", "SLR2","SLR0","SLR1", "SLR2","SLR0","SLR1","SLR2","SLR1","SLR2"]

big_kernel_slr = slr_list[NUM_LITTLE_KERNELS:NUM_LITTLE_KERNELS+NUM_BIG_KERNELS]
little_kernel_slr = slr_list[:NUM_LITTLE_KERNELS]

little_merger_slr = ["SLR1"]
big_merger_slr = ["SLR1"]

apply_kernel_slr = ["SLR1"]
hbm_writer_slr = ["SLR0"]
# 
# 
# 

# NUM_BIG_KERNELS = 1
# NUM_LITTLE_KERNELS = 1
# 
# # HBM channel IDs for Big Kernels. The list length must match NUM_BIG_KERNELS.
# big_kernel_hbm_edge_id = [22]
# big_kernel_hbm_node_id = [23] # for hbm writer
# 
# little_kernel_hbm_edge_id = [0]
# little_kernel_hbm_node_id = [1] # for hbm writer
# 
# apply_kernel_hbm_node_id = [30]
# hbm_writer_output_id = [1]
# big_kernel_slr = ["SLR2"]
# little_kernel_slr = ["SLR0"]
# 
# little_merger_slr = ["SLR1"]
# big_merger_slr = ["SLR1"]
# 
# apply_kernel_slr = ["SLR1"]
# hbm_writer_slr = ["SLR0"]
# 

# --- END CONFIGURATION SECTION ---


def _copy_and_template(src: Path, dest: Path, replacements: Dict[str, str]):
    """Reads a file, replaces placeholders, and writes to a new location."""
    content = src.read_text()
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    dest.write_text(content)



def _create_cfg(dest: Path, kernel_name: str):
    """
    根据 *全局配置变量* 动态生成 Vitis 连接配置文件，
    严格仿照 system.cfg 的格式和注释。

    Args:
        dest (Path): 要写入配置文件的目标路径。
        kernel_name (str): 内核的基本名称 (未使用, 但保留接口)。
    """

    # --- 1. 派生配置和验证 ---
    try:
        # 验证 Big Kernels
        assert len(big_kernel_hbm_edge_id) == NUM_BIG_KERNELS
        assert len(big_kernel_hbm_node_id) == NUM_BIG_KERNELS
        assert len(big_kernel_slr) == NUM_BIG_KERNELS

        # 验证 Little Kernels
        assert len(little_kernel_hbm_edge_id) == NUM_LITTLE_KERNELS
        assert len(little_kernel_hbm_node_id) == NUM_LITTLE_KERNELS
        assert len(little_kernel_slr) == NUM_LITTLE_KERNELS
        
        # 验证单实例内核
        assert len(little_merger_slr) == 1
        assert len(big_merger_slr) == 1
        assert len(apply_kernel_slr) == 1
        assert len(hbm_writer_slr) == 1
        assert len(apply_kernel_hbm_node_id) == 1
        
    except AssertionError as e:
        print(f"配置错误: 全局列表长度与 NUM_... 变量或实例数量不匹配。", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        return

    # --- 辅助函数：用于根据SLR差异自动添加FIFO (:16) ---
    def get_fifo_suffix(slr_a: str, slr_b: str) -> str:
        """如果SLR不同，则返回 ':16'，否则返回空字符串。"""
        return ":16" if slr_a != slr_b else ""

    content = []
    content.append("[connectivity]")

    # --- 1. Kernel Instantiation (nk) ---
    content.append("\n# --- 1. Kernel Instantiation (nk) ---")
    has_little = NUM_LITTLE_KERNELS > 0
    has_big = NUM_BIG_KERNELS > 0

    if has_little:
        content.append(f"nk=graphyflow_little:{NUM_LITTLE_KERNELS}")
    if has_big:
        content.append(f"nk=graphyflow_big:{NUM_BIG_KERNELS}")

    content.append(f"nk=hbm_writer:1")
    content.append(f"nk=apply_kernel:1")
    if has_little:
        content.append(f"nk=little_merger:1")
    if has_big:
        content.append(f"nk=big_merger:1")

    # --- 2. HBM Port Mapping (sp) ---
    content.append("\n# --- 2. HBM Port Mapping (sp) ---")

    # -- graphyflow_little (edge_props) --
    for i in range(NUM_LITTLE_KERNELS):
        instance_num = i + 1
        hbm_id = little_kernel_hbm_edge_id[i]
        content.append(f"\n# -- Mapping for instance: graphyflow_little_{instance_num} --")
        content.append(f"sp=graphyflow_little_{instance_num}.edge_props:HBM[{hbm_id}]")

    # -- graphyflow_big (edge_props) --
    for i in range(NUM_BIG_KERNELS):
        instance_num = i + 1
        hbm_id = big_kernel_hbm_edge_id[i]
        content.append(f"\n# -- Mapping for instance: graphyflow_big_{instance_num} --")
        content.append(f"sp=graphyflow_big_{instance_num}.edge_props:HBM[{hbm_id}]")

    # -- Mapping for instance: hbm_writer_1 --
    content.append(f"\n# -- Mapping for instance: hbm_writer_1 --")
    content.append(f"# little first then big")
    
    # little 
    for i in range(NUM_LITTLE_KERNELS):
        port_index = i + 1
        hbm_id = little_kernel_hbm_node_id[i]
        content.append(f"sp=hbm_writer_1.src_prop_{port_index}:HBM[{hbm_id}]")
        
    # big 
    for i in range(NUM_BIG_KERNELS):
        port_index = i + 1 + NUM_LITTLE_KERNELS
        hbm_id = big_kernel_hbm_node_id[i]
        content.append(f"sp=hbm_writer_1.src_prop_{port_index}:HBM[{hbm_id}]")
    
    # hbm_writer_1 output 
    content.append(f"sp=hbm_writer_1.output:HBM[{hbm_writer_output_id[0]}]") # reserved

    # -- Mapping for instance: apply_kernel_1 --
    content.append(f"\n# -- Mapping for instance: apply_kernel_1 --")
    content.append(f"sp=apply_kernel_1.node_props:HBM[{apply_kernel_hbm_node_id[0]}]")

    # --- 3. Stream Connections ---
    content.append("\n# --- 3. Stream Connections ---")


    writer_slr = hbm_writer_slr[0]
    l_merger_slr = little_merger_slr[0]
    b_merger_slr = big_merger_slr[0]
    app_slr = apply_kernel_slr[0]

    # -- Stream connections for little --
    for i in range(NUM_LITTLE_KERNELS):
        instance_num = i + 1
        kernel_slr = little_kernel_slr[i]
        
        # graphyflow (SLR[i]) <-> hbm_writer_1 (SLR0)
        fifo_writer = get_fifo_suffix(kernel_slr, writer_slr)
        # graphyflow (SLR[i]) <-> little_merger_1 (SLR1)
        fifo_merger = get_fifo_suffix(kernel_slr, l_merger_slr)
        
        content.append(f"\n# -- Stream connections for graphyflow_little_{instance_num} --")
        content.append(f"stream_connect=graphyflow_little_{instance_num}.ppb_req_stream:hbm_writer_1.ppb_req_stream_{instance_num}{fifo_writer}")
        content.append(f"stream_connect=hbm_writer_1.ppb_resp_stream_{instance_num}:graphyflow_little_{instance_num}.ppb_resp_stream{fifo_writer}")
        content.append(f"stream_connect=graphyflow_little_{instance_num}.kernel_out_stream:little_merger_1.little_kernel_{instance_num}_out_stream{fifo_merger}")

    # -- Stream connections for big --
    for i in range(NUM_BIG_KERNELS):
        instance_num = i + 1
        kernel_slr = big_kernel_slr[i]
        
        # graphyflow (SLR[i]) <-> hbm_writer_1 (SLR0)
        fifo_writer = get_fifo_suffix(kernel_slr, writer_slr)
        # graphyflow (SLR[i]) <-> big_merger_1 (SLR1)
        fifo_merger = get_fifo_suffix(kernel_slr, b_merger_slr)

        content.append(f"\n# -- Stream connections for graphyflow_big_{instance_num} --")
        content.append(f"stream_connect=graphyflow_big_{instance_num}.cacheline_req_stream:hbm_writer_1.cacheline_req_stream_{instance_num}{fifo_writer}")
        content.append(f"stream_connect=hbm_writer_1.cacheline_resp_stream_{instance_num}:graphyflow_big_{instance_num}.cacheline_resp_stream{fifo_writer}")
        content.append(f"stream_connect=graphyflow_big_{instance_num}.kernel_out_stream:big_merger_1.big_kernel_{instance_num}_out_stream{fifo_merger}")

    # -- Stream connections for mergers and apply_kernel --
    
    # little_merger_1 (SLR1) -> apply_kernel_1 (SLR1)
    if NUM_LITTLE_KERNELS > 0:
        fifo_l_merger_to_app = get_fifo_suffix(l_merger_slr, app_slr)
        content.append(f"\n# -- Stream connections for little_merger_1 --")
        content.append(f"stream_connect=little_merger_1.kernel_out_stream:apply_kernel_1.little_kernel_out_stream{fifo_l_merger_to_app}")

    # big_merger_1 (SLR1) -> apply_kernel_1 (SLR1)
    if NUM_BIG_KERNELS > 0:
        fifo_b_merger_to_app = get_fifo_suffix(b_merger_slr, app_slr)
        content.append(f"\n# -- Stream connections for big_merger_1 --")
        content.append(f"stream_connect=big_merger_1.kernel_out_stream:apply_kernel_1.big_kernel_out_stream{fifo_b_merger_to_app}")

    # apply_kernel_1 (SLR1) -> hbm_writer_1 (SLR0)
    fifo_app_to_writer = get_fifo_suffix(app_slr, writer_slr)
    content.append(f"\n# -- Stream connections for apply_kernel_1 --")
    content.append(f"stream_connect=apply_kernel_1.kernel_out_stream:hbm_writer_1.write_burst_stream{fifo_app_to_writer}")


    # --- 4. SLR Placement ---
    content.append("\n# --- 4. SLR Placement ---")
    
    for i in range(NUM_LITTLE_KERNELS):
        content.append(f"slr=graphyflow_little_{i+1}:{little_kernel_slr[i]}")
        
    for i in range(NUM_BIG_KERNELS):
        content.append(f"slr=graphyflow_big_{i+1}:{big_kernel_slr[i]}")


    content.append(f"slr=hbm_writer_1:{hbm_writer_slr[0]}")
    content.append(f"slr=apply_kernel_1:{apply_kernel_slr[0]}")
    if has_little:
        content.append(f"slr=little_merger_1:{little_merger_slr[0]}")
    if has_big:
        content.append(f"slr=big_merger_1:{big_merger_slr[0]}")

    # --- 5. 写入文件 ---
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
            f.write("\n") 
        print(f"successfully create cfg: {dest}")
    except IOError as e:
        print(f"error when writing: {dest}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"error when creating: {dest.parent}: {e}", file=sys.stderr)


def _format_apply_params(params: List[str]) -> str:
    if not params:
        return ""
    if len(params) == 1:
        return params[0]
    return ",\n             ".join(params)


def _build_apply_kernel_templates(has_little: bool, has_big: bool) -> Dict[str, str]:
    """Create apply_kernel template fragments for the current kernel counts."""

    params = ["bus_word_t *node_props"]
    interface_pragmas = [
        "#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem0",
        "#pragma HLS INTERFACE s_axilite port = node_props bundle = control",
    ]

    if has_little:
        params.append("uint32_t little_kernel_length")
        interface_pragmas.append(
            "#pragma HLS INTERFACE s_axilite port = little_kernel_length bundle = control"
        )
    if has_big:
        params.append("uint32_t big_kernel_length")
        interface_pragmas.append(
            "#pragma HLS INTERFACE s_axilite port = big_kernel_length bundle = control"
        )
    if has_little:
        params.append("uint32_t little_kernel_st_offset")
        interface_pragmas.append(
            "#pragma HLS INTERFACE s_axilite port = little_kernel_st_offset bundle = control"
        )
    if has_big:
        params.append("uint32_t big_kernel_st_offset")
        interface_pragmas.append(
            "#pragma HLS INTERFACE s_axilite port = big_kernel_st_offset bundle = control"
        )

    if has_little:
        params.append("hls::stream<write_burst_pkt_t> &little_kernel_out_stream")
    if has_big:
        params.append("hls::stream<write_burst_pkt_t> &big_kernel_out_stream")
    params.append("hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream")

    interface_pragmas.append(
        "#pragma HLS INTERFACE s_axilite port = return bundle = control"
    )

    apply_params = _format_apply_params(params)
    apply_interface_pragmas = "\n".join(interface_pragmas)

    if has_little and has_big:
        merge_function = textwrap.dedent(
            """
            void merge_big_little_writes(
                hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
                hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
                hls::stream<in_write_burst_w_dst_pkt_t> &kernel_out_stream,
                uint32_t little_kernel_length, uint32_t big_kernel_length,
                uint32_t little_kernel_st_offset, uint32_t big_kernel_st_offset) {
                write_burst_pkt_t big_tmp_prop_pkt;
                write_burst_pkt_t little_tmp_prop_pkt;

                uint32_t little_idx = little_kernel_st_offset;
                uint32_t big_idx = big_kernel_st_offset;
                uint32_t total_length = little_kernel_length + big_kernel_length;

            LOOP_MERGE_WRITES:
                while (true) {
                    if (total_length == 0) {
                        in_write_burst_w_dst_pkt_t end_pkt;
                        end_pkt.end_flag = true;
                        kernel_out_stream.write(end_pkt);
                        break;
                    }

                    if (little_kernel_out_stream.read_nb(little_tmp_prop_pkt)) {
                        in_write_burst_w_dst_pkt_t little_write_burst;
                        little_write_burst.data = little_tmp_prop_pkt.data;
                        little_write_burst.dest_addr = little_idx;
                        little_write_burst.end_flag = false;
                        kernel_out_stream.write(little_write_burst);
                        little_idx++;
                        total_length--;
                    } else if (big_kernel_out_stream.read_nb(big_tmp_prop_pkt)) {
                        in_write_burst_w_dst_pkt_t big_write_burst;
                        big_write_burst.data = big_tmp_prop_pkt.data;
                        big_write_burst.dest_addr = big_idx;
                        big_write_burst.end_flag = false;
                        kernel_out_stream.write(big_write_burst);
                        big_idx++;
                        total_length--;
                    }
                }
            }
            """
        )
        merge_call = (
            "merge_big_little_writes(little_kernel_out_stream, "
            "big_kernel_out_stream, write_burst_stream, "
            "little_kernel_length, big_kernel_length, "
            "little_kernel_st_offset, big_kernel_st_offset);"
        )
    elif has_little:
        merge_function = textwrap.dedent(
            """
            void merge_little_writes(
                hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
                hls::stream<in_write_burst_w_dst_pkt_t> &kernel_out_stream,
                uint32_t little_kernel_length,
                uint32_t little_kernel_st_offset) {
                write_burst_pkt_t little_tmp_prop_pkt;

                uint32_t little_idx = little_kernel_st_offset;
                uint32_t remaining = little_kernel_length;

            LOOP_MERGE_WRITES:
                while (true) {
                    if (remaining == 0) {
                        in_write_burst_w_dst_pkt_t end_pkt;
                        end_pkt.end_flag = true;
                        kernel_out_stream.write(end_pkt);
                        break;
                    }

                    if (little_kernel_out_stream.read_nb(little_tmp_prop_pkt)) {
                        in_write_burst_w_dst_pkt_t little_write_burst;
                        little_write_burst.data = little_tmp_prop_pkt.data;
                        little_write_burst.dest_addr = little_idx;
                        little_write_burst.end_flag = false;
                        kernel_out_stream.write(little_write_burst);
                        little_idx++;
                        remaining--;
                    }
                }
            }
            """
        )
        merge_call = (
            "merge_little_writes(little_kernel_out_stream, write_burst_stream, "
            "little_kernel_length, little_kernel_st_offset);"
        )
    elif has_big:
        merge_function = textwrap.dedent(
            """
            void merge_big_writes(
                hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
                hls::stream<in_write_burst_w_dst_pkt_t> &kernel_out_stream,
                uint32_t big_kernel_length, uint32_t big_kernel_st_offset) {
                write_burst_pkt_t big_tmp_prop_pkt;

                uint32_t big_idx = big_kernel_st_offset;
                uint32_t remaining = big_kernel_length;

            LOOP_MERGE_WRITES:
                while (true) {
                    if (remaining == 0) {
                        in_write_burst_w_dst_pkt_t end_pkt;
                        end_pkt.end_flag = true;
                        kernel_out_stream.write(end_pkt);
                        break;
                    }

                    if (big_kernel_out_stream.read_nb(big_tmp_prop_pkt)) {
                        in_write_burst_w_dst_pkt_t big_write_burst;
                        big_write_burst.data = big_tmp_prop_pkt.data;
                        big_write_burst.dest_addr = big_idx;
                        big_write_burst.end_flag = false;
                        kernel_out_stream.write(big_write_burst);
                        big_idx++;
                        remaining--;
                    }
                }
            }
            """
        )
        merge_call = (
            "merge_big_writes(big_kernel_out_stream, write_burst_stream, "
            "big_kernel_length, big_kernel_st_offset);"
        )
    else:
        merge_function = textwrap.dedent(
            """
            void merge_empty_writes(
                hls::stream<in_write_burst_w_dst_pkt_t> &kernel_out_stream) {
                in_write_burst_w_dst_pkt_t end_pkt;
                end_pkt.end_flag = true;
                kernel_out_stream.write(end_pkt);
            }
            """
        )
        merge_call = "merge_empty_writes(write_burst_stream);"

    return {
        "apply_params": apply_params,
        "apply_interface_pragmas": apply_interface_pragmas,
        "merge_function": merge_function,
        "merge_call": merge_call,
    }


def _build_apply_kernel_host_setargs(has_little: bool, has_big: bool) -> str:
    """Host-side setArg snippet matching the synthesized apply_kernel signature."""

    if has_little and has_big:
        block = textwrap.dedent(
            """
            int arg_idx = 0;
            OCL_CHECK(err, err = apply_kernel.setArg(
                                   arg_idx++, apply_kernel_node_prop_buffer));
            OCL_CHECK(err,
                      err = apply_kernel.setArg(arg_idx++, little_dst_word_num));
            OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, big_dst_word_num));
            OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, (uint32_t)0));
            OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, big_dst_offset));
            """
        )
        return textwrap.indent(block, "        ")

    if has_little:
        block = textwrap.dedent(
            """
            int arg_idx = 0;
            OCL_CHECK(err, err = apply_kernel.setArg(
                                   arg_idx++, apply_kernel_node_prop_buffer));
            OCL_CHECK(err,
                      err = apply_kernel.setArg(arg_idx++, little_dst_word_num));
            OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, (uint32_t)0));
            """
        )
        return textwrap.indent(block, "        ")

    if has_big:
        block = textwrap.dedent(
            """
            int arg_idx = 0;
            OCL_CHECK(err, err = apply_kernel.setArg(
                                   arg_idx++, apply_kernel_node_prop_buffer));
            OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, big_dst_word_num));
            OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, big_dst_offset));
            """
        )
        return textwrap.indent(block, "        ")

    block = textwrap.dedent(
        """
        int arg_idx = 0;
        OCL_CHECK(err, err = apply_kernel.setArg(
                               arg_idx++, apply_kernel_node_prop_buffer));
        """
    )
    return textwrap.indent(block, "        ")


def _generate_hbm_writer():
    code = "extern \"C\" void hbm_writer(\n"
    for i in range(NUM_LITTLE_KERNELS+NUM_BIG_KERNELS):
        code += f"    bus_word_t *src_prop_{i+1},\n"

    code += "    bus_word_t *output,\n"
    code += "    uint32_t num_partitions_little,\n"
    code += "    uint32_t num_partitions_big,\n"

    for i in range(NUM_LITTLE_KERNELS):
        code += f"    hls::stream<ppb_request_pkt_t> &ppb_req_stream_{i+1},\n"
        code += f"    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_{i+1},\n"
    for i in range(NUM_BIG_KERNELS):
        code += f"    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_{i+1},\n"
        code += f"    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_{i+1},\n"

   
    code += "hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream) {\n"

    for i in range(NUM_LITTLE_KERNELS+NUM_BIG_KERNELS):
        code += f"#pragma HLS INTERFACE m_axi port = src_prop_{i+1} offset = slave bundle = gmem{i}\n"
    code += "#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1\n"
    for i in range(NUM_LITTLE_KERNELS+NUM_BIG_KERNELS):
        code += f"#pragma HLS INTERFACE s_axilite port = src_prop_{i+1} bundle = control\n"
    

    code += "#pragma HLS INTERFACE s_axilite port = output bundle = control\n "
    code += "#pragma HLS INTERFACE s_axilite port = num_partitions_little bundle = control\n"
    code += "#pragma HLS INTERFACE s_axilite port = num_partitions_big bundle = control\n"
    code += "#pragma HLS INTERFACE s_axilite port = return bundle = control\n"
    code += "#pragma HLS DATAFLOW\n"

    for i in range(NUM_LITTLE_KERNELS):
        code += f"hls::stream<little_ppb_resp_t> little_prop_loader_out_{i+1};\n"
        code += f"#pragma HLS STREAM variable = little_prop_loader_out_{i+1} depth = 16\n"
    
    for i in range(NUM_LITTLE_KERNELS):
        code += f"    little_node_prop_loader({i}, src_prop_{i+1}, num_partitions_little,ppb_req_stream_{i+1}, little_prop_loader_out_{i+1});\n"
        code += f"    little_response_packer({i}, little_prop_loader_out_{i+1}, ppb_resp_stream_{i+1},num_partitions_little);\n"

    for i in range(NUM_BIG_KERNELS):
        code += f"    big_node_prop_loader({i}, src_prop_{i+NUM_LITTLE_KERNELS+1}, num_partitions_big,cacheline_req_stream_{i+1}, cacheline_resp_stream_{i+1});\n"

    code += "    write_out(output, write_burst_stream);\n"
    code += "}\n"

    return code
def _generate_little_merger(little_merger_inline_codes):

    code = "#include \"shared_kernel_params.h\"\n\n"
    code += "void merge_little_kernels(\n"

    for i in range(NUM_LITTLE_KERNELS):
        code += f"    hls::stream<little_out_pkt_t> &little_kernel_{i+1}_out_stream,\n"
    code += "hls::stream<write_burst_pkt_t> &kernel_out_stream){\n"

    code +=f"""
    little_out_pkt_t tmp_prop_pkt[LITTLE_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = tmp_prop_pkt dim = 0 complete

    bool process_flag[LITTLE_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = process_flag dim = 0 complete

    for (int i = 0; i < LITTLE_MERGER_LENGTH; i++) {{
#pragma HLS unroll
        process_flag[i] = 0;
    }}

    reduce_word_t merged_write_burst;

    bus_word_t one_write_burst;

    uint32_t inner_idx = 0;

    distance_t max_val = (distance_t)({INIT_VALUE});
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);

    //distance_t cc_ini = (distance_t)(0.0);
    //ap_fixed_pod_t cc_ini_pod = *reinterpret_cast<ap_fixed_pod_t *>(&cc_ini);

merge_tmp_prop_big_krnls:
    while (true) {{
#pragma HLS pipeline style = flp
"""
    for i in range(NUM_LITTLE_KERNELS):
        code += f"""
        if (!process_flag[{i}])
            process_flag[{i}] =
                little_kernel_{i+1}_out_stream.read_nb(tmp_prop_pkt[{i}]);"""

    code += "\n        bool merge_flag = \n"

    for i in range(NUM_LITTLE_KERNELS):
        code += f"        process_flag[{i}] & \n"
    
    code += "        1;\n"

    code += f"""
if (merge_flag) {{
            ap_fixed_pod_t uram_high = max_pod;;
            ap_fixed_pod_t uram_low = max_pod;

            //ap_fixed_pod_t uram_high = cc_ini_pod;//max_pod;;
            //ap_fixed_pod_t uram_low = cc_ini_pod;//max_pod;

            for (int i = 0; i < LITTLE_MERGER_LENGTH; i++) {{
#pragma HLS UNROLL
                ap_fixed_pod_t update_low = tmp_prop_pkt[i].data.range(31, 0);
                ap_fixed_pod_t update_high = tmp_prop_pkt[i].data.range(63, 32);

                {little_merger_inline_codes}
            }}

            merged_write_burst.range(31, 0) = uram_low;
            merged_write_burst.range(63, 32) = uram_high;

            one_write_burst.range(63 + (inner_idx << 6), (inner_idx << 6)) =
                merged_write_burst;
            inner_idx++;

            if (inner_idx == 8) {{
                write_burst_pkt_t out_pkt;
                out_pkt.data = one_write_burst;
                out_pkt.last = 0;
                kernel_out_stream.write(out_pkt);
                inner_idx = 0;
                one_write_burst = 0;
            }}

            for (int i = 0; i < LITTLE_MERGER_LENGTH; i++) {{
#pragma HLS unroll
                process_flag[i] = 0;
            }}
        }}
    }}
}}
"""
    code +="\n\n"

    code += "extern \"C\" void\n"
    code += "little_merger(\n"

    for i in range(NUM_LITTLE_KERNELS):
        code += f"    hls::stream<little_out_pkt_t> &little_kernel_{i+1}_out_stream,\n"

    code += "    hls::stream<write_burst_pkt_t> &kernel_out_stream) {\n"

    code += "#pragma HLS interface ap_ctrl_none port = return\n"
    code += "#pragma HLS DATAFLOW\n"
    code += "    merge_little_kernels(\n"

    for i in range(NUM_LITTLE_KERNELS):
        code += f"        little_kernel_{i+1}_out_stream,\n"
    code += "        kernel_out_stream);\n"
    code += "}\n"

    return code

def _generate_big_merger(big_merger_inline_codes):
    code = "#include \"shared_kernel_params.h\"\n\n"

    code += "void merge_big_kernels("

    for i in range(NUM_BIG_KERNELS):
        code += f"hls::stream<write_burst_pkt_t> &big_kernel_{i+1}_out_stream,\n"
    
    code += "hls::stream<write_burst_pkt_t> &kernel_out_stream) {\n"

    code += f"""
 write_burst_pkt_t tmp_prop_pkt[BIG_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = tmp_prop_pkt dim = 0 complete

    bool process_flag[BIG_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = process_flag dim = 0 complete

    for (int i = 0; i < BIG_MERGER_LENGTH; i++) {{
#pragma HLS unroll
        process_flag[i] = 0;
    }}

    bus_word_t merged_write_burst;

    write_burst_pkt_t one_write_burst;

    uint32_t outer_idx = 0;

    ap_fixed_pod_t tmp_prop_arrary[16];
#pragma HLS ARRAY_PARTITION variable = tmp_prop_arrary dim = 0 complete

    distance_t max_val = (distance_t)({INIT_VALUE});
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);


    // distance_t cc_ini = (distance_t)(0.0);
    // ap_fixed_pod_t cc_ini_pod = *reinterpret_cast<ap_fixed_pod_t *>(&cc_ini);

merge_tmp_prop_big_krnls:
    while (true) {{
#pragma HLS pipeline style = flp
"""
    for i in range(NUM_BIG_KERNELS):
        code += f"""
        if (!process_flag[{i}])
                    process_flag[{i}] = big_kernel_{i+1}_out_stream.read_nb(tmp_prop_pkt[{i}]);"""
    code += "\n    bool merge_flag = \n"
    for i in range(NUM_BIG_KERNELS):
        code += f"        process_flag[{i}] & \n"
    code += "                        1;\n"

    code += f"""
if (merge_flag) {{
            for (int i = 0; i < 16; i++) {{
#pragma HLS UNROLL
                tmp_prop_arrary[i] = max_pod;
                
            }}

            for (int i = 0; i < BIG_MERGER_LENGTH; i++) {{
#pragma HLS UNROLL
                for (int j = 0; j < 16; j++) {{
#pragma HLS UNROLL
                    ap_fixed_pod_t update =
                        tmp_prop_pkt[i].data.range(31 + (j << 5), (j << 5));
                    {big_merger_inline_codes}
                }}
            }}

            for (int i = 0; i < 16; i++) {{
#pragma HLS UNROLL
                merged_write_burst.range(31 + (i << 5), (i << 5)) =
                    tmp_prop_arrary[i];
            }}

            one_write_burst.data = merged_write_burst;
            kernel_out_stream.write(one_write_burst);

            for (int i = 0; i < BIG_MERGER_LENGTH; i++) {{
#pragma HLS unroll
                process_flag[i] = 0;
            }}
        }}
    }}
}}
"""
    code +="\n\n"
    code +="extern \"C\" void\n"
    code +="big_merger(\n"
    for i in range(NUM_BIG_KERNELS):
        code += f"    hls::stream<write_burst_pkt_t> &big_kernel_{i+1}_out_stream,\n"
    code +="    hls::stream<write_burst_pkt_t> &kernel_out_stream) {\n"
    code +="#pragma HLS interface ap_ctrl_none port = return\n"
    code +="#pragma HLS DATAFLOW\n"
    code +="    merge_big_kernels(\n"
    for i in range(NUM_BIG_KERNELS):
        code += f"        big_kernel_{i+1}_out_stream,\n"
    code +="        kernel_out_stream);\n"
    code +="}\n"

    return code

def _generate_shared_params():
    apply_templates = _build_apply_kernel_templates(
        has_little=NUM_LITTLE_KERNELS > 0, has_big=NUM_BIG_KERNELS > 0
    )

    code = "extern \"C\" void\n"
    code += f"apply_kernel({apply_templates['apply_params']});\n\n"
    code += "extern \"C\" void\n"
    code += "big_merger("
    for i in range(NUM_BIG_KERNELS):
        code += f"hls::stream<write_burst_pkt_t> &big_kernel_{i+1}_out_stream,\n"
    code += "hls::stream<write_burst_pkt_t> &kernel_out_stream);\n\n"

    code += "extern \"C\" void\n"
    code += "little_merger("
    for i in range(NUM_LITTLE_KERNELS):
        code += f"hls::stream<little_out_pkt_t> &little_kernel_{i+1}_out_stream,\n"
    code += "hls::stream<write_burst_pkt_t> &kernel_out_stream);\n\n"

    code += "extern \"C\" void\n"
    code += "hbm_writer(\n"
    for i in range(NUM_LITTLE_KERNELS+NUM_BIG_KERNELS):
        code += f"    bus_word_t *src_props_{i+1},\n"

    code += "    bus_word_t *output,\n"
    code += "    uint32_t num_partitions_little,\n"
    code += "    uint32_t num_partitions_big,\n"
    for i in range(NUM_LITTLE_KERNELS):
        code += f"    hls::stream<ppb_request_pkt_t> &ppb_req_stream_{i+1},\n"
        code += f"    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_{i+1},\n"
    for i in range(NUM_BIG_KERNELS):
        code += f"    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_{i+1},\n"
        code += f"    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_{i+1},\n"
    
    code += "    hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream);\n\n"

    code += "#endif // SHARED_KERNEL_PARAMS_H\n"
    return code

def fill_host_config(file_to_modify: Path):
    """
    Reads the specified file, replaces placeholders with the content of global variables,
    and writes the modified content back to the original file.
    """
    try:
        if not file_to_modify.is_file():
            print(f"Error: File not found '{file_to_modify}'")
            return

        replacements = {
            "%BIG_KERNEL_NUM%": NUM_BIG_KERNELS,
            "%LITTLE_KERNEL_NUM%": NUM_LITTLE_KERNELS,
            "%BIG_KERNEL_HBM_EDGE_ID%": big_kernel_hbm_edge_id,
            "%BIG_KERNEL_HBM_NODE_ID%": big_kernel_hbm_node_id,

            "%LITTLE_KERNEL_HBM_EDGE_ID%": little_kernel_hbm_edge_id,
            "%LITTLE_KERNEL_HBM_NODE_ID%": little_kernel_hbm_node_id,
        }

        original_content = file_to_modify.read_text(encoding="utf-8")
        modified_content = original_content

        for placeholder, value in replacements.items():
            replacement_string = ""
            if isinstance(value, list):
                replacement_string = f"{{{', '.join(map(str, value))}}}"
            elif isinstance(value, int):
                replacement_string = str(value)

            if replacement_string:
                modified_content = modified_content.replace(placeholder, replacement_string)

        file_to_modify.write_text(modified_content, encoding="utf-8")

    except Exception as e:
        print(f"An error occurred while processing file '{file_to_modify}': {e}")


def generate_project(
    comp_col: ComponentCollection,
    global_graph: Any,
    kernel_name: str,
    output_dir: Path,
    executable_name: str = "host",
    template_dir_override: Optional[Path] = None,
):
    """
    Generates a complete Vitis project directory from a DFG-IR.
    This version includes templating for build and run scripts.
    """
    print(f"--- Starting Project Generation for Kernel '{kernel_name}' ---")

    # 1. Define paths
    if template_dir_override:
        template_dir = template_dir_override
    else:
        # Assuming this script is in graphyflow/
        template_dir = Path(__file__).parent / "project_template"

    if not template_dir.exists():
        raise FileNotFoundError(f"Project template directory not found at: {template_dir}")

    # 2. Create output directory structure
    print(f"[1/6] Setting up Output Directory: '{output_dir}'")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    scripts_dir = output_dir / "scripts"
    host_script_dir = scripts_dir / "host"
    kernel_script_dir = scripts_dir / "kernel"

    # Copy the entire template directory first
    shutil.copytree(template_dir, output_dir, dirs_exist_ok=True)

    # Ensure specific directories exist after copy
    host_script_dir.mkdir(parents=True, exist_ok=True)
    kernel_script_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "xclbin").mkdir(exist_ok=True)

    # 3. Fill dynamic configurations
    print("[2/6] Filling Dynamic Configuration Files...")
    fill_host_config(host_script_dir / "host_config.h")

    # Call the new dynamic system.cfg generator
    _create_cfg(output_dir/"system.cfg", kernel_name)

    replacements = {
        "{{EXECUTABLE_NAME}}": executable_name,
    }
    _copy_and_template(template_dir / "Makefile", output_dir / "Makefile", replacements)
    _copy_and_template(template_dir / "run.sh", output_dir / "run.sh", replacements)
    (output_dir / "run.sh").chmod(0o755)

    has_big = NUM_BIG_KERNELS > 0
    has_little = NUM_LITTLE_KERNELS > 0

    kernel_names = ["apply_kernel", "hbm_writer"]
    if has_big:
        kernel_names.insert(0, "graphyflow_big")
        kernel_names.append("big_merger")
    if has_little:
        kernel_names.insert(0, "graphyflow_little")
        kernel_names.append("little_merger")

    _copy_and_template(
        template_dir / "scripts" / "kernel" / "kernel.mk",
        output_dir / "scripts" / "kernel" / "kernel.mk",
        {"{{KERNEL_NAMES}}": " ".join(kernel_names)},
    )

    # 4. Instantiate backend and generate all dynamic code
    print("[3/6] Generating Dynamic Source Code via BackendManager...")
    bkd_mng = BackendManager()

    bkd_mng.big_kernel_num = NUM_BIG_KERNELS
    bkd_mng.little_kernel_num = NUM_LITTLE_KERNELS
    bkd_mng.init_value = INIT_VALUE

    # Perform type analysis once
    # bkd_mng.analyze_graph_types(comp_col, global_graph)

    # Generate Big Kernel
    bkd_mng.REDUCE_MODE = "big_pipeline"
    kernel_h_big, kernel_h_little, shared_kernel_params,kernel_cpp_big,kernel_cpp_little,apply_func,little_merger_inline_codes,big_merger_inline_codes = bkd_mng.generate_backend(
        copy.deepcopy(comp_col), global_graph, f"{kernel_name}_big"
    )

    apply_templates = _build_apply_kernel_templates(
        has_little=has_little, has_big=has_big
    )
    replacements = {
        "{{GRAPHYFLOW_APPLY_FUNC}}": apply_func,
        "{{MERGE_FUNCTION}}": apply_templates["merge_function"],
        "{{APPLY_PARAMS}}": apply_templates["apply_params"],
        "{{APPLY_INTERFACE_PRAGMAS}}": apply_templates["apply_interface_pragmas"],
        "{{MERGE_CALL}}": apply_templates["merge_call"],
    }
    _copy_and_template(
        template_dir / "scripts" / "kernel" / "apply_kernel.cpp",
        output_dir / "scripts" / "kernel" / "apply_kernel.cpp",
        replacements,
    )

    host_apply_setargs = _build_apply_kernel_host_setargs(
        has_little=has_little, has_big=has_big
    )
    _copy_and_template(
        template_dir / "scripts" / "host" / "generated_host.cpp",
        output_dir / "scripts" / "host" / "generated_host.cpp",
        {"{{APPLY_KERNEL_SETARGS}}": host_apply_setargs},
    )

    hbm_writer = _generate_hbm_writer()
    little_merger = _generate_little_merger(little_merger_inline_codes)
    big_merger = _generate_big_merger(big_merger_inline_codes)
    # 5. Deploy all dynamically generated files
    print(f"[4/6] Deploying Generated Kernel Files to '{kernel_script_dir}'")
    if has_big:
        (kernel_script_dir / f"{kernel_name}_big.h").write_text(kernel_h_big)
        (kernel_script_dir / f"{kernel_name}_big.cpp").write_text(kernel_cpp_big)
        (kernel_script_dir / f"big_merger.cpp").write_text(big_merger)

    if has_little:
        (kernel_script_dir / f"{kernel_name}_little.h").write_text(kernel_h_little)
        (kernel_script_dir / f"{kernel_name}_little.cpp").write_text(kernel_cpp_little)
        (kernel_script_dir / f"little_merger.cpp").write_text(little_merger)

    (kernel_script_dir / f"shared_kernel_params.h").write_text(shared_kernel_params)
    
    hbm_writer_file = kernel_script_dir / f"hbm_writer.cpp"
    with open(hbm_writer_file, 'a', encoding='utf-8') as f:
        f.write("\n") 
        f.write(hbm_writer)
    shared_params_file = kernel_script_dir / f"shared_kernel_params.h"
    with open(shared_params_file, 'a', encoding='utf-8') as f:
        f.write("\n") 
        f.write(_generate_shared_params())
    


    print("[6/6] Project Generation Complete!")

#include "shared_kernel_params.h"

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

{{GRAPHYFLOW_APPLY_FUNC}}

extern "C" void
apply_kernel(bus_word_t *node_props, uint32_t little_kernel_length,
             uint32_t big_kernel_length, uint32_t little_kernel_st_offset,
             uint32_t big_kernel_st_offset,
             hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
             hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
             hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream) {
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = node_props bundle = control
#pragma HLS INTERFACE s_axilite port = little_kernel_length bundle = control
#pragma HLS INTERFACE s_axilite port = big_kernel_length bundle = control
#pragma HLS INTERFACE s_axilite port = little_kernel_st_offset bundle = control
#pragma HLS INTERFACE s_axilite port = big_kernel_st_offset bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    hls::stream<in_write_burst_w_dst_pkt_t> write_burst_stream;
#pragma HLS STREAM variable = write_burst_stream depth = 16

    merge_big_little_writes(little_kernel_out_stream, big_kernel_out_stream,
                            write_burst_stream, little_kernel_length,
                            big_kernel_length, little_kernel_st_offset,
                            big_kernel_st_offset);
    apply_func(node_props, write_burst_stream, kernel_out_stream);
}
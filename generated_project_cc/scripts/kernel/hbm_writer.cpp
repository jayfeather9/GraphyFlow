#include "shared_kernel_params.h"

static void little_node_prop_loader(
    int i, const bus_word_t *node_distances_ddr, uint32_t num_partitions,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream
    // hls::stream<cacheline_data_pkt_t> &cacheline_data_stream
) {
#pragma HLS function_instantiate variable = i

    ppb_request_pkt_t one_ppb_request_pkg;
    ppb_response_pkt_t one_ppb_response_pkg;
    uint32_t left_partitions = num_partitions;

littleKernelReadMemory:
    while (true) {
#pragma HLS PIPELINE
        if (ppb_req_stream.read_nb(one_ppb_request_pkg)) {
            uint32_t request_round = one_ppb_request_pkg.data;
            bool end_flag = one_ppb_request_pkg.last;

            uint32_t base_addr = request_round << LOG_SRC_BUFFER_SIZE >> 4;

            if (end_flag) {
                one_ppb_response_pkg.last = end_flag;
                ppb_resp_stream.write(one_ppb_response_pkg);
                left_partitions--;
                if (left_partitions == 0) {
                    break;
                }
            } else {
                for (int i = 0; i < (SRC_BUFFER_SIZE >> 4); i++) {
                    int addr = base_addr + i;

                    one_ppb_response_pkg.data = node_distances_ddr[addr];
                    one_ppb_response_pkg.dest = addr;
                    one_ppb_response_pkg.last = false;
                    ppb_resp_stream.write(one_ppb_response_pkg);
                }
            }
        }
    }
}

static void big_node_prop_loader(
    int i, const bus_word_t *node_distances_ddr, uint32_t num_partitions,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream
    // hls::stream<cacheline_data_pkt_t> &cacheline_data_stream
) {
#pragma HLS function_instantiate variable = i

    cacheline_request_pkt_t cache_req;
    cacheline_response_pkt_t cache_resp;

    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx = -1;
    bus_word_t last_cacheline;

    uint32_t left_partitions = num_partitions;

LOOP_BIG_KRL_READ_MEMORY:
    while (true) {
#pragma HLS PIPELINE II = 1
        bool process_flag = cacheline_req_stream.read_nb(cache_req);

        ap_uint<26> idx = cache_req.data;
        ap_uint<8> target_pe = cache_req.dest;
        bool end_flag = cache_req.last;

        ap_uint<8> dst_pe;
        bus_word_t out_data;
        bool out_end_flag;

        if (process_flag) {
            if (end_flag) {
                out_data = 0;
                last_cache_idx = -1;
                left_partitions--;
            } else {
                if (idx == last_cache_idx) {
                    out_data = last_cacheline;
                } else {
                    out_data = node_distances_ddr[idx];
                    last_cache_idx = idx;
                    last_cacheline = out_data;
                }
            }

            out_end_flag = end_flag;
            dst_pe = target_pe;

            cache_resp.data = out_data;
            cache_resp.dest = dst_pe;
            cache_resp.last = out_end_flag;
            cacheline_resp_stream.write(cache_resp);
        }
        if (left_partitions == 0) {
            break;
        }
    }
}

void write_out(bus_word_t *output,
               hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream) {
LOOP_WRITE_OUT:
    while (true) {
#pragma HLS PIPELINE II = 1

        write_burst_w_dst_pkt_t one_write_burst;

        if (write_burst_stream.read_nb(one_write_burst)) {
            uint32_t dest_addr = one_write_burst.dest;
            bus_word_t data = one_write_burst.data;
            bool end_flag = one_write_burst.last;

            if (end_flag) {
                break;
            }

            output[dest_addr] = data;
        }
    }
}

extern "C" void hbm_writer(
    bus_word_t *src_prop_1,
    bus_word_t *src_prop_2,
    bus_word_t *output,
    uint32_t num_partitions_little,
    uint32_t num_partitions_big,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_1,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_1,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_1,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_1,
hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream) {
#pragma HLS INTERFACE m_axi port = src_prop_1 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = src_prop_2 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = src_prop_1 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_2 bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
 #pragma HLS INTERFACE s_axilite port = num_partitions_little bundle = control
#pragma HLS INTERFACE s_axilite port = num_partitions_big bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW
    little_node_prop_loader(0, src_prop_1, num_partitions_little,ppb_req_stream_1, ppb_resp_stream_1);
    big_node_prop_loader(0, src_prop_2, num_partitions_big,cacheline_req_stream_1, cacheline_resp_stream_1);
    write_out(output, write_burst_stream);
}

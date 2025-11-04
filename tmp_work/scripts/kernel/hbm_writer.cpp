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
            // printf("Waiting for cacheline request...\n");fflush(NULL);
            // printf("Received cacheline request for idx %d from PE %d\n",
            // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
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
            // printf("Sent cacheline response for idx %d to PE %d\n",
            // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
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
    bus_word_t *src_prop_1, bus_word_t *src_prop_2, bus_word_t *src_prop_3,
    bus_word_t *src_prop_4, bus_word_t *src_prop_5, bus_word_t *src_prop_6,
    bus_word_t *src_prop_7, bus_word_t *src_prop_8, bus_word_t *src_prop_9,
    bus_word_t *src_prop_10, bus_word_t *src_prop_11, bus_word_t *src_prop_12,
    bus_word_t *src_prop_13, bus_word_t *src_prop_14, bus_word_t *output,
    uint32_t num_partitions_little, uint32_t num_partitions_big,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_1,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_1,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_2,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_2,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_3,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_3,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_4,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_4,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_5,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_5,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_6,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_6,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_7,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_7,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_8,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_8,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_9,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_9,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_10,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_10,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_11,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_11,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_1,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_1,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_2,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_2,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_3,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_3,
    hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream) {
#pragma HLS INTERFACE m_axi port = src_prop_1 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = src_prop_2 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = src_prop_3 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = src_prop_4 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = src_prop_5 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = src_prop_6 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = src_prop_7 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = src_prop_8 offset = slave bundle = gmem7
#pragma HLS INTERFACE m_axi port = src_prop_9 offset = slave bundle = gmem8
#pragma HLS INTERFACE m_axi port = src_prop_10 offset = slave bundle = gmem9
#pragma HLS INTERFACE m_axi port = src_prop_11 offset = slave bundle = gmem10
#pragma HLS INTERFACE m_axi port = src_prop_12 offset = slave bundle = gmem11
#pragma HLS INTERFACE m_axi port = src_prop_13 offset = slave bundle = gmem12
#pragma HLS INTERFACE m_axi port = src_prop_14 offset = slave bundle = gmem13
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = src_prop_1 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_2 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_3 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_4 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_5 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_6 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_7 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_8 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_9 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_10 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_11 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_12 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_13 bundle = control
#pragma HLS INTERFACE s_axilite port = src_prop_14 bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = num_partitions_little bundle = control
#pragma HLS INTERFACE s_axilite port = num_partitions_big bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    little_node_prop_loader(0, src_prop_1, num_partitions_little,
                            ppb_req_stream_1, ppb_resp_stream_1);
    little_node_prop_loader(1, src_prop_2, num_partitions_little,
                            ppb_req_stream_2, ppb_resp_stream_2);
    little_node_prop_loader(2, src_prop_3, num_partitions_little,
                            ppb_req_stream_3, ppb_resp_stream_3);
    little_node_prop_loader(3, src_prop_4, num_partitions_little,
                            ppb_req_stream_4, ppb_resp_stream_4);
    little_node_prop_loader(4, src_prop_5, num_partitions_little,
                            ppb_req_stream_5, ppb_resp_stream_5);
    little_node_prop_loader(5, src_prop_6, num_partitions_little,
                            ppb_req_stream_6, ppb_resp_stream_6);
    little_node_prop_loader(6, src_prop_7, num_partitions_little,
                            ppb_req_stream_7, ppb_resp_stream_7);
    little_node_prop_loader(7, src_prop_8, num_partitions_little,
                            ppb_req_stream_8, ppb_resp_stream_8);
    little_node_prop_loader(8, src_prop_9, num_partitions_little,
                            ppb_req_stream_9, ppb_resp_stream_9);
    little_node_prop_loader(9, src_prop_10, num_partitions_little,
                            ppb_req_stream_10, ppb_resp_stream_10);
    little_node_prop_loader(10, src_prop_11, num_partitions_little,
                            ppb_req_stream_11, ppb_resp_stream_11);
    big_node_prop_loader(0, src_prop_12, num_partitions_big,
                         cacheline_req_stream_1, cacheline_resp_stream_1);
    big_node_prop_loader(1, src_prop_13, num_partitions_big,
                         cacheline_req_stream_2, cacheline_resp_stream_2);
    big_node_prop_loader(2, src_prop_14, num_partitions_big,
                         cacheline_req_stream_3, cacheline_resp_stream_3);

    write_out(output, write_burst_stream);
}
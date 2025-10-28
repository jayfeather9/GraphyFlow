#include "shared_kernel_params.h"

static void little_node_prop_loader(
    int i, const bus_word_t *node_distances_ddr, uint32_t dst_num,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream) {
#pragma HLS function_instantiate variable = i

    ppb_request_pkt_t one_ppb_request_pkg;
    ppb_response_pkt_t one_ppb_response_pkg;

littleKernelReadMemory:
    while (true) {
#pragma HLS PIPELINE
        if (ppb_req_stream.read_nb(one_ppb_request_pkg)) {
            uint32_t request_round = one_ppb_request_pkg.data;
            bool end_flag = one_ppb_request_pkg.last;

            uint32_t base_addr =
                request_round << LOG_SRC_BUFFER_SIZE >> LOG_DIST_PER_WORD;

            if (end_flag) {
                one_ppb_response_pkg.last = end_flag;
                ppb_resp_stream.write(one_ppb_response_pkg);
                break;
            } else {
                for (int i = 0; i < (SRC_BUFFER_SIZE >> LOG_DIST_PER_WORD);
                     i++) {
                    int addr = base_addr + i;

                    one_ppb_response_pkg.data = node_distances_ddr[addr];
                    one_ppb_response_pkg.dest = addr;
                    one_ppb_response_pkg.last = false;
                    ppb_resp_stream.write(one_ppb_response_pkg);
                }
            }
        }
    }

LOOP_LOADER_2:
    cacheline_data_pkt_t cache_data;
    uint32_t total_cachelines = (dst_num + DIST_PER_WORD - 1) /
                                DIST_PER_WORD; // Total number of cache lines
    for (uint32_t i = 0; i < total_cachelines; i++) {
#pragma HLS PIPELINE II = 1
        cache_data.data = node_distances_ddr[i];
        cache_data.last = (i == total_cachelines - 1) ? true : false;
        cacheline_data_stream.write(cache_data);
    }
}

static void big_node_prop_loader(
    int i, const bus_word_t *node_distances_ddr, uint32_t dst_num,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream) {
#pragma HLS function_instantiate variable = i

    cacheline_request_pkt_t cache_req;
    cacheline_response_pkt_t cache_resp;

    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx = -1;
    bus_word_t last_cacheline;
    bool end_flag_get = false;

    // Stream 0
LOOP_NPL_S0_READ:
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
                end_flag_get = true;
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
        if (end_flag_get) {
            break;
        }
    }

LOOP_LOADER_2:
    cacheline_data_pkt_t cache_data;
    uint32_t total_cachelines = (dst_num + DIST_PER_WORD - 1) /
                                DIST_PER_WORD; // Total number of cache lines
    for (uint32_t i = 0; i < total_cachelines; i++) {
#pragma HLS PIPELINE II = 1
        cache_data.data = node_distances_ddr[i];
        cache_data.last = (i == total_cachelines - 1) ? true : false;
        cacheline_data_stream.write(cache_data);
    }
}

void write_out(int i, bus_word_t *output, uint32_t dst_num,
               hls::stream<write_burst_pkt_t> &write_burst_stream) {
    uint32_t write_idx = 0;
    uint32_t target_writes = ((dst_num + DIST_PER_WORD - 1) / DIST_PER_WORD) -
                             1; // Total number of write bursts
#pragma HLS function_instantiate variable = i
write_out:
    while (true) {
#pragma HLS PIPELINE II = 1

        write_burst_pkt_t one_write_burst;

        if (write_burst_stream.read_nb(one_write_burst)) {
            output[write_idx] = one_write_burst.data;

            if (write_idx >= target_writes) {
                break;
            }
            write_idx = write_idx + 1;
        }
    }
}

extern "C" void
hbm_writer(bus_word_t *node_props_1, bus_word_t *node_props_2,
           bus_word_t *node_props_3, bus_word_t *node_props_4,
           bus_word_t *node_props_5, bus_word_t *output_1, bus_word_t *output_2,
           bus_word_t *output_3, bus_word_t *output_4, bus_word_t *output_5,
           uint32_t dst_num_1, uint32_t dst_num_2, uint32_t dst_num_3,
           uint32_t dst_num_4, uint32_t dst_num_5,
           hls::stream<ppb_request_pkt_t> &ppb_req_stream_1,
           hls::stream<ppb_request_pkt_t> &ppb_req_stream_2,
           hls::stream<ppb_request_pkt_t> &ppb_req_stream_3,
           hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_4,
           hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_5,
           hls::stream<ppb_response_pkt_t> &ppb_resp_stream_1,
           hls::stream<ppb_response_pkt_t> &ppb_resp_stream_2,
           hls::stream<ppb_response_pkt_t> &ppb_resp_stream_3,
           hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_4,
           hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_5,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_1,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_2,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_3,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_4,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_5,
           hls::stream<write_burst_pkt_t> &write_burst_stream_1,
           hls::stream<write_burst_pkt_t> &write_burst_stream_2,
           hls::stream<write_burst_pkt_t> &write_burst_stream_3,
           hls::stream<write_burst_pkt_t> &write_burst_stream_4,
           hls::stream<write_burst_pkt_t> &write_burst_stream_5) {
    // --- Interface Pragmas ---
    // These pragmas map the pointers to separate AXI memory interfaces (gmem1,
    // gmem2, gmem3), allowing for parallel access to different HBM banks.
#pragma HLS INTERFACE m_axi port = node_props_1 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = output_1 offset = slave bundle = gmem1

#pragma HLS INTERFACE m_axi port = node_props_2 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = output_2 offset = slave bundle = gmem2

#pragma HLS INTERFACE m_axi port = node_props_3 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = output_3 offset = slave bundle = gmem3

#pragma HLS INTERFACE m_axi port = node_props_4 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = output_4 offset = slave bundle = gmem4

#pragma HLS INTERFACE m_axi port = node_props_5 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = output_5 offset = slave bundle = gmem5

    // All scalar arguments and pointer addresses are mapped to a single control
    // bus.
#pragma HLS INTERFACE s_axilite port = node_props_1 bundle = control
#pragma HLS INTERFACE s_axilite port = node_props_2 bundle = control
#pragma HLS INTERFACE s_axilite port = node_props_3 bundle = control
#pragma HLS INTERFACE s_axilite port = node_props_4 bundle = control
#pragma HLS INTERFACE s_axilite port = node_props_5 bundle = control
#pragma HLS INTERFACE s_axilite port = output_1 bundle = control
#pragma HLS INTERFACE s_axilite port = output_2 bundle = control
#pragma HLS INTERFACE s_axilite port = output_3 bundle = control
#pragma HLS INTERFACE s_axilite port = output_4 bundle = control
#pragma HLS INTERFACE s_axilite port = output_5 bundle = control
#pragma HLS INTERFACE s_axilite port = dst_num_1 bundle = control
#pragma HLS INTERFACE s_axilite port = dst_num_2 bundle = control
#pragma HLS INTERFACE s_axilite port = dst_num_3 bundle = control
#pragma HLS INTERFACE s_axilite port = dst_num_4 bundle = control
#pragma HLS INTERFACE s_axilite port = dst_num_5 bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // --- Dataflow Pragma ---
    // This pragma enables task-level parallelism, allowing the function calls
    // below to execute concurrently as soon as their input data is available.
#pragma HLS DATAFLOW

    // --- Function Instantiations ---
    // Instantiate the processing logic for each of the three parallel channels.
    // The first argument (0, 1, 2) is a constant integer used by HLS to create
    // three distinct hardware instances of each function.

    little_node_prop_loader(0, node_props_1, dst_num_1, ppb_req_stream_1,
                            ppb_resp_stream_1, cacheline_data_stream_1);
    little_node_prop_loader(1, node_props_2, dst_num_2, ppb_req_stream_2,
                            ppb_resp_stream_2, cacheline_data_stream_2);
    little_node_prop_loader(2, node_props_3, dst_num_3, ppb_req_stream_3,
                            ppb_resp_stream_3, cacheline_data_stream_3);
    big_node_prop_loader(3, node_props_4, dst_num_4, cacheline_req_stream_4,
                         cacheline_resp_stream_4, cacheline_data_stream_4);
    big_node_prop_loader(4, node_props_5, dst_num_5, cacheline_req_stream_5,
                         cacheline_resp_stream_5, cacheline_data_stream_5);

    write_out(0, output_1, dst_num_1, write_burst_stream_1);
    write_out(1, output_2, dst_num_2, write_burst_stream_2);
    write_out(2, output_3, dst_num_3, write_burst_stream_3);
    write_out(3, output_4, dst_num_4, write_burst_stream_4);
    write_out(4, output_5, dst_num_5, write_burst_stream_5);
}
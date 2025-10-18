#include "graphyflow_little.h"
// #include <stdio.h>

static void hbm_memory_reader_logic(
    const bus_word_t *node_distances, // HBM 指针
    int32_t num_nodes,

    hls::stream<l_ppb_request_pkt> &l_ppb_request_stm,
    hls::stream<l_ppb_response_pkt> &l_ppb_response_stm,
    hls::stream<b_node_distance_burst_t> &node_dist_stream) {

    l_ppb_request_pkt one_ppb_request_pkg;
    l_ppb_response_pkt one_ppb_response_pkg;

    unsigned int processed_nodes = 0;
    ppb_response_dt one_ppb_response;

    // printf("DEBUG HBM wrapper start num_nodes:%d\n",num_nodes);fflush(NULL);

littleKernelReadMemory:
    while (true) {
#pragma HLS PIPELINE // II=1

        if (l_ppb_request_stm.read_nb(one_ppb_request_pkg)) {

            ppb_request_dt one_ppb_request;

            one_ppb_request.request_round = one_ppb_request_pkg.data;
            one_ppb_request.end_flag = one_ppb_request_pkg.last;

            ap_uint<32> base_addr =
                one_ppb_request.request_round << LOG2_SRC_BUFFER_SIZE >> 4;
            // printf("DEBUG receive request %d\n");fflush(NULL);
            if (one_ppb_request.end_flag) {

                one_ppb_response.end_flag = one_ppb_request.end_flag;
                one_ppb_response_pkg.last = one_ppb_response.end_flag;
                // write_to_stream(l_ppb_response_stm, one_ppb_response_pkg);
                l_ppb_response_stm.write(one_ppb_response_pkg);

                processed_nodes += 256;
                //// printf("DEBUG read %d
                /// nodes\n",&processed_nodes);fflush(NULL);
                // DEBUG_// printf("processed_num_partitions %d, %d !\n",
                // processed_num_partitions, num_paritions);
            } else {

                for (int i = 0; i < (SRC_BUFFER_SIZE >> 4); i++) {
                    int addr = base_addr + i;
                    one_ppb_response.addr = addr;
                    one_ppb_response.data = node_distances[addr];
                    one_ppb_response.end_flag = one_ppb_request.end_flag;

                    one_ppb_response_pkg.data = one_ppb_response.data;
                    one_ppb_response_pkg.dest = one_ppb_response.addr;
                    one_ppb_response_pkg.last = one_ppb_response.end_flag;
                    l_ppb_response_stm.write(one_ppb_response_pkg);
                    // printf("DEBUG write done %d\n",i);fflush(NULL);
                }
            }
        }
        if (processed_nodes >= num_nodes) {
            // DEBUG_// printf("HBMwrapperReadMemory exists %d!\n",
            // processed_num_partitions);
            //// printf("DEBUG HBM read done\n");fflush(NULL);
            break;
        }
    }

    const int num_dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
    const int num_wide_reads =
        (num_nodes + num_dists_per_word - 1) / num_dists_per_word;

    // Stream 1
    int nodes_read_s1 = 0;
    int burst_idx1 = 0, burst_idx2 = 0;
    // // printf("Loading node distances for %d nodes (%d wide reads)\n",
    // num_nodes, num_wide_reads); fflush(NULL);
LOOP_NPL_S1_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 2
        bus_word_t wide_word = node_distances[i];
        node_distance_burst_t burst;

        b_node_distance_burst_t b_burst;
        b_burst.last = false;
        b_burst.dest = 0;
        b_burst.data = wide_word.range(255, 0);
        node_dist_stream.write(b_burst);

        b_burst.data = wide_word.range(511, 256);
        node_dist_stream.write(b_burst);
    }
    // write last packet
    b_node_distance_burst_t b_burst;
    b_burst.last = true;
    b_burst.dest = 0;
    b_burst.data = 0;
    node_dist_stream.write(b_burst);
}

// --- HBM Manager 内核顶层函数 ---
extern "C" void
hbm_manager(const bus_word_t *node_distances, int32_t num_nodes,
            hls::stream<l_ppb_request_pkt> &l_ppb_request_stm,
            hls::stream<l_ppb_response_pkt> &l_ppb_response_stm,
            hls::stream<b_node_distance_burst_t> &node_dist_stream // 保留或移除
) {

#pragma HLS INTERFACE m_axi port = node_distances offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = node_distances
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = return

#pragma HLS DATAFLOW

    hbm_memory_reader_logic(node_distances, num_nodes, l_ppb_request_stm,
                            l_ppb_response_stm, node_dist_stream);
}

// static void
// in_hbm_manager(const bus_word_t *node_distances, int32_t num_nodes,
//                hls::stream<b_cacheline_req_t> &cacheline_req_stream,
//                hls::stream<b_cacheline_resp_t> &cacheline_resp_stream,
//                hls::stream<b_node_distance_burst_t> &node_dist_stream) {
//     ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx = -1;
//     bus_word_t last_cacheline;
//     cacheline_resp_t cache_resp;
//     bool end_flag_get = false;

//     // // printf("Starting hbm_manager.\n"); fflush(NULL);

//     // Stream 0
// LOOP_NPL_S0_READ:
//     while (true) {
// #pragma HLS PIPELINE
//         if (!cacheline_req_stream.empty()) {
//             // // printf("Waiting for cacheline request...\n");fflush(NULL);
//             cacheline_req_t cache_req;
//             b_cacheline_req_t b_cache_req = cacheline_req_stream.read();
//             cache_req.idx = b_cache_req.data;
//             cache_req.dst = b_cache_req.dest;
//             cache_req.end_flag = b_cache_req.last;
//             // // printf("Received cacheline request for idx %d from PE
//             %d\n",
//             // (int)cache_req.idx, (int)cache_req.dst); fflush(NULL);
//             if (cache_req.end_flag) {
//                 cache_resp.end_flag = true;
//                 end_flag_get = true;
//             } else {
//                 cache_resp.end_flag = false;
//                 if (cache_req.idx == last_cache_idx) {
//                     cache_resp.data = last_cacheline;
//                 } else {
//                     cache_resp.data = node_distances[cache_req.idx];
//                 }
//             }

//             last_cacheline = cache_resp.data;
//             last_cache_idx = cache_req.idx;
//             cache_resp.dst = cache_req.dst;

//             b_cacheline_resp_t b_cache_resp;
//             b_cache_resp.data = cache_resp.data;
//             b_cache_resp.dest = cache_resp.dst;
//             b_cache_resp.last = cache_resp.end_flag;
//             cacheline_resp_stream.write(b_cache_resp);
//             // // printf("Sent cacheline response for idx %d to PE %d\n",
//             // (int)cache_req.idx, (int)cache_req.dst); fflush(NULL);
//             if (end_flag_get) {
//                 break;
//             }
//         }
//     }

//     const int num_dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
//     const int num_wide_reads =
//         (num_nodes + num_dists_per_word - 1) / num_dists_per_word;

//     // Stream 1
//     int nodes_read_s1 = 0;
//     int burst_idx1 = 0, burst_idx2 = 0;
//     // // printf("Loading node distances for %d nodes (%d wide reads)\n",
//     // num_nodes, num_wide_reads); fflush(NULL);
// LOOP_NPL_S1_READ:
//     for (int i = 0; i < num_wide_reads; i++) {
// #pragma HLS PIPELINE II = 2
//         bus_word_t wide_word = node_distances[i];
//         node_distance_burst_t burst;

//         b_node_distance_burst_t b_burst;
//         b_burst.last = false;
//         b_burst.dest = 0;
//         b_burst.data = wide_word.range(255, 0);
//         node_dist_stream.write(b_burst);

//         b_burst.data = wide_word.range(511, 256);
//         node_dist_stream.write(b_burst);
//     }
//     // write last packet
//     b_node_distance_burst_t b_burst;
//     b_burst.last = true;
//     b_burst.dest = 0;
//     b_burst.data = 0;
//     node_dist_stream.write(b_burst);
//     // // printf("Finished loading node distances.\n"); fflush(NULL);
// }

// big_hbm_manager(const bus_word_t *node_distances, int32_t num_nodes,
//             hls::stream<b_cacheline_req_t> &cacheline_req_stream,
//             hls::stream<b_cacheline_resp_t> &cacheline_resp_stream,
//             hls::stream<b_node_distance_burst_t> &node_dist_stream) {

// #pragma HLS INTERFACE m_axi port = node_distances offset = slave bundle =
// gmem3 #pragma HLS INTERFACE s_axilite port = node_distances #pragma HLS
// INTERFACE s_axilite port = num_nodes #pragma HLS INTERFACE s_axilite port =
// return

// #pragma HLS DATAFLOW

//     in_hbm_manager(node_distances, num_nodes, cacheline_req_stream,
//                    cacheline_resp_stream, node_dist_stream);
// }

#include "graphyflow_big.h"

extern "C" void
hbm_manager(const bus_word_t *node_distances, int32_t num_nodes,
            hls::stream<b_cacheline_req_t> &cacheline_req_stream,
            hls::stream<b_cacheline_resp_t> &cacheline_resp_stream,
            hls::stream<b_node_distance_burst_t> &node_dist_stream) {

#pragma HLS INTERFACE m_axi port = node_distances offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = node_distances
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = return

    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx = -1;
    bus_word_t last_cacheline;
    cacheline_resp_t cache_resp;
    bool end_flag_get = false;

    // Stream 0
LOOP_NPL_S0_READ:
    while (true) {
#pragma HLS PIPELINE II = 1
        // printf("Waiting for cacheline request...\n");fflush(NULL);
        cacheline_req_t cache_req;
        b_cacheline_req_t b_cache_req = cacheline_req_stream.read();
        cache_req.idx = b_cache_req.data;
        cache_req.dst = b_cache_req.dest;
        cache_req.end_flag = b_cache_req.last;
        // printf("Received cacheline request for idx %d from PE %d\n",
        // (int)cache_req.idx, (int)cache_req.dst); fflush(NULL);
        if (cache_req.end_flag) {
            cache_resp.end_flag = true;
            end_flag_get = true;
        } else {
            cache_resp.end_flag = false;
            if (cache_req.idx == last_cache_idx) {
                cache_resp.data = last_cacheline;
            } else {
                cache_resp.data = node_distances[cache_req.idx];
            }
        }

        last_cacheline = cache_resp.data;
        last_cache_idx = cache_req.idx;
        cache_resp.dst = cache_req.dst;

        b_cacheline_resp_t b_cache_resp;
        b_cache_resp.data = cache_resp.data;
        b_cache_resp.dest = cache_resp.dst;
        b_cache_resp.last = cache_resp.end_flag;
        cacheline_resp_stream.write(b_cache_resp);
        // printf("Sent cacheline response for idx %d to PE %d\n",
        // (int)cache_req.idx, (int)cache_req.dst); fflush(NULL);
        if (end_flag_get) {
            break;
        }
    }

    const int num_dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
    const int num_wide_reads =
        (num_nodes + num_dists_per_word - 1) / num_dists_per_word;

    // Stream 1
    int nodes_read_s1 = 0;
    int burst_idx1 = 0, burst_idx2 = 0;
    // printf("Loading node distances for %d nodes (%d wide reads)\n",
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
}
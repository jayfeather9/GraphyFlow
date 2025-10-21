#include "graphyflow_big.h"

static void node_property_loader(
    const bus_word_t *node_distances_ddr,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream) {

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
}

void write_out(bus_word_t *output,
               uint32_t dst_num,
               hls::stream<write_burst_pkt_t> &write_burst_stream) {
    uint32_t write_idx = 0;

    uint32_t total_writes = 0;
    uint32_t target_writes =
        (dst_num + DBL_PE_NUM - 1) / DBL_PE_NUM; // Total number of write bursts
write_out:
    while (true) {
#pragma HLS PIPELINE II = 1

        write_burst_pkt_t one_write_burst;

        if (write_burst_stream.read_nb(one_write_burst)) {

            write_idx = one_write_burst.dest;

            bus_word_t new_prop = one_write_burst.data;
            output[write_idx] = new_prop;

            total_writes++;
            if (total_writes >= target_writes) {
                break;
            }
        }
    }
}

extern "C" void
hbm_writer(bus_word_t *node_props, bus_word_t *output, uint32_t dst_num,
           hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
           hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
           hls::stream<write_burst_pkt_t> &write_burst_stream) {
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = node_props bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW
    node_property_loader(node_props, cacheline_req_stream,
                         cacheline_resp_stream);
    write_out(output, dst_num, write_burst_stream);
}
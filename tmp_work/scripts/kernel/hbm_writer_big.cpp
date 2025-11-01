#include "shared_kernel_params.h"

static void big_node_prop_loader(
    int i, const bus_word_t *node_distances_ddr, uint32_t dst_num,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
    hls::stream<bus_word_t> &cacheline_data_stream) {
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
    bus_word_t cache_data;
    uint32_t total_cachelines = (dst_num + DIST_PER_WORD - 1) /
                                DIST_PER_WORD; // Total number of cache lines
    for (uint32_t i = 0; i < total_cachelines; i++) {
#pragma HLS PIPELINE II = 1
        cache_data = node_distances_ddr[i];
        cacheline_data_stream.write(cache_data);
    }
}

static void apply_updates(int i, uint32_t dst_num,
                          hls::stream<bus_word_t> &cacheline_data_stream,
                          hls::stream<write_burst_pkt_t> &kernel_out_stream,
                          hls::stream<bus_word_t> &write_burst_stream) {
#pragma HLS function_instantiate variable = i

    uint32_t read_idx = 0;
    uint32_t addr = 0;
    bool pkt_ready = false;
    bool data_ready = false;
    write_burst_pkt_t pkt;
    bus_word_t input_node_prop;

LOOP_APPLY:
    while (true) {
#pragma HLS PIPELINE II = 1
        if (!pkt_ready) {
            pkt_ready = kernel_out_stream.read_nb(pkt);
        }

        if (!data_ready) {
            data_ready = cacheline_data_stream.read_nb(input_node_prop);
        }

        if (pkt_ready && data_ready) {
            pkt_ready = false;
            data_ready = false;
            bus_word_t wide_word = pkt.data;
            bus_word_t node_prop = input_node_prop;
            bus_word_t new_node_prop = 0;

        LOOP_REDUCE_MIN:
            for (int idx = 0; idx < DBL_PE_NUM; idx++) {
#pragma HLS UNROLL
                ap_fixed_pod_t update_dist =
                    wide_word.range(31 + (idx << 5), (idx << 5));
                ap_fixed_pod_t current_dist =
                    node_prop.range(31 + (idx << 5), (idx << 5));
                ap_fixed_pod_t new_dist =
                    (update_dist < current_dist) ? update_dist : current_dist;
                new_node_prop.range(31 + (idx << 5), (idx << 5)) = new_dist;
            }

            write_burst_stream.write(new_node_prop);
            read_idx++;
            addr += (PE_NUM << 1);
            if (addr >= dst_num) {
                break;
            }
        }
    }
}

void write_out(int i, bus_word_t *output, uint32_t dst_num,
               hls::stream<bus_word_t> &write_burst_stream) {
    uint32_t write_idx = 0;
    uint32_t target_writes = ((dst_num + DBL_PE_NUM - 1) / DBL_PE_NUM) -
                             1; // Total number of write bursts
#pragma HLS function_instantiate variable = i
write_out:
    while (true) {
#pragma HLS PIPELINE II = 1 style = frp

        bus_word_t one_write_burst;

        if (write_burst_stream.read_nb(one_write_burst)) {
            output[write_idx] = one_write_burst;

            if (write_idx >= target_writes) {
                break;
            }
            write_idx = write_idx + 1;
        }
    }
}

extern "C" void
hbm_writer_big(bus_word_t *node_props, bus_word_t *output, uint32_t dst_num,
               hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
               hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
               hls::stream<write_burst_pkt_t> &kernel_out_stream) {
    // --- Interface Pragmas ---
    // These pragmas map the pointers to separate AXI memory interfaces (gmem1,
    // gmem2, gmem3), allowing for parallel access to different HBM banks.
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1

    // All scalar arguments and pointer addresses are mapped to a single control
    // bus.
#pragma HLS INTERFACE s_axilite port = node_props bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = dst_num bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // --- Dataflow Pragma ---
    // This pragma enables task-level parallelism, allowing the function calls
    // below to execute concurrently as soon as their input data is available.
#pragma HLS DATAFLOW

    // --- Function Instantiations ---

    hls::stream<bus_word_t> write_burst_stream("write_burst_stream");
#pragma HLS STREAM variable = write_burst_stream depth = 4

    hls::stream<bus_word_t> cacheline_data_stream("cacheline_data_stream");
#pragma HLS STREAM variable = cacheline_data_stream depth = 16

    big_node_prop_loader(0, node_props, dst_num, cacheline_req_stream,
                         cacheline_resp_stream, cacheline_data_stream);

    apply_updates(0, dst_num, cacheline_data_stream, kernel_out_stream,
                  write_burst_stream);

    write_out(0, output, dst_num, write_burst_stream);
}
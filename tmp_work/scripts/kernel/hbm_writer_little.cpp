#include "shared_kernel_params.h"

static void
little_node_prop_loader(int i, const bus_word_t *node_distances_ddr,
                        uint32_t dst_num,
                        hls::stream<ppb_request_pkt_t> &ppb_req_stream,
                        hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
                        hls::stream<reduce_word_t> &cacheline_data_stream) {
#pragma HLS function_instantiate variable = i

    ppb_request_pkt_t one_ppb_request_pkg;
    ppb_response_pkt_t one_ppb_response_pkg;

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
                break;
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

LOOP_LOADER_2:
    bus_word_t cache_data;
    uint32_t total_cachelines = (dst_num + DIST_PER_WORD - 1) /
                                DIST_PER_WORD; // Total number of cache lines
    for (uint32_t i = 0; i < total_cachelines; i++) {
#pragma HLS PIPELINE
        cache_data = node_distances_ddr[i];
        reduce_word_t reduce_data;
        reduce_data = cache_data.range(63, 0);
        cacheline_data_stream.write(reduce_data);
        reduce_data = cache_data.range(127, 64);
        cacheline_data_stream.write(reduce_data);
        reduce_data = cache_data.range(191, 128);
        cacheline_data_stream.write(reduce_data);
        reduce_data = cache_data.range(255, 192);
        cacheline_data_stream.write(reduce_data);
        reduce_data = cache_data.range(319, 256);
        cacheline_data_stream.write(reduce_data);
        reduce_data = cache_data.range(383, 320);
        cacheline_data_stream.write(reduce_data);
        reduce_data = cache_data.range(447, 384);
        cacheline_data_stream.write(reduce_data);
        reduce_data = cache_data.range(511, 448);
        cacheline_data_stream.write(reduce_data);
    }
}

static void apply_updates(int i, uint32_t dst_num,
                          hls::stream<reduce_word_t> &cacheline_data_stream,
                          hls::stream<little_out_pkt_t> &kernel_out_stream,
                          hls::stream<reduce_word_t> &write_burst_stream) {
#pragma HLS function_instantiate variable = i

    uint32_t read_idx = 0;
    uint32_t addr = 0;
    bool pkt_ready = false;
    bool data_ready = false;
    little_out_pkt_t pkt;
    reduce_word_t input_node_prop;

    // round dst_num to be multiple of DISTANCES_PER_REDUCE_WORD * 8
    const uint32_t rounded_dst_num =
        ((dst_num + (8 * DISTANCES_PER_REDUCE_WORD) - 1) /
         (8 * DISTANCES_PER_REDUCE_WORD)) *
        (8 * DISTANCES_PER_REDUCE_WORD);

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
            reduce_word_t wide_word = pkt.data;
            reduce_word_t node_prop = input_node_prop;
            reduce_word_t new_node_prop = 0;

        LOOP_REDUCE_MIN:
            for (int idx = 0; idx < DISTANCES_PER_REDUCE_WORD; idx++) {
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
            addr += DISTANCES_PER_REDUCE_WORD;
            if (addr >= rounded_dst_num) {
                break;
            }
        }
    }
}

void write_out(int i, bus_word_t *output, uint32_t dst_num,
               hls::stream<reduce_word_t> &reduce_word_stream) {
    uint32_t write_idx = 0;
    uint32_t target_writes = ((dst_num + DBL_PE_NUM - 1) / DBL_PE_NUM) -
                             1; // Total number of write bursts
#pragma HLS function_instantiate variable = i
    bus_word_t write_word;
    uint32_t cnt = 0;
write_out:
    while (true) {
#pragma HLS PIPELINE II = 1 style = frp

        reduce_word_t one_word;

        if (reduce_word_stream.read_nb(one_word)) {
            switch (cnt) {
            case 0: write_word.range(63, 0) = one_word; break;
            case 1: write_word.range(127, 64) = one_word; break;
            case 2: write_word.range(191, 128) = one_word; break;
            case 3: write_word.range(255, 192) = one_word; break;
            case 4: write_word.range(319, 256) = one_word; break;
            case 5: write_word.range(383, 320) = one_word; break;
            case 6: write_word.range(447, 384) = one_word; break;
            case 7: write_word.range(511, 448) = one_word; break;
            default: break;
            }
            cnt++;
            if (cnt == 8) {
                cnt = 0;
                output[write_idx] = write_word;

                if (write_idx >= target_writes) {
                    break;
                }
                write_idx = write_idx + 1;
            }
        }
    }
}

extern "C" void
hbm_writer_little(bus_word_t *node_props, bus_word_t *output, uint32_t dst_num,
                  hls::stream<ppb_request_pkt_t> &ppb_req_stream,
                  hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
                  hls::stream<little_out_pkt_t> &kernel_out_stream) {
    // --- Interface Pragmas ---
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem0

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
    // Instantiate the processing logic for each of the three parallel channels.
    // The first argument (0, 1, 2) is a constant integer used by HLS to create
    // three distinct hardware instances of each function.

    hls::stream<reduce_word_t> write_burst_stream("write_burst_stream");
#pragma HLS STREAM variable = write_burst_stream depth = 4

    hls::stream<reduce_word_t> cacheline_data_stream("cacheline_data_stream");
#pragma HLS STREAM variable = cacheline_data_stream depth = 16

    little_node_prop_loader(0, node_props, dst_num, ppb_req_stream,
                            ppb_resp_stream, cacheline_data_stream);

    apply_updates(0, dst_num, cacheline_data_stream, kernel_out_stream,
                  write_burst_stream);

    write_out(0, output, dst_num, write_burst_stream);
}
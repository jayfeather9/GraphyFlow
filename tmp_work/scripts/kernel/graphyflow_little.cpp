#include "graphyflow_little.h"

template <typename T1, typename T2>
void stream2axistream(hls::stream<T1> &stream, hls::stream<T2> &axi_stream) {

stream2axistream:
    while (true) {

        T1 tmp_t1 = stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.request_round;
        tmp_t2.last = tmp_t1.end_flag;
        // write_to_stream(axi_stream, tmp_t2);
        axi_stream.write(tmp_t2);
        if (tmp_t1.end_flag)
            break;
    }
}

template <typename T1, typename T2>
void axistream2stream(hls::stream<T1> &axi_stream, hls::stream<T2> &stream) {

axistream2stream:
    while (true) {

        T1 tmp_t1 = axi_stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.data;
        tmp_t2.addr = tmp_t1.dest;
        tmp_t2.end_flag = tmp_t1.last;
        // write_to_stream(stream, tmp_t2);
        stream.write(tmp_t2);
        if (tmp_t2.end_flag)
            break;
    }
}

void request_manager(hls::stream<edge_descriptor_batch_t> &edge_burst_stm,
                     hls::stream<ppb_request_t> &ppb_request_stm,
                     hls::stream<ppb_response_t> &ppb_response_stm,
                     hls::stream<update_tuple_t> &update_set_stm,
                     uint32_t memory_offset, uint32_t part_edge_num) {
    // as we can buffer two vertices in one row with width of 64-bit, we can let
    // the depth go as MAX_VERTICES_IN_ONE_PARTITION / 2.
    bus_word_t src_prop_buffer[PE_NUM][2][SRC_BUFFER_SIZE >> 4];
#pragma HLS ARRAY_PARTITION variable = src_prop_buffer dim = 1 complete
#pragma HLS BIND_STORAGE variable = src_prop_buffer type = RAM_S2P impl = BRAM
#pragma HLS dependence variable = src_prop_buffer inter false

    int32_t pp_read_idx = 0;
    int32_t pp_write_idx = 0;

    int32_t pp_reponse_idx = 0;

    int32_t pp_read_round = 0;
    int32_t pp_write_round = 0;

    int32_t pp_request_round = 0;

    int32_t edge_set_cnt = 0;
    const int32_t total_edge_sets = part_edge_num >> LOG_PE_NUM;

    bool wait_flag = 0;

    edge_descriptor_batch_t an_edge_burst;

    ppb_request_t one_ppb_request;

    ppb_response_t one_ppb_response;

    distance_t real_edge_weight =
        1.0; // All edge weights are 1.0 in unweighted graph
    const ap_fixed_pod_t edge_weight =
        (*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight));

scatterLoop:
    while (true) {
#pragma HLS PIPELINE II = 1
        // logic to fill the ping-pong buffer.
        if ((pp_request_round - pp_read_round) <= 1) {
            if (pp_request_round < pp_read_round)
                pp_request_round = pp_read_round;
            one_ppb_request.request_round = pp_request_round + memory_offset;
            one_ppb_request.end_flag = 0;
            ppb_request_stm.write(one_ppb_request);
            pp_request_round++;
        }

        if (ppb_response_stm.read_nb(one_ppb_response)) {
            pp_write_round =
                (one_ppb_response.addr << 4 >> LOG_SRC_BUFFER_SIZE) -
                memory_offset;

            bool write_buffer = pp_write_round & 0x1;

            int32_t write_idx =
                one_ppb_response.addr & ((SRC_BUFFER_SIZE >> 4) - 1);

            bus_word_t one_read_burst =
                one_ppb_response
                    .data; // src_prop[(base_addr >> 4) + pp_write_idx];

            for (int u = 0; u < PE_NUM; u++) {
#pragma HLS UNROLL
                src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;
            }
        }

        // logic to read the ping-pong buffer and synchronization.
        if (!wait_flag)
            an_edge_burst = edge_burst_stm.read();

        pp_read_round = (an_edge_burst.edges[0].src_id / SRC_BUFFER_SIZE);

        wait_flag = (pp_read_round >= pp_write_round) ? 1 : 0;

        bool exit_flag = (wait_flag == 0)
                             ? (edge_set_cnt + 1 >= total_edge_sets)
                             : (edge_set_cnt >= total_edge_sets);

        if (!wait_flag) {

            bool read_buffer = pp_read_round & 0x1;

            update_tuple_t an_update_set;

            for (int u = 0; u < PE_NUM; u++) {
#pragma HLS UNROLL
                ap_uint<12> idx = an_edge_burst.edges[u].src_id.range(11, 0);
                ap_uint<8> uram_row_idx = idx.range(11, 4);
                ap_uint<4> uram_row_offset = idx.range(3, 0);

                bus_word_t uram_row =
                    src_prop_buffer[u][read_buffer][uram_row_idx];
                ap_fixed_pod_t src_prop =
                    uram_row.range(31 + ((ap_uint<9>)uram_row_offset << 5),
                                   ((ap_uint<9>)uram_row_offset << 5));
                ap_fixed_pod_t update = src_prop + edge_weight;

                an_update_set.data[u].node_id = an_edge_burst.edges[u].dst_id;
                an_update_set.data[u].prop = update;
            }
            update_set_stm.write(an_update_set);

            edge_set_cnt++;
        }

        if (exit_flag) {
            one_ppb_request.end_flag = 1;
            ppb_request_stm.write(one_ppb_request);
        exitscatter:
            while (true) {
                ppb_response_stm.read(one_ppb_response);
                if (one_ppb_response.end_flag)
                    break;
            }
            break;
        }
    }
}

// Single-PE aggregation function
// Handles initialization and aggregation for one PE
static void
Reduc_105_unit_reduce(hls::stream<update_tuple_t> &update_set_stm,
                      hls::stream<reduce_word_t> (&pe_mem_outs)[PE_NUM],
                      int32_t edge_num, int32_t dst_num) {
    // --- Phase 1: Memory Declaration ---
    const int MEM_SIZE = MAX_NUM / DISTANCES_PER_REDUCE_WORD;
    reduce_word_t prop_mem[PE_NUM][MEM_SIZE];
#pragma HLS ARRAY_PARTITION variable = prop_mem complete dim = 1
#pragma HLS BIND_STORAGE variable = prop_mem type = RAM_S2P impl = URAM
#pragma HLS dependence variable = prop_mem inter false

    // Latency-hiding cache for recently accessed URAM words
    reduce_word_t cache_data_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
    ap_uint<20> cache_addr_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0

    const uint32_t num_words = (dst_num + 1) >> 1;
    const uint32_t rounded_num_words = ((num_words + 7) & ~7);

#ifdef EMULATION
    memset(prop_mem, 0, sizeof(reduce_word_t) * PE_NUM * MEM_SIZE);
#endif

LOOP_INIT_CACHE_ADDR:
    for (int i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            cache_addr_buffer[pe][i] = 0x0; // Invalidate cache
            cache_data_buffer[pe][i] = 0x0;
        }
    }

    const uint32_t total_updates =
        edge_num >> LOG_PE_NUM; // Assuming one update per node
    // --- Phase 3: Aggregation Loop ---
LOOP_AGGREGATE:
    for (int update_idx = 0; update_idx < total_updates; update_idx++) {
#pragma HLS PIPELINE II = 1
        update_tuple_t one_update;
        one_update = update_set_stm.read();

        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            if ((one_update.data[pe].node_id.range(19, 19) ==
                 0)) { // Valid key check
                ap_uint<20> key = one_update.data[pe].node_id;
                uint32_t incoming_dist_pod = one_update.data[pe].prop;

                ap_uint<20> word_addr = (key >> 1);

                reduce_word_t current_word = prop_mem[pe][word_addr];

                // Check cache first
                for (int i = L; i >= 0; --i) {
#pragma HLS UNROLL
                    if (cache_addr_buffer[pe][i] == word_addr) {
                        current_word = cache_data_buffer[pe][i];
                        break;
                    }
                }

                // Shift cache
                for (int i = 0; i < L; i++) {
#pragma HLS UNROLL
                    cache_addr_buffer[pe][i] = cache_addr_buffer[pe][i + 1];
                    cache_data_buffer[pe][i] = cache_data_buffer[pe][i + 1];
                }

                reduce_word_t tmp_cur_word = current_word;

                uint32_t msb = current_word.range(63, 32);
                uint32_t lsb = current_word.range(31, 0);

                uint32_t msb_out = (msb < incoming_dist_pod && msb != 0x0)
                                       ? msb
                                       : incoming_dist_pod;
                uint32_t lsb_out = (lsb < incoming_dist_pod && lsb != 0x0)
                                       ? lsb
                                       : incoming_dist_pod;

                reduce_word_t accumulate_msb;
                reduce_word_t accumulate_lsb;

                accumulate_msb.range(63, 32) = msb_out;
                accumulate_msb.range(31, 0) = tmp_cur_word.range(31, 0);

                accumulate_lsb.range(63, 32) = tmp_cur_word.range(63, 32);
                accumulate_lsb.range(31, 0) = lsb_out;

                if (key & 0x01) {
                    prop_mem[pe][word_addr] = accumulate_msb;
                    cache_data_buffer[pe][L] = accumulate_msb;
                } else {
                    prop_mem[pe][word_addr] = accumulate_lsb;
                    cache_data_buffer[pe][L] = accumulate_lsb;
                }
                cache_addr_buffer[pe][L] = word_addr;
            }
        }
    }

    // --- Phase 4: Stream out aggregated memory ---
LOOP_STREAM_OUT:
    for (int i = 0; i < rounded_num_words; i++) {
#pragma HLS PIPELINE
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            reduce_word_t word = prop_mem[pe][i];
            prop_mem[pe][i] = 0;
            pe_mem_outs[pe].write(word);
        }
    }
}

// Multi-PE drain function
// Collects aggregated data from all PEs and outputs final results
static void
Reduc_105_drain_multi_pe(hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM],
                         hls::stream<little_out_pkt_t> &kernel_out_stream,
                         int32_t dst_num) {

    // --- Phase 2: High-Performance Drain Loop ---
    little_out_pkt_t one_write_burst;
    one_write_burst.last = 0;
    distance_t max_val = (distance_t)(16384.0);
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);

    // round dst_num to be 8 * DISTANCES_PER_REDUCE_WORD
    int32_t rounded_dst_num = ((dst_num + (8 * DISTANCES_PER_REDUCE_WORD) - 1) /
                               (8 * DISTANCES_PER_REDUCE_WORD)) *
                              (8 * DISTANCES_PER_REDUCE_WORD);

LOOP_DRAIN_ADDR:
    for (int32_t base_addr = 0; base_addr < rounded_dst_num;
         base_addr += DISTANCES_PER_REDUCE_WORD) {
#pragma HLS PIPELINE II = 1
        ap_fixed_pod_t uram_res_low = max_pod;
        ap_fixed_pod_t uram_res_high = max_pod;
    LOOP_FOR_57:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            reduce_word_t word = pe_mem_in[pe_idx].read();

            ap_fixed_pod_t incoming_dist_pod_low = word.range(31, 0);
            ap_fixed_pod_t incoming_dist_pod_high = word.range(63, 32);
            uram_res_low = (uram_res_low < incoming_dist_pod_low ||
                            incoming_dist_pod_low == 0x0)
                               ? uram_res_low
                               : incoming_dist_pod_low;
            uram_res_high = (uram_res_high < incoming_dist_pod_high ||
                             incoming_dist_pod_high == 0x0)
                                ? uram_res_high
                                : incoming_dist_pod_high;
        }
        reduce_word_t merged_word;
        merged_word.range(31, 0) = uram_res_low;
        merged_word.range(63, 32) = uram_res_high;
        one_write_burst.data = merged_word;
        kernel_out_stream.write(one_write_burst);
    }
}

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void
graphyflow_little(const bus_word_t *edge_props, int32_t num_nodes,
                  int32_t num_edges, int32_t dst_num, int32_t memory_offset,
                  hls::stream<ppb_request_pkt_t> &ppb_req_stream,
                  hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
                  hls::stream<little_out_pkt_t> &kernel_out_stream) {
#pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = edge_props
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = dst_num
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW

    // Existing streams
    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 8
    hls::stream<update_tuple_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 8
    hls::stream<reduce_word_t> pe_mem_outs[PE_NUM];
#pragma HLS STREAM variable = pe_mem_outs depth = 8

    hls::stream<ppb_request_t> ppb_req_stream_internal;
#pragma HLS STREAM variable = ppb_req_stream_internal depth = 8
    hls::stream<ppb_response_t> ppb_resp_stream_internal;
#pragma HLS STREAM variable = ppb_resp_stream_internal depth = 8

    // --- Data Loading ---
    // --- Data Loading ---
    const int edges_per_word =
        AXI_BUS_WIDTH / (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH);
    const int num_wide_reads = num_edges / edges_per_word;

LOOP_EDL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = edge_props[i];
        edge_descriptor_batch_t edge_batch;
    LOOP_EDL_UNPACK:
        for (int j = 0; j < edges_per_word; j++) {
#pragma HLS UNROLL
            ap_uint<64> packed_edge = wide_word.range(63 + (j << 6), (j << 6));
            edge_batch.edges[j].dst_id = packed_edge.range(19, 0);
            edge_batch.edges[j].src_id = packed_edge.range(63, 32);
        }
        edge_stream.write(edge_batch);
    }

    stream2axistream(ppb_req_stream_internal, ppb_req_stream);
    axistream2stream(ppb_resp_stream, ppb_resp_stream_internal);
    request_manager(edge_stream, ppb_req_stream_internal,
                    ppb_resp_stream_internal, stream_edge_data, memory_offset,
                    num_edges);

    // --- Reduction ---
    Reduc_105_unit_reduce(stream_edge_data, pe_mem_outs, num_edges, dst_num);
    Reduc_105_drain_multi_pe(pe_mem_outs, kernel_out_stream, dst_num);
}

#include "graphyflow_little.h"

static void
edge_descriptor_loader(const bus_word_t *edge_props_ddr,
                       hls::stream<edge_descriptor_batch_t> &edge_stream,
                       int32_t num_edges) {
    const int bits_per_edge = NODE_ID_BITWIDTH + NODE_ID_BITWIDTH;
    const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
    const int num_wide_reads =
        (num_edges + edges_per_word - 1) / edges_per_word;

    int edges_read = 0;
    edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
    edge_batch.end_pos = 0;

    node_id_burst_t src_id_burst;
#pragma HLS ARRAY_PARTITION variable = src_id_burst.data complete dim = 0

LOOP_EDL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = edge_props_ddr[i];
    LOOP_EDL_UNPACK:
        for (int j = 0; j < edges_per_word; j++) {
#pragma HLS UNROLL
            if (edges_read + j < num_edges) {
                ap_uint<bits_per_edge> packed_edge = wide_word.range(
                    (j + 1) * bits_per_edge - 1, j * bits_per_edge);
                edge_t edge;
                node_id_t src_id;
                edge.dst_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);
                edge.src_id =
                    packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);
                src_id = edge.src_id;

                edge_batch.edges[j] = edge;
                src_id_burst.data[j] = src_id;
            }
        }
        edges_read += edges_per_word;
        edge_batch.end_pos = (edges_read <= num_edges)
                                 ? edges_per_word
                                 : (num_edges % edges_per_word);
        edge_stream.write(edge_batch);
        edge_batch.end_pos = 0;
    }
}

// ap_fixed_pod_t get_val_from_bus(bus_word_t bus_data,
//                                 ap_uint<30> position_in_bus) {
// #pragma HLS INLINE
//     switch (position_in_bus) {
//     case 0:
//         return bus_data.range(31, 0);
//     case 1:
//         return bus_data.range(63, 32);
//     case 2:
//         return bus_data.range(95, 64);
//     case 3:
//         return bus_data.range(127, 96);
//     case 4:
//         return bus_data.range(159, 128);
//     case 5:
//         return bus_data.range(191, 160);
//     case 6:
//         return bus_data.range(223, 192);
//     case 7:
//         return bus_data.range(255, 224);
//     case 8:
//         return bus_data.range(287, 256);
//     case 9:
//         return bus_data.range(319, 288);
//     case 10:
//         return bus_data.range(351, 320);
//     case 11:
//         return bus_data.range(383, 352);
//     case 12:
//         return bus_data.range(415, 384);
//     case 13:
//         return bus_data.range(447, 416);
//     case 14:
//         return bus_data.range(479, 448);
//     case 15:
//         return bus_data.range(511, 480);
//     default:
//         return 0;
//     }
// }

void request_manager(hls::stream<edge_descriptor_batch_t> &edge_burst_stm,
                     hls::stream<ppb_request_pkt_t> &ppb_request_stm,
                     hls::stream<ppb_response_pkt_t> &ppb_response_stm,
                     hls::stream<update_tuple_t> &update_set_stm,
                     int32_t part_edge_num) {
    // as we can buffer two vertices in one row with width of 64-bit, we can let
    // the depth go as MAX_VERTICES_IN_ONE_PARTITION / 2.
    bus_word_t src_prop_buffer[PE_NUM][2][SRC_BUFFER_SIZE >> LOG_DIST_PER_WORD];
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
    const int32_t total_edge_sets = (part_edge_num + PE_NUM - 1) / PE_NUM;

    bool wait_flag = 0;

    edge_descriptor_batch_t an_edge_burst;
#pragma HLS ARRAY_PARTITION variable = an_edge_burst.edges complete dim = 0

    ppb_request_pkt_t one_ppb_request;

    ppb_response_pkt_t one_ppb_response;

    distance_t real_edge_weight = (distance_t)1;
    const ap_fixed_pod_t edge_weight =
        (*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight));

    const uint32_t total_rounds =
        (part_edge_num + SRC_BUFFER_SIZE - 1) / SRC_BUFFER_SIZE;

scatterLoop:
    while (true) {
#pragma HLS PIPELINE II = 1
        // logic to fill the ping-pong buffer.
        if ((pp_request_round - pp_read_round) <= 1) {
            if (pp_request_round < pp_read_round)
                pp_request_round = pp_read_round;
            one_ppb_request.data = pp_request_round;
            one_ppb_request.last = 0;
            ppb_request_stm.write(one_ppb_request);
            pp_request_round++;
        }

        if (ppb_response_stm.read_nb(one_ppb_response)) {
            pp_write_round = one_ppb_response.dest << LOG_DIST_PER_WORD >>
                             LOG_SRC_BUFFER_SIZE;

            bool write_buffer = pp_write_round & 0x1;

            int32_t write_idx = one_ppb_response.dest &
                                ((SRC_BUFFER_SIZE >> LOG_DIST_PER_WORD) - 1);

            bus_word_t one_read_burst = one_ppb_response.data;

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
#pragma HLS ARRAY_PARTITION variable = an_update_set.prop complete dim = 0
#pragma HLS ARRAY_PARTITION variable = an_update_set.node_id complete dim = 0

            for (int u = 0; u < PE_NUM; u++) {
#pragma HLS UNROLL
                ap_uint<31> idx =
                    (an_edge_burst.edges[u].src_id % SRC_BUFFER_SIZE);
                ap_uint<30> uram_row_idx = idx >> LOG_DIST_PER_WORD;
                ap_uint<30> uram_row_offset = (idx & (DIST_PER_WORD - 1));

                bus_word_t uram_row =
                    src_prop_buffer[u][read_buffer][uram_row_idx];
                // ap_fixed_pod_t src_prop =
                //     get_val_from_bus(uram_row, uram_row_offset);
                ap_fixed_pod_t src_prop =
                    uram_row.range(DISTANCE_BITWIDTH - 1 +
                                       (uram_row_offset << LOG_DIST_BITWIDTH),
                                   uram_row_offset << LOG_DIST_BITWIDTH);

                an_update_set.prop[u] = (src_prop + edge_weight);
                an_update_set.node_id[u] = an_edge_burst.edges[u].dst_id;
            }
            update_set_stm.write(an_update_set);

            edge_set_cnt++;
        }

        if (exit_flag) {
            one_ppb_request.last = 1;
            ppb_request_stm.write(one_ppb_request);
        exitscatter:
            while (true) {
                ppb_response_stm.read(one_ppb_response);
                if (one_ppb_response.last)
                    break;
            }
            break;
        }
    }
}

inline ap_fixed_pod_t get_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> bits;
    switch (idx) {
    case 0:
        bits = word.range(7, 0);
        break;
    case 1:
        bits = word.range(15, 8);
        break;
    case 2:
        bits = word.range(23, 16);
        break;
    case 3:
        bits = word.range(31, 24);
        break;
    case 4:
        bits = word.range(39, 32);
        break;
    case 5:
        bits = word.range(47, 40);
        break;
    case 6:
        bits = word.range(55, 48);
        break;
    case 7:
        bits = word.range(63, 56);
        break;
    default:
        bits = 0;
        break;
    }
    return bits;
}

inline void set_val(reduce_word_t &word, int idx, ap_fixed_pod_t val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits =
        *reinterpret_cast<ap_uint<DISTANCE_BITWIDTH> *>(&val);
    switch (idx) {
    case 0:
        word.range(7, 0) = val_bits;
        break;
    case 1:
        word.range(15, 8) = val_bits;
        break;
    case 2:
        word.range(23, 16) = val_bits;
        break;
    case 3:
        word.range(31, 24) = val_bits;
        break;
    case 4:
        word.range(39, 32) = val_bits;
        break;
    case 5:
        word.range(47, 40) = val_bits;
        break;
    case 6:
        word.range(55, 48) = val_bits;
        break;
    case 7:
        word.range(63, 56) = val_bits;
        break;
    default:
        break;
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
    int32_t cache_addr_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0

    const int32_t num_words =
        (dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;

#ifdef EMULATION
    memset(prop_mem, 0, sizeof(reduce_word_t) * PE_NUM * MEM_SIZE);
#endif

LOOP_INIT_CACHE_ADDR:
    for (int i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            cache_addr_buffer[pe][i] = -1; // Invalidate cache
        }
    }

    const int32_t total_updates =
        (edge_num + PE_NUM - 1) / PE_NUM; // Assuming one update per node
    const int32_t last_pack_size =
        (edge_num % PE_NUM == 0) ? PE_NUM : (edge_num % PE_NUM);
    // --- Phase 3: Aggregation Loop ---
LOOP_AGGREGATE:
    for (int update_idx = 0; update_idx < total_updates; update_idx++) {
#pragma HLS PIPELINE II = 1
        update_tuple_t one_update;
#pragma HLS ARRAY_PARTITION variable = one_update.prop complete dim = 0
#pragma HLS ARRAY_PARTITION variable = one_update.node_id complete dim = 0
        one_update = update_set_stm.read();
        int32_t cur_pe_end =
            (update_idx == total_updates - 1) ? last_pack_size : PE_NUM;

        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            int32_t key = one_update.node_id[pe];
            if (pe < cur_pe_end && (key & 0x40000000) == 0) { // Valid key check
                ap_fixed_pod_t incoming_dist_pod = one_update.prop[pe];

                int32_t word_addr = (key >> LOG_DISTANCES_PER_REDUCE_WORD);
                int32_t pack_idx = (key & (DISTANCES_PER_REDUCE_WORD - 1));

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

                ap_fixed_pod_t old_dist_pod = get_val(current_word, pack_idx);
                ap_fixed_pod_t new_dist_pod =
                    (old_dist_pod < incoming_dist_pod && old_dist_pod != 0x0)
                        ? old_dist_pod
                        : incoming_dist_pod;

                set_val(current_word, pack_idx, new_dist_pod);

                // Write back to URAM and update cache
                prop_mem[pe][word_addr] = current_word;
                cache_addr_buffer[pe][L] = word_addr;
                cache_data_buffer[pe][L] = current_word;
            }
        }
    }

    // --- Phase 4: Stream out aggregated memory ---
LOOP_STREAM_OUT:
    for (int i = 0; i < num_words; i++) {
#pragma HLS PIPELINE II = 1
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            reduce_word_t word = prop_mem[pe][i];
            pe_mem_outs[pe].write(word);
            prop_mem[pe][i] = 0;
        }
    }
}

void set_word_in_bus(bus_word_t &bus_word, int idx, reduce_word_t reduce_word) {
#pragma HLS INLINE
    switch (idx) {
    case 0:
        bus_word.range(63, 0) = reduce_word;
        break;
    case 1:
        bus_word.range(127, 64) = reduce_word;
        break;
    case 2:
        bus_word.range(191, 128) = reduce_word;
        break;
    case 3:
        bus_word.range(255, 192) = reduce_word;
        break;
    case 4:
        bus_word.range(319, 256) = reduce_word;
        break;
    case 5:
        bus_word.range(383, 320) = reduce_word;
        break;
    case 6:
        bus_word.range(447, 384) = reduce_word;
        break;
    case 7:
        bus_word.range(511, 448) = reduce_word;
        break;
    default:
        break;
    }
}

// Multi-PE drain function
// Collects aggregated data from all PEs and outputs final results
static void
Reduc_105_drain_multi_pe(hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM],
                         hls::stream<write_burst_pkt_t> &kernel_out_stream,
                         int32_t dst_num) {

    // --- Phase 2: High-Performance Drain Loop ---
    write_burst_pkt_t one_write_burst;
    one_write_burst.last = 0;
    uint32_t waiting_count = 0;
    distance_t max_val = (distance_t)(INFINITY_DIST);
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);

LOOP_DRAIN_ADDR:
    for (int32_t base_addr = 0; base_addr < dst_num;
         base_addr += DISTANCES_PER_REDUCE_WORD) {
#pragma HLS PIPELINE II = 1
        ap_fixed_pod_t uram_res[DISTANCES_PER_REDUCE_WORD];
#pragma HLS ARRAY_PARTITION variable = uram_res complete dim = 0
        for (uint32_t i = 0; i < DISTANCES_PER_REDUCE_WORD; i++) {
#pragma HLS UNROLL
            uram_res[i] = max_pod;
        }
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            reduce_word_t word = pe_mem_in[pe_idx].read();
            for (uint32_t i = 0; i < DISTANCES_PER_REDUCE_WORD; i++) {
#pragma HLS UNROLL
                ap_fixed_pod_t incoming_dist_pod = get_val(word, i);
                uram_res[i] = (uram_res[i] < incoming_dist_pod ||
                               incoming_dist_pod == 0x0)
                                  ? uram_res[i]
                                  : incoming_dist_pod;
            }
        }
        reduce_word_t sum_word;
        sum_word.range(7, 0) = uram_res[0];
        sum_word.range(15, 8) = uram_res[1];
        sum_word.range(23, 16) = uram_res[2];
        sum_word.range(31, 24) = uram_res[3];
        sum_word.range(39, 32) = uram_res[4];
        sum_word.range(47, 40) = uram_res[5];
        sum_word.range(55, 48) = uram_res[6];
        sum_word.range(63, 56) = uram_res[7];
        set_word_in_bus(one_write_burst.data, waiting_count, sum_word);
        waiting_count++;
        if (waiting_count == 8) {
            waiting_count = 0;
            kernel_out_stream.write(one_write_burst);
        }
    }
    if (waiting_count != 0) {
        kernel_out_stream.write(one_write_burst);
    }
}

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void
graphyflow_little(const bus_word_t *edge_props, int32_t num_nodes,
                  int32_t num_edges, int32_t dst_num,
                  hls::stream<ppb_request_pkt_t> &ppb_req_stream,
                  hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
                  hls::stream<write_burst_pkt_t> &kernel_out_stream) {
#pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = edge_props
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = dst_num
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW

    // Existing streams
    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 32
    hls::stream<update_tuple_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 8
    hls::stream<reduce_word_t> pe_mem_outs[PE_NUM];
#pragma HLS STREAM variable = pe_mem_outs depth = 8

    // --- Data Loading ---
    edge_descriptor_loader(edge_props, edge_stream, num_edges);
    request_manager(edge_stream, ppb_req_stream, ppb_resp_stream,
                    stream_edge_data, num_edges);

    // --- Reduction ---
    Reduc_105_unit_reduce(stream_edge_data, pe_mem_outs, num_edges, dst_num);
    Reduc_105_drain_multi_pe(pe_mem_outs, kernel_out_stream, dst_num);
}

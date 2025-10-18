#include "graphyflow_little.h"
// #include <stdio.h>

using uint = unsigned int;

ap_uint<4> count_end_ones(ap_uint<PE_NUM> valid_mask) {
#pragma HLS INLINE
    ap_uint<4> count = 0;
    switch (valid_mask) {
    case 0:
        count = 0;
        break;
    case 1:
        count = 1;
        break;
    case 3:
        count = 2;
        break;
    case 7:
        count = 3;
        break;
    case 15:
        count = 4;
        break;
    case 31:
        count = 5;
        break;
    case 63:
        count = 6;
        break;
    case 127:
        count = 7;
        break;
    case 255:
        count = 8;
        break;
    default:
        break;
    }
    return count;
}

// --- MODIFIED: Reads 512-bit words and unpacks packed edge data (48 bits per
// edge).
// static void
// edge_descriptor_loader(const bus_word_t *edge_props_ddr,
//                        hls::stream<edge_descriptor_batch_t> &edge_stream,
//                        int32_t num_edges) {
//     const int bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
//     const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
//     const int num_wide_reads =
//         (num_edges + edges_per_word - 1) / edges_per_word;

//     int edges_read = 0;
//     edge_descriptor_batch_t edge_batch;
// #pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
//     edge_batch.end_pos = 0;

// LOOP_EDL_READ:
//     for (int i = 0; i < num_wide_reads; i++) {
// #pragma HLS PIPELINE II = 1
//         bus_word_t wide_word = edge_props_ddr[i];
//         // // printf("Read a new edge.\n");
//     LOOP_EDL_UNPACK:
//         for (int j = 0; j < edges_per_word; j++) {
// #pragma HLS UNROLL
//             if (edges_read < num_edges) {
//                 ap_uint<bits_per_edge> packed_edge = wide_word.range(
//                     (j + 1) * bits_per_edge - 1, j * bits_per_edge);

//                 node_with_prop_t edge;
//                 edge.node_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);
//                 edge.prop =
//                     packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);
//                 //// printf("[LITTLE]Loaded edge: dst=%d, weight=%f\n",
//                 //edge.node_id, (float)*reinterpret_cast<distance_t
//                 //*>(&edge.prop)); fflush(NULL);

//                 edge_batch.edges[edge_batch.end_pos++] = edge;
//                 edges_read++;

//                 if (edge_batch.end_pos == PE_NUM) {
//                     edge_stream.write(edge_batch);
//                     edge_batch.end_pos = 0;
//                 }
//             }
//         }
//     }

//     // Send any remaining partial batch
//     if (edge_batch.end_pos > 0) {
//         edge_stream.write(edge_batch);
//     }
// }

static void
edge_descriptor_loader(const bus_word_t *edge_props_ddr,
                       hls::stream<edge_descriptor_batch_t> &edge_stream,
                       int32_t num_edges) {
    const int bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
    const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
    const int num_wide_reads =
        (num_edges + edges_per_word - 1) / edges_per_word;

    int edges_read = 0;
    edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
    edge_batch.end_pos = 0;

#if (NODE_ID_BITWIDTH == 32) && (WEIGHT_BITWIDTH == 32)
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
                node_with_prop_t edge;
                edge.node_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);
                edge.prop =
                    packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);

                edge_batch.edges[j] = edge;
            }
        }
        edges_read += edges_per_word;
        edge_batch.end_pos = (edges_read <= num_edges)
                                 ? edges_per_word
                                 : (num_edges % edges_per_word);
        edge_stream.write(edge_batch);
        edge_batch.end_pos = 0;
    }
#else
// Add support for other bitwidth combinations if needed.
#error                                                                         \
    "edge_descriptor_loader currently only supports 32-bit node_id and 32-bit weight."
#endif
}

// new COO loader

template <typename T1, typename T2>
void stream2axistream(hls::stream<T1> &stream, hls::stream<T2> &axi_stream) {

// printf("DEBUG line 106\n");fflush(NULL);
stream2axistream:
    while (true) {

        T1 tmp_t1 = stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.request_round;
        tmp_t2.last = tmp_t1.end_flag;
        // write_to_stream(axi_stream, tmp_t2);
        axi_stream.write(tmp_t2);
        if (tmp_t1.end_flag) {
            break;
            // printf("DEBUG done 119\n");fflush(NULL);
        }
    }
}

template <typename T1, typename T2>
void axistream2stream(hls::stream<T1> &axi_stream, hls::stream<T2> &stream) {
// printf("DEBUG line 129\n");fflush(NULL);
axistream2stream:
    while (true) {

        T1 tmp_t1 = axi_stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.data;
        tmp_t2.addr = tmp_t1.dest;
        tmp_t2.end_flag = tmp_t1.last;
        // write_to_stream(stream, tmp_t2);
        stream.write(tmp_t2);
        if (tmp_t2.end_flag) {
            break;
            // printf("DEBUG line 143\n");fflush(NULL);
        }
    }
}

// 新增
static void ping_pong_buffer_manager(
    hls::stream<ppb_request_dt> &ppb_request_stm,
    hls::stream<ppb_response_dt> &ppb_response_stm,
    hls::stream<edge_descriptor_batch_t> &edge_batch_stream,
    hls::stream<node_id_burst_t> &stream_src_id,
    hls::stream<edge_batch_t> &stream_edge_data, int32_t num_edges) {

    // as we can buffer two vertices in one row with width of 64-bit, we can let
    // the depth go as MAX_VERTICES_IN_ONE_PARTITION / 2.
    ap_uint<512> src_prop_buffer[SCATTER_PE_NUM][2][SRC_BUFFER_SIZE >> 4];

#pragma HLS ARRAY_PARTITION variable = src_prop_buffer dim = 1 complete
#pragma HLS BIND_STORAGE variable = src_prop_buffer type = RAM_S2P impl = BRAM
#pragma HLS dependence variable = src_prop_buffer inter false

    ap_uint<32> pp_read_idx = 0;
    ap_uint<32> pp_write_idx = 0;

    ap_uint<32> pp_reponse_idx = 0;

    ap_uint<32> pp_read_round = 0;
    ap_uint<32> pp_write_round = 0;

    ap_uint<32> pp_request_round = 0;

    ap_uint<32> edge_set_cnt = 0;

    bool wait_flag = 0;

    ppb_request_dt one_ppb_request;

    ppb_response_dt one_ppb_response;

    edge_descriptor_batch_t an_edge_desc_batch;
    node_id_burst_t a_src_id_burst;

scatterLoop:
    while (true) {
#pragma HLS PIPELINE II = 1
        if ((pp_request_round - pp_read_round) <= 1) {
            if (pp_request_round < pp_read_round)
                pp_request_round = pp_read_round;

            one_ppb_request.request_round = pp_request_round;
            one_ppb_request.end_flag = 0;

            ppb_request_stm.write(one_ppb_request);
            pp_request_round++;
            // printf("DEBUG write once\n");fflush(NULL);
        }

        if (ppb_response_stm.read_nb(one_ppb_response)) {

            pp_write_round = one_ppb_response.addr << 4 >> LOG2_SRC_BUFFER_SIZE;

            bool write_buffer = pp_write_round & 0x1;

            uint write_idx =
                one_ppb_response.addr & ((SRC_BUFFER_SIZE >> 4) - 1);

            ap_uint<512> one_read_burst = one_ppb_response.data;

            for (int u = 0; u < SCATTER_PE_NUM; u++) {
#pragma HLS UNROLL
                src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;
            }
            // printf("DEBUG read once\n");fflush(NULL);
        }

        if (!wait_flag) {
            // 从两个流中读取数据
            an_edge_desc_batch = edge_batch_stream.read();
            a_src_id_burst = stream_src_id.read();
        }

        pp_read_round = (a_src_id_burst.data[0].range(30, 0) / SRC_BUFFER_SIZE);

        if (pp_read_round >= pp_write_round)
            wait_flag = 1;
        else
            wait_flag = 0;

        if (!wait_flag) {

            bool read_buffer = pp_read_round & 0x1;

            edge_batch_t output_batch;
#pragma HLS DATA_PACK variable = output_batch

            for (int u = 0; u < PE_NUM; u++) {
#pragma HLS UNROLL

                node_id_t src_id = a_src_id_burst.data[u];
                ap_uint<31> idx = (src_id.range(30, 0) % SRC_BUFFER_SIZE);
                ap_uint<30> uram_row_idx = idx >> 4;
                ap_uint<30> uram_row_offset = (idx & 0xf);

                ap_uint<512> uram_row =
                    src_prop_buffer[u][read_buffer][uram_row_idx];

                ap_uint<32> src_prop = uram_row.range(
                    31 + (uram_row_offset << 5), (uram_row_offset << 5));

                // 打包成 edge_batch_t
                output_batch.src_distances[u] = src_prop;
                output_batch.dsts[u] = an_edge_desc_batch.edges[u].node_id;
                output_batch.weights[u] = an_edge_desc_batch.edges[u].prop;
            }

            output_batch.end_pos = an_edge_desc_batch.end_pos;
            output_batch.end_flag = false;

            stream_edge_data.write(output_batch);
            edge_set_cnt++;
        }

        if (edge_set_cnt >= (num_edges >> LOG_PE_NUM)) {
            one_ppb_request.end_flag = 1;
            ppb_request_stm.write(one_ppb_request);
        exitscatter:
            while (1) {
                ppb_response_stm.read_nb(one_ppb_response);
                if (one_ppb_response.end_flag)
                    break;
            }

            break;
        }
    }
}

ap_fixed_pod_t get_val_from_256_bus(const ap_uint<256> bus, int offset) {
#pragma HLS INLINE
    switch (offset) {
    case 0:
        return bus.range(31, 0);
    case 1:
        return bus.range(63, 32);
    case 2:
        return bus.range(95, 64);
    case 3:
        return bus.range(127, 96);
    case 4:
        return bus.range(159, 128);
    case 5:
        return bus.range(191, 160);
    case 6:
        return bus.range(223, 192);
    case 7:
        return bus.range(255, 224);
    default:
        return 0;
    }
}

static void node_property_responder(
    hls::stream<b_node_distance_burst_t> &node_distance_burst_stream,
    int32_t num_nodes, hls::stream<node_dist_batch_t> &all_distances_stream) {
    node_dist_batch_t dist_batch;
    dist_batch.end_flag = false;
    //     const int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
    // LOOP_FOR_14:
    // for (uint32_t node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
    //      node_burst_idx++) {
    uint32_t node_burst_idx = 0;
    while (true) {
#pragma HLS PIPELINE II = 1
        b_node_distance_burst_t node_distance_burst;
        node_distance_burst = node_distance_burst_stream.read();
        if (node_distance_burst.last) {
            break;
        }
        const int32_t base_idx = (node_burst_idx << LOG_PE_NUM);
        node_burst_idx++;
    LOOP_FOR_13:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            // Direct indexing with pe_idx to avoid race conditions in UNROLL
            dist_batch.data[pe_idx] =
                get_val_from_256_bus(node_distance_burst.data, pe_idx);
        }
        // Calculate end_pos as the number of valid nodes in this burst
        const int32_t remaining_nodes = num_nodes - base_idx;
        dist_batch.end_pos =
            (remaining_nodes < PE_NUM) ? remaining_nodes : PE_NUM;
        all_distances_stream.write(dist_batch);
    }
    dist_batch.end_flag = true;
    dist_batch.end_pos = 0;
    all_distances_stream.write(dist_batch);
}

// --- REWRITTEN: New final_writeback function packs results into 512-bit words.
static void final_writeback(hls::stream<internal_end_data_batch_t> &in_stream,
                            bus_word_t *out_ddr) {
    const int bits_per_output =
        NODE_ID_BITWIDTH + DISTANCE_BITWIDTH + OUT_END_MARKER_BITWIDTH;
    const int outputs_per_word = AXI_BUS_WIDTH / bits_per_output;

    bus_word_t write_word = 0;
    int pack_count = 0;
    int ddr_addr = 0;

LOOP_WRITEBACK_MAIN:
    while (true) {
#pragma HLS PIPELINE II = 1
        internal_end_data_batch_t in_batch;
        if (in_stream.read_nb(in_batch)) {

        LOOP_WRITEBACK_PACK:
            for (int i = 0; i < in_batch.end_pos; i++) {
#pragma HLS UNROLL
                node_with_prop_t item = in_batch.data[i];
                ap_uint<bits_per_output> packed_output;
                // // printf("[LITTLE]Packing output: node_id=%d,
                // distance=%f\n", (int)item.node_id,
                // (float)*reinterpret_cast<distance_t
                // *>(&item.prop)); fflush(NULL);
                packed_output.range(NODE_ID_BITWIDTH - 1, 0) = item.node_id;
                packed_output.range(NODE_ID_BITWIDTH + DISTANCE_BITWIDTH - 1,
                                    NODE_ID_BITWIDTH) = item.prop;
                out_end_marker_t end_marker =
                    0; // No end marker for regular entries
                packed_output.range(bits_per_output - 1,
                                    NODE_ID_BITWIDTH + DISTANCE_BITWIDTH) =
                    end_marker;
                ap_fixed_pod_t tmp_dist = packed_output.range(
                    NODE_ID_BITWIDTH + DISTANCE_BITWIDTH - 1, NODE_ID_BITWIDTH);
                // // printf("[LITTLE]Packing output: node_id=%d,
                // distance=%f\n", (int)packed_output.range(NODE_ID_BITWIDTH -
                // 1, 0), (float)*reinterpret_cast<distance_t *>(&tmp_dist));
                // fflush(NULL);

                int start_bit = pack_count * bits_per_output;
                write_word.range(start_bit + bits_per_output - 1, start_bit) =
                    packed_output;

                pack_count++;
                if (pack_count == outputs_per_word) {
                    // // printf("[LITTLE] Writing packed word to DDR at address
                    // %d\n", ddr_addr); fflush(NULL);
                    out_ddr[ddr_addr++] = write_word;
                    write_word = 0;
                    pack_count = 0;
                }
            }

            if (in_batch.end_flag) {
                break;
            }
        }
    }

    // if pack_count is 0, write a final end marker word
    // else, put the end marker in the current write_word and write it
    ap_uint<bits_per_output> end_marker;
    end_marker.range(NODE_ID_BITWIDTH - 1, 0) = 0;
    end_marker.range(NODE_ID_BITWIDTH + DISTANCE_BITWIDTH - 1,
                     NODE_ID_BITWIDTH) = 0; // Distance = 0
    end_marker.range(bits_per_output - 1,
                     NODE_ID_BITWIDTH + DISTANCE_BITWIDTH) =
        (out_end_marker_t)1; // End marker = 1
    if (pack_count == 0) {
        bus_word_t end_word = 0;
        end_word.range(bits_per_output - 1, 0) = end_marker;
        out_ddr[ddr_addr++] = end_word;
    } else {
        int start_bit = pack_count * bits_per_output;
        write_word.range(start_bit + bits_per_output - 1, start_bit) =
            end_marker;
        out_ddr[ddr_addr++] = write_word;
    }
}

// --- 2. Utility Network Functions ---
static void stream_zipper_3(
    hls::stream<struct_ibu_14_t> &in_key_batch_stream,
    hls::stream<internal_end_data_batch_t> &in_transform_batch_stream,
    hls::stream<struct_kbu_50_t> &out_pair_batch_stream) {
    struct_ibu_14_t key_batch;
    internal_end_data_batch_t transform_batch;
    struct_kbu_50_t out_batch;
LOOP_WHILE_79:
    while (true) {
#pragma HLS PIPELINE
        key_batch = in_key_batch_stream.read();
        transform_batch = in_transform_batch_stream.read();
    LOOP_FOR_78:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch.data[i].key = key_batch.data[i];
            out_batch.data[i].transform = transform_batch.data[i];
        }
        out_batch.end_flag = key_batch.end_flag;
        out_batch.end_pos = key_batch.end_pos;
        out_pair_batch_stream.write(out_batch);
        if (key_batch.end_flag) {
            break;
        }
    }
}

// --- 3. DFIR Component Functions ---
static void Reduc_105_pre_process(
    hls::stream<struct_ibu_14_t> &i_global_data_0,
    hls::stream<struct_nbu_11_t> &i_global_data_1,
    hls::stream<struct_abu_9_t> &i_global_data_2,
    hls::stream<struct_abu_9_t> &i_global_data_3,
    hls::stream<struct_ibu_14_t> &intermediate_key,
    hls::stream<internal_end_data_batch_t> &intermediate_transform) {
    struct_ibu_14_t in_batch_i_global_data_0;
    struct_nbu_11_t in_batch_i_global_data_1;
    struct_abu_9_t in_batch_i_global_data_2;
    struct_abu_9_t in_batch_i_global_data_3;
    struct_ibu_14_t out_batch_intermediate_key;
    internal_end_data_batch_t out_batch_intermediate_transform;
    bool end_flag;
LOOP_WHILE_81:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_global_data_0 = i_global_data_0.read();
        in_batch_i_global_data_1 = i_global_data_1.read();
        in_batch_i_global_data_2 = i_global_data_2.read();
        in_batch_i_global_data_3 = i_global_data_3.read();
    LOOP_FOR_80:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            int32_t key_out_elem;
            node_with_prop_t transform_out_elem;
            // -- Inline sub graph --
            // Inlining fused_op_155
            // -- Begin Nested Inline for FusedOp fused_op_155 --
            // Inlining Place_151
            key_out_elem = in_batch_i_global_data_0.data[i];
            // -- End Nested Inline for FusedOp fused_op_155 --
            // -- Inline sub graph end --
            // -- Inline sub graph --
            // Inlining fused_op_185
            // -- Begin Nested Inline for FusedOp fused_op_185 --
            // Inlining BinOp_68
            ap_fixed_pod_t fused_temp_BinOp_68_o_0;
            distance_t lhs_68 = *reinterpret_cast<distance_t *>(
                &in_batch_i_global_data_2.data[i]);
            distance_t rhs_68 = *reinterpret_cast<distance_t *>(
                &in_batch_i_global_data_3.data[i]);
            distance_t temp_BinOp_68_o_0_ap_result;
            temp_BinOp_68_o_0_ap_result = (lhs_68 + rhs_68);
            fused_temp_BinOp_68_o_0 = *reinterpret_cast<ap_fixed_pod_t *>(
                &temp_BinOp_68_o_0_ap_result);
            // Inlining Gathe_179
            transform_out_elem.prop = fused_temp_BinOp_68_o_0;
            transform_out_elem.node_id = in_batch_i_global_data_1.data[i];
            // -- End Nested Inline for FusedOp fused_op_185 --
            // -- Inline sub graph end --
            out_batch_intermediate_key.data[i] = key_out_elem;
            out_batch_intermediate_transform.data[i] = transform_out_elem;
        }
        out_batch_intermediate_key.end_flag = in_batch_i_global_data_0.end_flag;
        out_batch_intermediate_key.end_pos = in_batch_i_global_data_0.end_pos;
        out_batch_intermediate_transform.end_flag =
            in_batch_i_global_data_0.end_flag;
        out_batch_intermediate_transform.end_pos =
            in_batch_i_global_data_0.end_pos;
        intermediate_key.write(out_batch_intermediate_key);
        intermediate_transform.write(out_batch_intermediate_transform);
        end_flag = in_batch_i_global_data_0.end_flag;
        if (end_flag) {
            break;
        }
    }
}

inline distance_t get_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> bits;
    switch (idx) {
    case 0:
        bits = word.range(DISTANCE_BITWIDTH - 1, 0);
        break;
    case 1:
        bits = word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH);
        break;
    case 2:
        bits =
            word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1));
        break;
    default:
        bits = 0;
        break;
    }
    // Convert bits back to distance_t (floating point)
    return *reinterpret_cast<distance_t *>(&bits);
}

inline void set_val(reduce_word_t &word, int idx, distance_t val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits =
        *reinterpret_cast<ap_uint<DISTANCE_BITWIDTH> *>(&val);
    switch (idx) {
    case 0:
        word.range(DISTANCE_BITWIDTH - 1, 0) = val_bits;
        break;
    case 1:
        word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH) = val_bits;
        break;
    case 2:
        word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1)) =
            val_bits;
        break;
    default:
        break;
    }
}

static void
Reduc_105_unit_reduce(hls::stream<struct_kbu_50_t> &in_kt_pair_stream,
                      hls::stream<internal_end_data_batch_t> &o_0,
                      int32_t dst_num) { // <-- 函数签名已修改
    // --- Phase 1: Memory Declaration ---
    // (此部分保持不变)
    const int MEM_SIZE =
        (MAX_NUM + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;
    reduce_word_t prop_mem[PE_NUM][MEM_SIZE];
#pragma HLS BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM
#pragma HLS ARRAY_PARTITION variable = prop_mem complete dim = 1
#pragma HLS dependence variable = prop_mem inter false direction = WAW
#pragma HLS dependence variable = prop_mem inter false direction = RAW

    bool prop_valid[PE_NUM][MAX_NUM];
#pragma HLS BIND_STORAGE variable = prop_valid type = RAM_2P impl = BRAM
#pragma HLS ARRAY_PARTITION variable = prop_valid complete dim = 1
#pragma HLS dependence variable = prop_valid inter false direction = WAW
#pragma HLS dependence variable = prop_valid inter false direction = RAW

    typedef ap_uint<16> addr_map_t;
    addr_map_t key_to_addr_map[PE_NUM][MAX_NUM];
#pragma HLS BIND_STORAGE variable = key_to_addr_map type = RAM_1P impl = BRAM
#pragma HLS ARRAY_PARTITION variable = key_to_addr_map complete dim = 1

    reduce_word_t cache_data_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
    reduce_word_t tmp_cache_data_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable = tmp_cache_data_buffer complete dim = 0
    int cache_addr_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0
    int tmp_cache_addr_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable = tmp_cache_addr_buffer complete dim = 0

    // --- Phase 2: Initialization ---
    // (此部分保持不变)
LOOP_INIT_LITTLE:
    for (int i = 0; i < MAX_NUM; i++) {
#pragma HLS PIPELINE II = 1
        addr_map_t map_val;
        map_val.range(15, 2) = i / DISTANCES_PER_REDUCE_WORD; // word_addr
        map_val.range(1, 0) = i % DISTANCES_PER_REDUCE_WORD;  // pack_idx
        // key_to_addr_map[i] = map_val;
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            key_to_addr_map[pe][i] = map_val;
            prop_valid[pe][i] = false;
        }
    }
LOOP_INIT_LITTLE_CACHE:
    for (int i = 0; i < L + 1; i++) {
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            cache_addr_buffer[pe][i] = -1; // Invalidate cache
        }
    }

    // --- Phase 3: Aggregation Loop ---
    // (此部分保持不变)
LOOP_AGGREGATE_LITTLE:
    while (true) {
#pragma HLS PIPELINE II = 1
        struct_kbu_50_t in_batch;
        if (in_kt_pair_stream.read_nb(in_batch)) {
            for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
                if (pe < in_batch.end_pos) {
                    int key = in_batch.data[pe].key;
                    ap_fixed_pod_t incoming_dist_pod =
                        in_batch.data[pe].transform.prop;

                    // Note: In the original code, key_to_addr_map was
                    // PE-specific. It's more efficient to have a single shared
                    // map if the mapping is the same. Using a non-PE-specific
                    // map here for optimization.
                    addr_map_t map_val = key_to_addr_map[pe][key];
                    int word_addr = map_val.range(15, 2);
                    int pack_idx = map_val.range(1, 0);

                    reduce_word_t current_word = prop_mem[pe][word_addr];

                    for (int i = L; i >= 0; --i) {
#pragma HLS UNROLL
                        if (cache_addr_buffer[pe][i] == word_addr) {
                            current_word = cache_data_buffer[pe][i];
                            break;
                        }
                    }

                    bool is_valid = prop_valid[pe][key];

                    for (int i = 0; i < L; i++) {
#pragma HLS UNROLL
                        tmp_cache_addr_buffer[pe][i] =
                            cache_addr_buffer[pe][i + 1];
                        tmp_cache_data_buffer[pe][i] =
                            cache_data_buffer[pe][i + 1];
                    }

                    for (int i = 0; i < L; i++) {
#pragma HLS UNROLL
                        cache_addr_buffer[pe][i] = tmp_cache_addr_buffer[pe][i];
                        cache_data_buffer[pe][i] = tmp_cache_data_buffer[pe][i];
                    }

                    distance_t new_dist_fp;
                    distance_t incoming_dist_fp =
                        *reinterpret_cast<distance_t *>(&incoming_dist_pod);

                    if (is_valid) {
                        distance_t old_dist_fp =
                            get_val(current_word, pack_idx);
                        new_dist_fp = (old_dist_fp < incoming_dist_fp)
                                          ? old_dist_fp
                                          : incoming_dist_fp;
                    } else {
                        new_dist_fp = incoming_dist_fp;
                    }

                    prop_valid[pe][key] = true;
                    set_val(current_word, pack_idx, new_dist_fp);
                    prop_mem[pe][word_addr] = current_word;
                    cache_addr_buffer[pe][L] = word_addr;
                    cache_data_buffer[pe][L] = current_word;
                }
            }
            if (in_batch.end_flag) {
                break;
            }
        }
    }

    // --- Phase 4: OPTIMIZED Final Merge and Drain Loop ---
    internal_end_data_batch_t data_pack;
#pragma HLS ARRAY_PARTITION variable = data_pack.data complete dim = 0
    data_pack.end_flag = false;
    data_pack.end_pos = 0;

LOOP_DRAIN_OPTIMIZED:
    // 遍历已知的有效 key (0 - dst_num-1)
    for (int key = 0; key < dst_num; key++) {
#pragma HLS PIPELINE II = 1

        addr_map_t map_val = key_to_addr_map[key];
        int word_addr = map_val.range(15, 2);
        int pack_idx = map_val.range(1, 0);

        distance_t min_dist = (distance_t)INFINITY_DIST;

    LOOP_MERGE_PES:
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            // 仍然需要检查 prop_valid:
            // key有效意味着至少一个PE更新了它，但不代表所有PE都更新了它。
            // 此检查用于在跨PE合并时，过滤掉那些包含无效数据的PE。
            if (prop_valid[pe][key]) {
                reduce_word_t word = prop_mem[pe][word_addr];
                distance_t dist_fp = get_val(word, pack_idx);

                if (dist_fp < min_dist) {
                    min_dist = dist_fp;
                }
            }
        }

        // 无需 'valid_found' 标志，因为保证key有效，min_dist一定会被更新。
        data_pack.data[data_pack.end_pos].prop =
            *reinterpret_cast<ap_fixed_pod_t *>(&min_dist);
        data_pack.data[data_pack.end_pos].node_id = key;
        data_pack.end_pos++;

        if (data_pack.end_pos == PE_NUM) {
            data_pack.end_flag = false;
            o_0.write(data_pack);
            data_pack.end_pos = 0;
        }
    }

    // 发送最后一个（可能不满的）数据包
    if (data_pack.end_pos > 0) {
        data_pack.end_flag = false;
        o_0.write(data_pack);
    }

    // 发送结束标志
    data_pack.end_flag = true;
    data_pack.end_pos = 0;
    o_0.write(data_pack);
}
static void Scatt_234(hls::stream<struct_sbu_7_t> &i_0,
                      hls::stream<struct_abu_9_t> &o_0,
                      hls::stream<struct_nbu_11_t> &o_1,
                      hls::stream<struct_abu_9_t> &o_2) {
    struct_sbu_7_t in_batch_i_0;
    struct_abu_9_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    struct_abu_9_t out_batch_o_2;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_91:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_FOR_90:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].ele_0;
            out_batch_o_1.data[i] = in_batch_i_0.data[i].ele_1;
            out_batch_o_2.data[i] = in_batch_i_0.data[i].ele_2;
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        out_batch_o_2.end_flag = end_flag;
        out_batch_o_2.end_pos = end_pos;
        o_2.write(out_batch_o_2);
        if (end_flag) {
            break;
        }
    }
}

static void Memor_231(hls::stream<struct_ibu_14_t> &o_0_node_id,
                      hls::stream<struct_nbu_11_t> &i_0_node_id) {
    struct_nbu_11_t in_batch_i_0_node_id;
    struct_ibu_14_t out_batch_o_0_node_id;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_93:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_node_id = i_0_node_id.read();
    LOOP_FOR_92:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0_node_id.data[i] = in_batch_i_0_node_id.data[i];
        }
        end_flag = in_batch_i_0_node_id.end_flag;
        end_pos = in_batch_i_0_node_id.end_pos;
        out_batch_o_0_node_id.end_flag = end_flag;
        out_batch_o_0_node_id.end_pos = end_pos;
        o_0_node_id.write(out_batch_o_0_node_id);
        if (end_flag) {
            break;
        }
    }
}

static void CopyC_247(hls::stream<struct_nbu_11_t> &i_0,
                      hls::stream<struct_nbu_11_t> &o_0,
                      hls::stream<struct_nbu_11_t> &o_1) {
    struct_nbu_11_t in_batch_i_0;
    struct_nbu_11_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_95:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_FOR_94:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i];
            out_batch_o_1.data[i] = in_batch_i_0.data[i];
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        if (end_flag) {
            break;
        }
    }
}

static void fused_op_269(hls::stream<struct_abu_9_t> &i_0,
                         hls::stream<struct_nbu_11_t> &i_1,
                         hls::stream<struct_abu_9_t> &i_2,
                         hls::stream<struct_sbu_7_t> &o_0) {
    struct_abu_9_t in_batch_i_0;
    struct_nbu_11_t in_batch_i_1;
    struct_abu_9_t in_batch_i_2;
    struct_sbu_7_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_97:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
    LOOP_FOR_96:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // -- Inlining FusedOp fused_op_269 --
            // Inlining Gathe_262
            out_batch_o_0.data[i].ele_0 = in_batch_i_0.data[i];
            out_batch_o_0.data[i].ele_1 = in_batch_i_1.data[i];
            out_batch_o_0.data[i].ele_2 = in_batch_i_2.data[i];
            // -- End Inlining FusedOp fused_op_269 --
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        if (end_flag) {
            break;
        }
    }
}

static void Memor_274(hls::stream<edge_batch_t> &i_0_edge_id,
                      hls::stream<struct_abu_9_t> &o_0_edge_src_distance,
                      hls::stream<struct_nbu_11_t> &o_0_edge_dst,
                      hls::stream<struct_abu_9_t> &o_0_edge_weight) {
    edge_batch_t in_batch_i_0_edge_id;
    struct_abu_9_t out_batch_o_0_edge_src_distance;
    struct_nbu_11_t out_batch_o_0_edge_dst;
    struct_abu_9_t out_batch_o_0_edge_weight;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_99:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_edge_id = i_0_edge_id.read();
    LOOP_FOR_98:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0_edge_src_distance.data[i] =
                in_batch_i_0_edge_id.src_distances[i];
            out_batch_o_0_edge_dst.data[i] = in_batch_i_0_edge_id.dsts[i];
            out_batch_o_0_edge_weight.data[i] = in_batch_i_0_edge_id.weights[i];
        }
        end_flag = in_batch_i_0_edge_id.end_flag;
        end_pos = in_batch_i_0_edge_id.end_pos;
        out_batch_o_0_edge_src_distance.end_flag = end_flag;
        out_batch_o_0_edge_src_distance.end_pos = end_pos;
        o_0_edge_src_distance.write(out_batch_o_0_edge_src_distance);
        out_batch_o_0_edge_dst.end_flag = end_flag;
        out_batch_o_0_edge_dst.end_pos = end_pos;
        o_0_edge_dst.write(out_batch_o_0_edge_dst);
        out_batch_o_0_edge_weight.end_flag = end_flag;
        out_batch_o_0_edge_weight.end_pos = end_pos;
        o_0_edge_weight.write(out_batch_o_0_edge_weight);
        if (end_flag) {
            break;
        }
    }
}

static void Memor_299(hls::stream<node_dist_batch_t> &i_all_node_distances,
                      hls::stream<struct_abu_9_t> &o_0_node_distance,
                      int32_t dst_num) {
    // Efficiently filters a stream of all node distances against a stream of
    // requested node IDs.
    node_dist_batch_t in_dist_batch;
    struct_abu_9_t out_dist_batch;
#pragma HLS ARRAY_PARTITION variable = in_dist_batch.data complete dim = 0
#pragma HLS ARRAY_PARTITION variable = out_dist_batch.data complete dim = 0
    out_dist_batch.end_flag = false;
LOOP_WHILE_52:
    for (uint32_t in_node_base_id = 0; in_node_base_id < dst_num;
         in_node_base_id += PE_NUM) {
#pragma HLS PIPELINE II = 1
        in_dist_batch = i_all_node_distances.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_dist_batch.data[i] = in_dist_batch.data[i]; // Direct copy
            // // printf("[BIG] Reading node_id %d with distance %f\n",
            // in_node_base_id + i,
            // (float)*reinterpret_cast<distance_t*>(&out_dist_batch.data[i]));
            // fflush(NULL);
        }
        const int remaining_nodes = dst_num - in_node_base_id;
        out_dist_batch.end_pos =
            (remaining_nodes < PE_NUM) ? remaining_nodes : PE_NUM;

        o_0_node_distance.write(out_dist_batch);
    }
    // Send the final (empty) output batch with the end flag
    struct_abu_9_t final_batch;
    final_batch.end_flag = true;
    final_batch.end_pos = 0;
    o_0_node_distance.write(final_batch);
// Drain any remaining batches from the all_distances stream to prevent deadlock
LOOP_WHILE_53:
    while ((!in_dist_batch.end_flag)) {
        in_dist_batch = i_all_node_distances.read();
    }
}
static void Scatt_302(hls::stream<internal_end_data_batch_t> &i_0,
                      hls::stream<struct_abu_9_t> &o_0,
                      hls::stream<struct_nbu_11_t> &o_1) {
    internal_end_data_batch_t in_batch_i_0;
    struct_abu_9_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_103:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_FOR_102:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].prop;
            out_batch_o_1.data[i] = in_batch_i_0.data[i].node_id;
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        if (end_flag) {
            break;
        }
    }
}

static void CopyC_306(hls::stream<struct_nbu_11_t> &i_0,
                      hls::stream<struct_nbu_11_t> &o_0,
                      hls::stream<struct_nbu_11_t> &o_1) {
    struct_nbu_11_t in_batch_i_0;
    struct_nbu_11_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_105:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_FOR_104:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i];
            out_batch_o_1.data[i] = in_batch_i_0.data[i];
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        if (end_flag) {
            break;
        }
    }
}

static void fused_op_294(hls::stream<struct_abu_9_t> &i_0,
                         hls::stream<struct_abu_9_t> &i_1,
                         hls::stream<struct_nbu_11_t> &i_2,
                         hls::stream<internal_end_data_batch_t> &o_0) {
    struct_abu_9_t in_batch_i_0;
    struct_abu_9_t in_batch_i_1;
    struct_nbu_11_t in_batch_i_2;
    internal_end_data_batch_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_107:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
    LOOP_FOR_106:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // -- Inlining FusedOp fused_op_294 --
            // Inlining BinOp_128
            ap_fixed_pod_t fused_temp_BinOp_128_o_0;
            distance_t lhs_128 =
                *reinterpret_cast<distance_t *>(&in_batch_i_0.data[i]);
            distance_t rhs_128 =
                *reinterpret_cast<distance_t *>(&in_batch_i_1.data[i]);
            distance_t temp_BinOp_128_o_0_ap_result;
            temp_BinOp_128_o_0_ap_result =
                (((lhs_128) < (rhs_128) ? lhs_128 : rhs_128));
            fused_temp_BinOp_128_o_0 = *reinterpret_cast<ap_fixed_pod_t *>(
                &temp_BinOp_128_o_0_ap_result);
            // Inlining Gathe_288
            out_batch_o_0.data[i].prop = fused_temp_BinOp_128_o_0;
            out_batch_o_0.data[i].node_id = in_batch_i_2.data[i];
            // -- End Inlining FusedOp fused_op_294 --
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        if (end_flag) {
            break;
        }
    }
}

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void graphyflow_little(
    const bus_word_t *src_ids, const bus_word_t *edge_props, bus_word_t *output,
    int32_t num_nodes, int32_t num_edges, int32_t dst_num,

    // hls::stream<b_cacheline_req_t> &stream_outer_cache_req,
    // hls::stream<b_cacheline_resp_t> &stream_outer_cache_resp,
    // hls::stream<b_node_distance_burst_t> &stream_outer_node_dist

    hls::stream<l_ppb_request_pkt>
        &l_ppb_request_stm, // <--- 修改: 使用 AXI Stream 包类型
    hls::stream<l_ppb_response_pkt>
        &l_ppb_response_stm, // <--- 修改: 使用 AXI Stream 包类型
    hls::stream<b_node_distance_burst_t> &stream_outer_node_dist) {

// new:
#pragma HLS INTERFACE m_axi port = src_ids offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = src_ids
#pragma HLS INTERFACE s_axilite port = edge_props
#pragma HLS INTERFACE s_axilite port = output
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = dst_num
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW

    //// printf("DEBUG little start\n");fflush(NULL);
    // new:
    // Streams for the new COO-style property loading
    hls::stream<node_id_burst_t> stream_src_ids_1;
#pragma HLS STREAM variable = stream_src_ids_1 depth = 32
    hls::stream<node_id_burst_t> stream_src_ids_2;
#pragma HLS STREAM variable = stream_src_ids_2 depth = 32
    hls::stream<ppb_request_dt> ppb_request_stm;
#pragma HLS stream variable = ppb_request_stm depth = 32
    hls::stream<ppb_response_dt> ppb_response_stm;
#pragma HLS stream variable = ppb_response_stm depth = 32
    hls::stream<bus_word_t> prop_streams_for_merge[PE_NUM];
#pragma HLS STREAM variable = prop_streams_for_merge depth = 32

    // Existing streams
    hls::stream<node_distance_burst_t> node_distance_burst_stream_1;
#pragma HLS STREAM variable = node_distance_burst_stream_1 depth = 32
    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 32
    hls::stream<edge_batch_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 32
    hls::stream<node_dist_batch_t> stream_node_dist_data;
#pragma HLS STREAM variable = stream_node_dist_data depth = 32
    hls::stream<internal_end_data_batch_t> stream_result_data;
#pragma HLS STREAM variable = stream_result_data depth = 32

    // new COO loader
    const int num_ids_per_word = AXI_BUS_WIDTH / NODE_ID_BITWIDTH;
    const int num_wide_reads =
        (num_edges + num_ids_per_word - 1) / num_ids_per_word;

    int nodes_read = 0;
    int burst_idx = 0;
    node_id_burst_t burst1, burst2;
LOOP_SIL_READ:

    //// printf("DEBUG num_wide_reads: %d", num_wide_reads);fflush(NULL);
    for (int i = 0; i < num_wide_reads; i++) {
        // #pragma HLS PIPELINE II = 2
        bus_word_t wide_word = src_ids[i];
        //// printf("DEBUG current: %d\n", i);fflush(NULL);
    LOOP_SIL_UNPACK:
        for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
            if (nodes_read + j < num_edges) {
                node_id_t cur_id = wide_word.range(
                    (j + 1) * NODE_ID_BITWIDTH - 1, j * NODE_ID_BITWIDTH);
                burst1.data[j] = cur_id;
                // printf("Loaded node ID %d at burst %d, position %d\n",
                // (int)cur_id, burst_idx, j); fflush(NULL);
            }
        }
        bool burst2_valid = false;
        for (int j = 8; j < 16; j++) {
#pragma HLS UNROLL
            if (nodes_read + j < num_edges) {
                burst2.data[j - 8] = wide_word.range(
                    (j + 1) * NODE_ID_BITWIDTH - 1, j * NODE_ID_BITWIDTH);
                burst2_valid |= true;
                // printf("Loaded node ID %d at burst %d, position %d\n",
                // (int)burst2.data[j - 8], burst_idx + 1, j - 8); fflush(NULL);
            }
        }
        stream_src_ids_1.write(burst1);
        if (burst2_valid) {
            stream_src_ids_1.write(burst2);
        }
        stream_src_ids_2.write(burst1);
        if (burst2_valid) {
            stream_src_ids_2.write(burst2);
        }
        nodes_read += num_ids_per_word;
    }

    // printf("DEBUG:load done\n");fflush(NULL);
    edge_descriptor_loader(edge_props, edge_stream, num_edges);

    // printf("DEBUG start ping pong\n");fflush(NULL);
    ping_pong_buffer_manager(ppb_request_stm, ppb_response_stm, edge_stream,
                             stream_src_ids_1, stream_edge_data, num_edges);
    stream2axistream<ppb_request_dt, l_ppb_request_pkt>(ppb_request_stm,
                                                        l_ppb_request_stm);
    axistream2stream<l_ppb_response_pkt, ppb_response_dt>(l_ppb_response_stm,
                                                          ppb_response_stm);

// new : main dataflow processing
#pragma HLS DATAFLOW
    hls::stream<struct_kbu_50_t> reduce_105_z2u_pair;
#pragma HLS STREAM variable = reduce_105_z2u_pair depth = 4
    hls::stream<struct_ibu_14_t> intermediate_key;
#pragma HLS STREAM variable = intermediate_key depth = 4
    hls::stream<internal_end_data_batch_t> intermediate_transform;
#pragma HLS STREAM variable = intermediate_transform depth = 4
    hls::stream<struct_sbu_7_t> stream_o_0_273;
#pragma HLS STREAM variable = stream_o_0_273 depth = 4
    hls::stream<struct_abu_9_t> stream_o_0_236;
#pragma HLS STREAM variable = stream_o_0_236 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_1_237;
#pragma HLS STREAM variable = stream_o_1_237 depth = 4
    hls::stream<struct_abu_9_t> stream_o_2_238;
#pragma HLS STREAM variable = stream_o_2_238 depth = 4
    hls::stream<struct_ibu_14_t> stream_o_0_node_id_232;
#pragma HLS STREAM variable = stream_o_0_node_id_232 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_1_250;
#pragma HLS STREAM variable = stream_o_1_250 depth = 4
    hls::stream<internal_end_data_batch_t> stream_o_0_107;
#pragma HLS STREAM variable = stream_o_0_107 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_0_249;
#pragma HLS STREAM variable = stream_o_0_249 depth = 4
    hls::stream<struct_abu_9_t> stream_o_0_edge_src_distance_275;
#pragma HLS STREAM variable = stream_o_0_edge_src_distance_275 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_0_edge_dst_277;
#pragma HLS STREAM variable = stream_o_0_edge_dst_277 depth = 4
    hls::stream<struct_abu_9_t> stream_o_0_edge_weight_278;
#pragma HLS STREAM variable = stream_o_0_edge_weight_278 depth = 4
    hls::stream<struct_abu_9_t> stream_o_0_node_distance_300;
#pragma HLS STREAM variable = stream_o_0_node_distance_300 depth = 4
    // hls::stream<struct_nbu_11_t> stream_o_1_309;
    // #pragma HLS STREAM variable = stream_o_1_309 depth = 4
    hls::stream<struct_abu_9_t> stream_o_0_304;
#pragma HLS STREAM variable = stream_o_0_304 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_1_305;
#pragma HLS STREAM variable = stream_o_1_305 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_0_308;
#pragma HLS STREAM variable = stream_o_0_308 depth = 4
    // --- Function Calls (in topological order) ---
    Memor_274(stream_edge_data, stream_o_0_edge_src_distance_275,
              stream_o_0_edge_dst_277, stream_o_0_edge_weight_278);
    fused_op_269(stream_o_0_edge_src_distance_275, stream_o_0_edge_dst_277,
                 stream_o_0_edge_weight_278, stream_o_0_273);
    Scatt_234(stream_o_0_273, stream_o_0_236, stream_o_1_237, stream_o_2_238);
    CopyC_247(stream_o_1_237, stream_o_0_249, stream_o_1_250);
    Memor_231(stream_o_0_node_id_232, stream_o_1_250);
    // --- Start of Reduce Super-Block for Reduc_105 ---
    Reduc_105_pre_process(stream_o_0_node_id_232, stream_o_0_249,
                          stream_o_0_236, stream_o_2_238, intermediate_key,
                          intermediate_transform);
    stream_zipper_3(intermediate_key, intermediate_transform,
                    reduce_105_z2u_pair);
    // printf("[LITTLE]Entering reduction phase.\n");
    // fflush(NULL);
    Reduc_105_unit_reduce(reduce_105_z2u_pair, stream_o_0_107, dst_num);
    // printf("[LITTLE]Reduction phase complete.\n");
    // fflush(NULL);
    // --- End of Reduce Super-Block for Reduc_105 ---

    Scatt_302(stream_o_0_107, stream_o_0_304, stream_o_1_305);
    // CopyC_306(stream_o_1_305, stream_o_0_308, stream_o_1_309);

    node_property_responder(stream_outer_node_dist, num_nodes,
                            stream_node_dist_data);
    Memor_299(stream_node_dist_data, stream_o_0_node_distance_300, dst_num);
    fused_op_294(stream_o_0_304, stream_o_0_node_distance_300, stream_o_1_305,
                 stream_result_data);

    final_writeback(stream_result_data, output);
}

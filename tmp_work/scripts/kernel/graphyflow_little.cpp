#include "graphyflow_little.h"

// --- 1. Memory Helper Functions ---
// --- MODIFIED: Reads 512-bit words and unpacks 24-bit distance values.
static void node_property_loader(
    const bus_word_t *node_distances_ddr,
    hls::stream<node_distance_burst_t> &node_distance_burst_stream_0,
    hls::stream<node_distance_burst_t> &node_distance_burst_stream_1,
    int32_t num_nodes) {
    const int num_dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
    const int num_wide_reads =
        (num_nodes + num_dists_per_word - 1) / num_dists_per_word;

    // Stream 0
    int nodes_read_s0 = 0;
    int burst_idx = 0;
    node_distance_burst_t burst;
LOOP_NPL_S0_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = node_distances_ddr[i];

    LOOP_NPL_S0_UNPACK:
        for (int j = 0; j < num_dists_per_word; j++) {
#pragma HLS UNROLL
            if (nodes_read_s0 < num_nodes) {
                burst.data[burst_idx] = wide_word.range(
                    (j + 1) * DISTANCE_BITWIDTH - 1, j * DISTANCE_BITWIDTH);
                printf("[LITTLE]Loaded node distance: %f\n",
                       (float)*reinterpret_cast<distance_t *>(
                           &burst.data[burst_idx]));
                fflush(NULL);
                burst_idx++;
                nodes_read_s0++;
                if (burst_idx == PE_NUM) {
                    node_distance_burst_stream_0.write(burst);
                    burst_idx = 0;
                }
            }
        }
    }
    if (burst_idx > 0) {
        node_distance_burst_stream_0.write(burst);
    }

    // Stream 1
    int nodes_read_s1 = 0;
    burst_idx = 0;
LOOP_NPL_S1_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = node_distances_ddr[i];

    LOOP_NPL_S1_UNPACK:
        for (int j = 0; j < num_dists_per_word; j++) {
#pragma HLS UNROLL
            if (nodes_read_s1 < num_nodes) {
                burst.data[burst_idx++] = wide_word.range(
                    (j + 1) * DISTANCE_BITWIDTH - 1, j * DISTANCE_BITWIDTH);
                nodes_read_s1++;
                if (burst_idx == PE_NUM) {
                    node_distance_burst_stream_1.write(burst);
                    burst_idx = 0;
                }
            }
        }
    }
    if (burst_idx > 0) {
        node_distance_burst_stream_1.write(burst);
    }
}

// --- MODIFIED: Reads 512-bit words and unpacks packed edge data (48 bits per
// edge).
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

LOOP_EDL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = edge_props_ddr[i];

    LOOP_EDL_UNPACK:
        for (int j = 0; j < edges_per_word; j++) {
#pragma HLS UNROLL
            if (edges_read < num_edges) {
                ap_uint<bits_per_edge> packed_edge = wide_word.range(
                    (j + 1) * bits_per_edge - 1, j * bits_per_edge);

                node_with_prop_t edge;
                edge.node_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);
                edge.prop =
                    packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);
                printf("[LITTLE]Loaded edge: dst=%d, weight=%f\n", edge.node_id,
                       (float)*reinterpret_cast<distance_t *>(&edge.prop));
                fflush(NULL);

                edge_batch.edges[edge_batch.end_pos++] = edge;
                edges_read++;

                if (edge_batch.end_pos == PE_NUM) {
                    edge_stream.write(edge_batch);
                    edge_batch.end_pos = 0;
                }
            }
        }
    }

    // Send any remaining partial batch
    if (edge_batch.end_pos > 0) {
        edge_stream.write(edge_batch);
    }
}

// --- MODIFIED: Reads 512-bit words and unpacks 32-bit offset values.
static void src_offset_loader(const bus_word_t *src_offsets_ddr,
                              hls::stream<int32_t> &src_offsets_stream,
                              int32_t num_nodes) {
    const int offsets_per_word = AXI_BUS_WIDTH / 32;
    const int num_total_offsets = num_nodes + 1;
    const int num_wide_reads =
        (num_total_offsets + offsets_per_word - 1) / offsets_per_word;

    int offsets_read = 0;
LOOP_SOL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = src_offsets_ddr[i];
    LOOP_SOL_UNPACK:
        for (int j = 0; j < offsets_per_word; j++) {
#pragma HLS UNROLL
            if (offsets_read < num_total_offsets) {
                int32_t offset = wide_word.range((j + 1) * 32 - 1, j * 32);
                printf("[LITTLE]Loaded src offset: %d\n", offset);
                fflush(NULL);
                src_offsets_stream.write(offset);
                offsets_read++;
            }
        }
    }
}

static void edge_property_loader_and_dispatcher(
    hls::stream<int32_t> &src_offsets_cache_stream,
    hls::stream<edge_descriptor_batch_t> &edge_stream,
    hls::stream<node_distance_burst_t> &node_distance_burst_stream,
    int32_t num_nodes, hls::stream<edge_batch_t> &response_stream) {
    edge_batch_t current_batch;
#pragma HLS ARRAY_PARTITION variable = current_batch.weights complete dim = 0
#pragma HLS ARRAY_PARTITION variable =                                         \
    current_batch.src_distances complete dim = 0
#pragma HLS ARRAY_PARTITION variable = current_batch.dsts complete dim = 0
#pragma HLS dependence variable = current_batch inter false direction = WAW
    current_batch.end_pos = 0;
    current_batch.end_flag = false;
    edge_descriptor_batch_t edge_batch;
    edge_batch.end_pos = 0;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
#pragma HLS dependence variable = edge_batch inter false direction = WAW
    int32_t edge_batch_pos = 0;
    node_distance_burst_t node_distance_burst;
#pragma HLS ARRAY_PARTITION variable = node_distance_burst.data complete dim = 0
#pragma HLS dependence variable = node_distance_burst inter false
    int32_t start_edge_idx;
    int32_t end_edge_idx;
    start_edge_idx = src_offsets_cache_stream.read();
    const int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
LOOP_FOR_72:
    for (uint32_t node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         node_burst_idx++) {
        node_distance_burst = node_distance_burst_stream.read();
        const int32_t base_idx = (node_burst_idx << LOG_PE_NUM);
    LOOP_FOR_71:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
            if (base_idx + pe_idx >= num_nodes) {
                break;
            }
            ap_fixed_pod_t src_dist;
            src_dist = node_distance_burst.data[pe_idx];
            end_edge_idx = src_offsets_cache_stream.read();
        LOOP_FOR_70:
            for (uint32_t e_idx = start_edge_idx; e_idx < end_edge_idx;
                 e_idx++) {
#pragma HLS PIPELINE II = 1
                if (edge_batch_pos == edge_batch.end_pos) {
                    edge_batch = edge_stream.read();
                    edge_batch_pos = 0;
                }
                node_with_prop_t edge;
                edge = edge_batch.edges[edge_batch_pos++];
                current_batch.weights[current_batch.end_pos] = edge.prop;
                current_batch.src_distances[current_batch.end_pos] = src_dist;
                current_batch.dsts[current_batch.end_pos] = edge.node_id;
                printf("[LITTLE]Dispatching edge: src_dist=%f, dst=%d, "
                       "weight=%f\n",
                       (float)*reinterpret_cast<distance_t *>(&src_dist),
                       edge.node_id,
                       (float)*reinterpret_cast<distance_t *>(&edge.prop));
                fflush(NULL);
                current_batch.end_pos = current_batch.end_pos + 1;
                if (current_batch.end_pos == PE_NUM) {
                    response_stream.write(current_batch);
                    current_batch.end_pos = 0;
                }
            }
            start_edge_idx = end_edge_idx;
        }
    }
    current_batch.end_flag = true;
    response_stream.write(current_batch);
}

static void node_property_responder(
    hls::stream<node_distance_burst_t> &node_distance_burst_stream,
    int32_t num_nodes, hls::stream<node_dist_batch_t> &all_distances_stream) {
    node_dist_batch_t dist_batch;
    dist_batch.end_flag = false;
    const int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
LOOP_FOR_74:
    for (uint32_t node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         node_burst_idx++) {
#pragma HLS PIPELINE II = 1
        node_distance_burst_t node_distance_burst;
        node_distance_burst = node_distance_burst_stream.read();
        const int32_t base_idx = (node_burst_idx << LOG_PE_NUM);
    LOOP_FOR_73:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            // Direct indexing with pe_idx to avoid race conditions in UNROLL
            dist_batch.data[pe_idx] = node_distance_burst.data[pe_idx];
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
    const int bits_per_output = NODE_ID_BITWIDTH + DISTANCE_BITWIDTH;
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
                packed_output.range(NODE_ID_BITWIDTH - 1, 0) = item.node_id;
                packed_output.range(bits_per_output - 1, NODE_ID_BITWIDTH) =
                    item.prop;
                distance_t tmp_dist =
                    packed_output.range(bits_per_output - 1, NODE_ID_BITWIDTH);
                printf("[LITTLE]Packing output: node_id=%d, distance=%f\n",
                       (int)packed_output.range(NODE_ID_BITWIDTH - 1, 0),
                       (float)tmp_dist);
                fflush(NULL);

                int start_bit = pack_count * bits_per_output;
                write_word.range(start_bit + bits_per_output - 1, start_bit) =
                    packed_output;

                pack_count++;
                if (pack_count == outputs_per_word) {
                    printf(
                        "[LITTLE] Writing packed word to DDR at address %d\n",
                        ddr_addr);
                    fflush(NULL);
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

    // Write the final partial word if it exists
    if (pack_count > 0) {
        printf("[LITTLE] Writing final packed word to DDR at address %d\n",
               ddr_addr);
        fflush(NULL);
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
            ap_fixed<32, 16> lhs_68 = *reinterpret_cast<ap_fixed<32, 16> *>(
                &in_batch_i_global_data_2.data[i]);
            ap_fixed<32, 16> rhs_68 = *reinterpret_cast<ap_fixed<32, 16> *>(
                &in_batch_i_global_data_3.data[i]);
            ap_fixed<32, 16> temp_BinOp_68_o_0_ap_result;
            temp_BinOp_68_o_0_ap_result = (lhs_68 + rhs_68);
            fused_temp_BinOp_68_o_0 =
                *reinterpret_cast<int32_t *>(&temp_BinOp_68_o_0_ap_result);
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

// --- REWRITTEN: High-performance Reduce unit for little kernel using batched
// input and 72-bit packed URAM.
static void
Reduc_105_unit_reduce(hls::stream<struct_kbu_50_t> &in_kt_pair_stream,
                      hls::stream<internal_end_data_batch_t> &o_0) {
    // --- Phase 1: Memory Declaration ---
    // Note: For the little kernel, each PE handles a full keyspace up to
    // MAX_NUM.

    // URAM for packed distance data (3 distances per 72-bit word)
    const int MEM_SIZE =
        (MAX_NUM + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;
    reduce_word_t prop_mem[PE_NUM][MEM_SIZE];
#pragma HLS BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM
#pragma HLS ARRAY_PARTITION variable = prop_mem complete dim = 1
#pragma HLS dependence variable = prop_mem inter false direction = WAW
#pragma HLS dependence variable = prop_mem inter false direction = RAW

    // BRAM for individual validity flags
    bool prop_valid[PE_NUM][MAX_NUM];
#pragma HLS BIND_STORAGE variable = prop_valid type = RAM_2P impl = BRAM
#pragma HLS ARRAY_PARTITION variable = prop_valid complete dim = 1
#pragma HLS dependence variable = prop_valid inter false direction = WAW
#pragma HLS dependence variable = prop_valid inter false direction = RAW

    // BRAM for pre-calculated address mapping
    typedef ap_uint<16> addr_map_t;
    addr_map_t key_to_addr_map[PE_NUM][MAX_NUM];
#pragma HLS BIND_STORAGE variable = key_to_addr_map type = RAM_1P impl = BRAM
#pragma HLS ARRAY_PARTITION variable = key_to_addr_map complete dim = 1

    // Latency-hiding cache for recently accessed URAM words
    reduce_word_t cache_data_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
    reduce_word_t tmp_cache_data_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable = tmp_cache_data_buffer complete dim = 0
    int cache_addr_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0
    int tmp_cache_addr_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable = tmp_cache_addr_buffer complete dim = 0

    printf("[LITTLE]Initialized memory structures for reduction.\n");
    fflush(NULL);

// --- Phase 2: Initialization ---
LOOP_INIT_LITTLE:
    for (int i = 0; i < MAX_NUM; i++) {
#pragma HLS PIPELINE II = 1
        // Populate the address map (shared by all PEs)
        addr_map_t map_val;
        map_val.range(15, 2) = i / DISTANCES_PER_REDUCE_WORD; // word_addr
        map_val.range(1, 0) = i % DISTANCES_PER_REDUCE_WORD;  // pack_idx
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            key_to_addr_map[pe][i] = map_val;
        }
        // Initialize valid flags for all PEs
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
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
    printf("[LITTLE]Completed initialization phase.\n");
    fflush(NULL);

// --- Phase 3: Aggregation Loop ---
LOOP_AGGREGATE_LITTLE:
    while (true) {
#pragma HLS PIPELINE II = 1
        struct_kbu_50_t in_batch;
        printf("[LITTLE]Waiting for input batch...\n");
        fflush(NULL);
        if (in_kt_pair_stream.read_nb(in_batch)) {
            printf("[LITTLE]Processing batch with end_pos=%d, end_flag=%d\n",
                   in_batch.end_pos, in_batch.end_flag);
            fflush(NULL);
            for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
                if (pe < in_batch.end_pos) {
                    printf("[LITTLE]PE %d processing key %d with incoming dist "
                           "%f\n",
                           pe, in_batch.data[pe].key,
                           (float)*reinterpret_cast<distance_t *>(
                               &in_batch.data[pe].transform.prop));
                    fflush(NULL);
                    int key = in_batch.data[pe].key;
                    ap_fixed_pod_t incoming_dist_pod =
                        in_batch.data[pe].transform.prop;

                    addr_map_t map_val = key_to_addr_map[pe][key];
                    int word_addr = map_val.range(15, 2);
                    int pack_idx = map_val.range(1, 0);

                    reduce_word_t current_word = prop_mem[pe][word_addr];

                    // Check cache for forwarding
                    for (int i = L; i >= 0; --i) {
#pragma HLS UNROLL
                        if (cache_addr_buffer[pe][i] == word_addr) {
                            current_word = cache_data_buffer[pe][i];
                            break;
                        }
                    }

                    bool is_valid = prop_valid[pe][key];

                    // Shift cache
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

                    // Perform reduction
                    distance_t new_dist_fp;
                    distance_t incoming_dist_fp =
                        *reinterpret_cast<distance_t *>(&incoming_dist_pod);
                    int start_bit = pack_idx * DISTANCE_BITWIDTH;

                    if (is_valid) {
                        ap_fixed_pod_t old_dist_pod = current_word.range(
                            start_bit + DISTANCE_BITWIDTH - 1, start_bit);
                        distance_t old_dist_fp =
                            *reinterpret_cast<distance_t *>(&old_dist_pod);
                        new_dist_fp = (old_dist_fp < incoming_dist_fp)
                                          ? old_dist_fp
                                          : incoming_dist_fp;
                    } else {
                        new_dist_fp = incoming_dist_fp;
                    }

                    // Update memories and cache
                    prop_valid[pe][key] = true;

                    ap_fixed_pod_t new_dist_pod =
                        *reinterpret_cast<ap_fixed_pod_t *>(&new_dist_fp);
                    current_word.range(start_bit + DISTANCE_BITWIDTH - 1,
                                       start_bit) = new_dist_pod;

                    // Write back to URAM and update cache
                    prop_mem[pe][word_addr] = current_word;
                    cache_addr_buffer[pe][L] = word_addr;
                    cache_data_buffer[pe][L] = current_word;
                }
            }
            if (in_batch.end_flag) {
                break;
            }
        }
        printf("[LITTLE]Finished processing batch.\n");
        fflush(NULL);
    }

    // --- Phase 4: Final Merge and Drain Loop ---
    internal_end_data_batch_t data_pack;
#pragma HLS ARRAY_PARTITION variable = data_pack.data complete dim = 0
    reduce_word_t words[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = words complete dim = 0
    int real_addr = 0;

    printf("[LITTLE]Starting final drain phase.\n");
    fflush(NULL);
LOOP_DRAIN_LITTLE_KEYS:
    for (int addr = 0; addr < MEM_SIZE; addr++) {
#pragma HLS PIPELINE II = 1
        distance_t min_dist;

    LOOP_DRAIN_LITTLE_READ:
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            words[pe] = prop_mem[pe][addr];
        }
    LOOP_DRAIN_PACK_IDX:
        for (int pack_idx = 0; pack_idx < DISTANCES_PER_REDUCE_WORD;
             pack_idx++) {
#pragma HLS UNROLL
            if (real_addr + pack_idx >= MAX_NUM) {
                break; // Avoid processing out-of-bounds keys
            }
            bool valid_found = false;
        LOOP_DRAIN_LITTLE_PES:
            for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
                if (prop_valid[pe][real_addr + pack_idx]) {
                    printf(
                        "[LITTLE]PE %d has valid dist for node %d (%d + %d)\n",
                        pe, real_addr + pack_idx, real_addr, pack_idx);
                    fflush(NULL);
                    int start_bit = pack_idx * DISTANCE_BITWIDTH;
                    ap_fixed_pod_t dist_pod = words[pe].range(
                        start_bit + DISTANCE_BITWIDTH - 1, start_bit);
                    distance_t dist_fp =
                        *reinterpret_cast<distance_t *>(&dist_pod);

                    if (!valid_found || dist_fp < min_dist) {
                        min_dist = dist_fp;
                        valid_found = true;
                    }
                }
            }

            if (valid_found) {
                printf("[LITTLE]Node %d (%d + %d) final min dist: %f\n",
                       real_addr + pack_idx, real_addr, pack_idx,
                       (float)min_dist);
                fflush(NULL);
                data_pack.data[data_pack.end_pos].prop =
                    *reinterpret_cast<ap_fixed_pod_t *>(&min_dist);
                data_pack.data[data_pack.end_pos].node_id =
                    real_addr + pack_idx;
                data_pack.end_pos++;

                if (data_pack.end_pos == PE_NUM) {
                    printf("[LITTLE]Writing out full batch.\n");
                    fflush(NULL);
                    data_pack.end_flag = false;
                    o_0.write(data_pack);
                    data_pack.end_pos = 0;
                }
            }
        }
        real_addr += DISTANCES_PER_REDUCE_WORD;
    }

    if (data_pack.end_pos > 0) {
        printf("[LITTLE]Writing out final partial batch with end_pos=%d.\n",
               data_pack.end_pos);
        fflush(NULL);
        data_pack.end_flag = false;
        o_0.write(data_pack);
    }

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
                      hls::stream<struct_nbu_11_t> &i_0_node_id) {
    // Efficiently filters a stream of all node distances against a stream of
    // requested node IDs.
    struct_nbu_11_t in_node_id_batch;
    node_dist_batch_t in_dist_batch;
    struct_abu_9_t out_dist_batch;
#pragma HLS ARRAY_PARTITION variable = in_node_id_batch.data complete dim = 0
#pragma HLS ARRAY_PARTITION variable = in_dist_batch.data complete dim = 0
#pragma HLS ARRAY_PARTITION variable = out_dist_batch.data complete dim = 0
    out_dist_batch.end_flag = false;
    // Initial reads to prime the pipeline
    in_node_id_batch = i_0_node_id.read();
    in_dist_batch = i_all_node_distances.read();
    uint32_t in_node_base_id = 0;
    uint32_t id_idx = 0;
    uint32_t in_node_end_id;
    in_node_end_id = in_dist_batch.end_pos;
#pragma HLS BIND_STORAGE variable = in_node_end_id type = register impl = srl
LOOP_WHILE_100:
    while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS expression_balance
        uint32_t current_batch_len = in_node_id_batch.end_pos;
        if (((current_batch_len == 0) | (id_idx >= current_batch_len))) {
            if ((id_idx > 0)) {
                out_dist_batch.end_pos = id_idx;
                o_0_node_distance.write(out_dist_batch);
            }
            if (in_node_id_batch.end_flag) {
                break;
            }
            in_node_id_batch = i_0_node_id.read();
            id_idx = 0;
            continue;
        }
        node_id_t target_node_id = in_node_id_batch.data[id_idx];
        if ((target_node_id >= in_node_end_id)) {
            // Target is in a future batch, load the next distance batch
            in_node_base_id = in_node_end_id;
            in_dist_batch = i_all_node_distances.read();
            uint32_t batch_len = in_dist_batch.end_pos;
            in_node_end_id = (in_node_base_id + batch_len);
#pragma HLS BIND_OP variable = in_node_end_id op = add impl = fabric latency = 0
            continue;
        }
        // Target found, calculate index and copy distance
        out_dist_batch.data[id_idx] =
            in_dist_batch.data[target_node_id - in_node_base_id];
        id_idx = (id_idx + 1);
    }
    // Send the final (empty) output batch with the end flag
    struct_abu_9_t final_batch;
    final_batch.end_flag = true;
    final_batch.end_pos = 0;
    o_0_node_distance.write(final_batch);
// Drain any remaining batches from the all_distances stream to prevent deadlock
LOOP_WHILE_101:
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
            ap_fixed<32, 16> lhs_128 =
                *reinterpret_cast<ap_fixed<32, 16> *>(&in_batch_i_0.data[i]);
            ap_fixed<32, 16> rhs_128 =
                *reinterpret_cast<ap_fixed<32, 16> *>(&in_batch_i_1.data[i]);
            ap_fixed<32, 16> temp_BinOp_128_o_0_ap_result;
            temp_BinOp_128_o_0_ap_result =
                (((lhs_128) < (rhs_128) ? lhs_128 : rhs_128));
            fused_temp_BinOp_128_o_0 =
                *reinterpret_cast<int32_t *>(&temp_BinOp_128_o_0_ap_result);
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

// --- 4. Top-level Memory/Dataflow Functions ---
// static void memory_loader(int32_t instantiate_idx, const int32_t*
// src_offsets, const edge_des_burst_t* edge_des_bursts, const int32_t*
// node_distances, int32_t num_nodes, int32_t num_edges,
// hls::stream<edge_batch_t> &response_to_318, hls::stream<node_dist_batch_t>
// &all_node_distances_to_343) { #pragma HLS function_instantiate
// variable=instantiate_idx #pragma HLS DATAFLOW
//     hls::stream<node_distance_burst_t> node_distance_burst_stream_0;
// #pragma HLS STREAM variable=node_distance_burst_stream_0 depth=12
//     hls::stream<node_distance_burst_t> node_distance_burst_stream_1;
// #pragma HLS STREAM variable=node_distance_burst_stream_1 depth=12
//     hls::stream<edge_descriptor_batch_t> edge_stream;
// #pragma HLS STREAM variable=edge_stream depth=12
//     hls::stream<int32_t> src_offsets_cache_stream;
// #pragma HLS STREAM variable=src_offsets_cache_stream depth=32
//     src_offset_loader(src_offsets, src_offsets_cache_stream, num_nodes);
//     node_property_loader(node_distances, node_distance_burst_stream_0,
//     node_distance_burst_stream_1, num_nodes);
//     edge_descriptor_loader(edge_des_bursts, edge_stream, num_edges);
//     edge_property_loader_and_dispatcher(src_offsets_cache_stream,
//     edge_stream, node_distance_burst_stream_0, num_nodes, response_to_318);
//     node_property_responder(node_distance_burst_stream_1, num_nodes,
//     all_node_distances_to_343);
// }

static void graphyflow_little_dataflow(
    hls::stream<edge_batch_t> &response_to_318,
    hls::stream<node_dist_batch_t> &all_node_distances_to_343,
    hls::stream<internal_end_data_batch_t> &internal_end_stream) {
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
    hls::stream<struct_nbu_11_t> stream_o_1_309;
#pragma HLS STREAM variable = stream_o_1_309 depth = 4
    hls::stream<struct_abu_9_t> stream_o_0_304;
#pragma HLS STREAM variable = stream_o_0_304 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_1_305;
#pragma HLS STREAM variable = stream_o_1_305 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_0_308;
#pragma HLS STREAM variable = stream_o_0_308 depth = 4
    // --- Function Calls (in topological order) ---
    Memor_274(response_to_318, stream_o_0_edge_src_distance_275,
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
    printf("[LITTLE]Entering reduction phase.\n");
    fflush(NULL);
    Reduc_105_unit_reduce(reduce_105_z2u_pair, stream_o_0_107);
    printf("[LITTLE]Reduction phase complete.\n");
    fflush(NULL);
    // --- End of Reduce Super-Block for Reduc_105 ---
    Scatt_302(stream_o_0_107, stream_o_0_304, stream_o_1_305);
    CopyC_306(stream_o_1_305, stream_o_0_308, stream_o_1_309);
    Memor_299(all_node_distances_to_343, stream_o_0_node_distance_300,
              stream_o_1_309);
    fused_op_294(stream_o_0_304, stream_o_0_node_distance_300, stream_o_0_308,
                 internal_end_stream);
}

// static void final_writeback(int32_t instantiate_idx,
// hls::stream<internal_end_data_batch_t> &internal_end_stream,
// KernelOutputBatch* out_o_0_342) { #pragma HLS function_instantiate
// variable=instantiate_idx #pragma HLS DATAFLOW
//     hls::stream<KernelOutputBatch> converted_stream;
// #pragma HLS STREAM variable=converted_stream depth=12
//     final_convert(internal_end_stream, converted_stream);
//     final_write(converted_stream, out_o_0_342);
// }

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void graphyflow_little(const bus_word_t *src_offsets,
                                  const bus_word_t *edge_props,
                                  const bus_word_t *node_props,
                                  bus_word_t *output, int32_t num_nodes,
                                  int32_t num_edges) {
#pragma HLS INTERFACE m_axi port = src_offsets offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = src_offsets
#pragma HLS INTERFACE s_axilite port = edge_props
#pragma HLS INTERFACE s_axilite port = node_props
#pragma HLS INTERFACE s_axilite port = output
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW

    hls::stream<node_distance_burst_t> node_distance_burst_stream_0;
#pragma HLS STREAM variable = node_distance_burst_stream_0 depth = 128
    hls::stream<node_distance_burst_t> node_distance_burst_stream_1;
#pragma HLS STREAM variable = node_distance_burst_stream_1 depth = 128

    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 128

    hls::stream<int32_t> src_offsets_cache_stream;
#pragma HLS STREAM variable = src_offsets_cache_stream depth = 256

    hls::stream<edge_batch_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 128

    hls::stream<node_dist_batch_t> stream_node_dist_data;
#pragma HLS STREAM variable = stream_node_dist_data depth = 128

    hls::stream<internal_end_data_batch_t> stream_result_data;
#pragma HLS STREAM variable = stream_result_data depth = 128
    printf("[LITTLE]GraphyFlow-Little HLS kernel started.\n");
    fflush(NULL);
    src_offset_loader(src_offsets, src_offsets_cache_stream, num_nodes);
    node_property_loader(node_props, node_distance_burst_stream_0,
                         node_distance_burst_stream_1, num_nodes);
    edge_descriptor_loader(edge_props, edge_stream, num_edges);
    printf("[LITTLE]Data loading complete, entering main processing loop.\n");
    fflush(NULL);

    edge_property_loader_and_dispatcher(src_offsets_cache_stream, edge_stream,
                                        node_distance_burst_stream_0, num_nodes,
                                        stream_edge_data);
    printf("[LITTLE]Data loading-1 complete.\n");
    fflush(NULL);
    node_property_responder(node_distance_burst_stream_1, num_nodes,
                            stream_node_dist_data);
    printf("[LITTLE]Data loading-2 complete, entering main processing loop.\n");
    fflush(NULL);

    graphyflow_little_dataflow(stream_edge_data, stream_node_dist_data,
                               stream_result_data);
    printf("[LITTLE]Main processing loop complete, entering final writeback "
           "stage.\n");
    fflush(NULL);

    final_writeback(stream_result_data, output);
    printf("[LITTLE]GraphyFlow-Little HLS kernel completed.\n");
    fflush(NULL);
}

#include "graphyflow_big.h"

// --- 1. Memory Helper Functions ---
static void node_property_loader(const int32_t* node_distances_ddr, hls::stream<node_distance_burst_t> &node_distance_burst_stream_0, hls::stream<node_distance_burst_t> &node_distance_burst_stream_1, int32_t num_nodes) {
    const ap_uint<AXI_BUS_WIDTH> *wide_bus_ptr = reinterpret_cast<const ap_uint<AXI_BUS_WIDTH> *>(node_distances_ddr);
    int32_t num_wide_reads = (num_nodes + NUM_WORDS_PER_BUS - 1) / NUM_WORDS_PER_BUS;
    int32_t sent_pack_cnt = 0;
    int32_t total_pack_cnt = (num_nodes + PE_NUM - 1) / PE_NUM;
    sent_pack_cnt = 0;
    LOOP_FOR_2:
    for (uint32_t i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<AXI_BUS_WIDTH> wide_word;
        wide_word = wide_bus_ptr[i];
        node_distance_burst_t burst;
        LOOP_FOR_1:
        for (uint32_t j = 0; j < NUM_WORDS_PER_BUS; j += PE_NUM) {
#pragma HLS UNROLL
            LOOP_FOR_0:
            for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
                if (i * NUM_WORDS_PER_BUS + j + pe < num_nodes) {
                    burst.data[pe] = wide_word.range((j + pe + 1) * DATA_TYPE_WIDTH - 1, (j + pe) * DATA_TYPE_WIDTH);
                }
            }
            if (sent_pack_cnt < total_pack_cnt) {
                node_distance_burst_stream_0.write(burst);
                sent_pack_cnt = sent_pack_cnt + 1;
            }
        }
    }
    sent_pack_cnt = 0;
    LOOP_FOR_5:
    for (uint32_t i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<AXI_BUS_WIDTH> wide_word;
        wide_word = wide_bus_ptr[i];
        node_distance_burst_t burst;
        LOOP_FOR_4:
        for (uint32_t j = 0; j < NUM_WORDS_PER_BUS; j += PE_NUM) {
#pragma HLS UNROLL
            LOOP_FOR_3:
            for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
                if (i * NUM_WORDS_PER_BUS + j + pe < num_nodes) {
                    burst.data[pe] = wide_word.range((j + pe + 1) * DATA_TYPE_WIDTH - 1, (j + pe) * DATA_TYPE_WIDTH);
                }
            }
            if (sent_pack_cnt < total_pack_cnt) {
                node_distance_burst_stream_1.write(burst);
                sent_pack_cnt = sent_pack_cnt + 1;
            }
        }
    }
}

static void edge_descriptor_loader(const edge_des_burst_t* edge_des_bursts, hls::stream<edge_descriptor_batch_t> &edge_stream, int32_t num_edges) {
#pragma HLS dependence variable=edge_des_bursts inter false
    const int32_t num_batches = (num_edges + PE_NUM - 1) / PE_NUM;
    LOOP_FOR_7:
    for (uint32_t i = 0; i < num_batches; i++) {
#pragma HLS PIPELINE II=1
        edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable=edge_batch.edges complete dim=0
#pragma HLS dependence variable = edge_batch inter false direction = WAW
        edge_des_burst_t burst;
        burst = edge_des_bursts[i];
        edge_batch.end_pos = 0;
        LOOP_FOR_6:
        for (uint32_t j = 0; j < PE_NUM; j++) {
#pragma HLS UNROLL
            if (i * PE_NUM + j < num_edges) {
                edge_batch.edges[edge_batch.end_pos] = burst.edges[j];
                edge_batch.end_pos = edge_batch.end_pos + 1;
            }
        }
        edge_stream.write(edge_batch);
    }
}

static void src_offset_loader(const int32_t* src_offsets_ddr, hls::stream<int32_t> &src_offsets_stream, int32_t num_nodes) {
    const ap_uint<AXI_BUS_WIDTH> *wide_bus_ptr = reinterpret_cast<const ap_uint<AXI_BUS_WIDTH> *>(src_offsets_ddr);
#pragma HLS dependence variable=wide_bus_ptr inter false
    const int32_t num_wide_reads = (num_nodes + 1 + NUM_WORDS_PER_BUS - 1) / NUM_WORDS_PER_BUS;
    LOOP_FOR_9:
    for (uint32_t i = 0; i <= num_wide_reads; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<AXI_BUS_WIDTH> wide_word;
        wide_word = wide_bus_ptr[i];
        LOOP_FOR_8:
        for (uint32_t j = 0; j < NUM_WORDS_PER_BUS; j++) {
#pragma HLS UNROLL
            if (i * NUM_WORDS_PER_BUS + j <= num_nodes) {
                int32_t cur_data = wide_word.range((j + 1) * DATA_TYPE_WIDTH - 1, j * DATA_TYPE_WIDTH);
                src_offsets_stream.write(cur_data);
            }
        }
    }
}

static void edge_property_loader_and_dispatcher(hls::stream<int32_t> &src_offsets_cache_stream, hls::stream<edge_descriptor_batch_t> &edge_stream, hls::stream<node_distance_burst_t> &node_distance_burst_stream, int32_t num_nodes, hls::stream<edge_batch_t> &response_stream) {
    edge_batch_t current_batch;
#pragma HLS ARRAY_PARTITION variable=current_batch.weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=current_batch.src_distances complete dim=0
#pragma HLS ARRAY_PARTITION variable=current_batch.dsts complete dim=0
#pragma HLS dependence variable=current_batch inter false direction=WAW
    current_batch.end_pos = 0;
    current_batch.end_flag = false;
    edge_descriptor_batch_t edge_batch;
    edge_batch.end_pos = 0;
#pragma HLS ARRAY_PARTITION variable=edge_batch.edges complete dim=0
#pragma HLS dependence variable=edge_batch inter false direction=WAW
    int32_t edge_batch_pos = 0;
    node_distance_burst_t node_distance_burst;
#pragma HLS ARRAY_PARTITION variable=node_distance_burst.data complete dim=0
#pragma HLS dependence variable=node_distance_burst inter false
    int32_t start_edge_idx;
    int32_t end_edge_idx;
    start_edge_idx = src_offsets_cache_stream.read();
    const int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
    LOOP_FOR_12:
    for (uint32_t node_burst_idx = 0; node_burst_idx < max_node_burst_idx; node_burst_idx++) {
        node_distance_burst = node_distance_burst_stream.read();
        const int32_t base_idx = (node_burst_idx << LOG_PE_NUM);
        LOOP_FOR_11:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
            if (base_idx + pe_idx >= num_nodes) {
                break;
            }
            int32_t src_dist;
            src_dist = node_distance_burst.data[pe_idx];
            end_edge_idx = src_offsets_cache_stream.read();
            LOOP_FOR_10:
            for (uint32_t e_idx = start_edge_idx; e_idx < end_edge_idx; e_idx++) {
#pragma HLS PIPELINE II=1
                if (edge_batch_pos == edge_batch.end_pos) {
                    edge_batch = edge_stream.read();
                    edge_batch_pos = 0;
                }
                node_with_prop_t edge;
                edge = edge_batch.edges[edge_batch_pos++];
                current_batch.weights[current_batch.end_pos] = edge.prop;
                current_batch.src_distances[current_batch.end_pos] = src_dist;
                current_batch.dsts[current_batch.end_pos] = edge.node_id;
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

static void node_property_responder(hls::stream<node_distance_burst_t> &node_distance_burst_stream, int32_t num_nodes, hls::stream<node_dist_batch_t> &all_distances_stream) {
    node_dist_batch_t dist_batch;
    dist_batch.end_flag = false;
    const int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
    LOOP_FOR_14:
    for (uint32_t node_burst_idx = 0; node_burst_idx < max_node_burst_idx; node_burst_idx++) {
#pragma HLS PIPELINE II=1
        node_distance_burst_t node_distance_burst;
        node_distance_burst = node_distance_burst_stream.read();
        const int32_t base_idx = (node_burst_idx << LOG_PE_NUM);
        LOOP_FOR_13:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            // Direct indexing with pe_idx to avoid race conditions in UNROLL
            dist_batch.data[pe_idx] = node_distance_burst.data[pe_idx];
        }
        // Calculate end_pos as the number of valid nodes in this burst
        const int32_t remaining_nodes = num_nodes - base_idx;
        dist_batch.end_pos = (remaining_nodes < PE_NUM) ? remaining_nodes : PE_NUM;
        all_distances_stream.write(dist_batch);
    }
    dist_batch.end_flag = true;
    dist_batch.end_pos = 0;
    all_distances_stream.write(dist_batch);
}

static void final_convert(hls::stream<internal_end_data_batch_t> &in_stream, hls::stream<KernelOutputBatch> &converted_stream) {
    LOOP_WHILE_16:
    while (true) {
#pragma HLS PIPELINE II=1
        internal_end_data_batch_t in_batch;
        in_batch = in_stream.read();
        KernelOutputBatch out_batch;
        out_batch.end_flag = in_batch.end_flag;
        out_batch.end_pos = in_batch.end_pos;
        LOOP_FOR_15:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            ap_fixed<32, 16> dist_fp = *reinterpret_cast<ap_fixed<32, 16> *>(&in_batch.data[i].prop);
            out_batch.data[i].distance = (float)dist_fp;
            out_batch.data[i].id = in_batch.data[i].node_id;
        }
        converted_stream.write(out_batch);
        if (in_batch.end_flag) {
            break;
        }
    }
}

static void final_write(hls::stream<KernelOutputBatch> &converted_stream, KernelOutputBatch* out_o_0_342) {
#pragma HLS dependence variable=out_o_0_342 inter false
    int32_t i = 0;
    LOOP_WHILE_17:
    while (true) {
#pragma HLS PIPELINE II=1
        KernelOutputBatch out_batch;
        out_batch = converted_stream.read();
        out_o_0_342[i] = out_batch;
        if (out_batch.end_flag) {
            break;
        }
        i = (i + 1);
    }
}

// --- 2. Utility Network Functions ---
static void stream_zipper_0(hls::stream<struct_ibu_14_t> &in_key_batch_stream, hls::stream<internal_end_data_batch_t> &in_transform_batch_stream, hls::stream<struct_kbu_50_t> &out_pair_batch_stream) {
    struct_ibu_14_t key_batch;
    internal_end_data_batch_t transform_batch;
    struct_kbu_50_t out_batch;
    LOOP_WHILE_19:
    while (true) {
#pragma HLS PIPELINE
        key_batch = in_key_batch_stream.read();
        transform_batch = in_transform_batch_stream.read();
        LOOP_FOR_18:
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

static void demux_1(hls::stream<struct_kbu_50_t> &in_batch_stream, hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8]) {
    struct_kbu_50_t in_batch;
    LOOP_WHILE_22:
    while (true) {
#pragma HLS PIPELINE
        in_batch = in_batch_stream.read();
        net_wrapper_kt_pair_105_t_t wrapper_data;
        LOOP_FOR_20:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if ((i < in_batch.end_pos)) {
                wrapper_data.data = in_batch.data[i];
                wrapper_data.end_flag = false;
                out_streams[i].write(wrapper_data);
            }
        }
        if (in_batch.end_flag) {
            break;
        }
    }
    // Propagate end_flag to all output streams
    net_wrapper_kt_pair_105_t_t end_wrapper;
    end_wrapper.end_flag = true;
    LOOP_FOR_21:
    for (uint32_t i = 0; i < 8; i++) {
#pragma HLS UNROLL
        out_streams[i].write(end_wrapper);
    }
}

static void sender_2(int32_t i, hls::stream<net_wrapper_kt_pair_105_t_t> &in1, hls::stream<net_wrapper_kt_pair_105_t_t> &in2, hls::stream<net_wrapper_kt_pair_105_t_t> &out1, hls::stream<net_wrapper_kt_pair_105_t_t> &out2, hls::stream<net_wrapper_kt_pair_105_t_t> &out3, hls::stream<net_wrapper_kt_pair_105_t_t> &out4) {
#pragma HLS function_instantiate variable=i
    bool in1_end_flag = false;
    bool in2_end_flag = false;
    LOOP_WHILE_23:
    while (true) {
#pragma HLS PIPELINE II=1
        if ((!in1.empty())) {
            net_wrapper_kt_pair_105_t_t data1;
            data1 = in1.read();
            if ((!data1.end_flag)) {
                if (((data1.data.key >> i) & 1)) {
                    out2.write(data1);
                } else {
                    out1.write(data1);
                }
            } else {
                in1_end_flag = true;
            }
        }
        if ((!in2.empty())) {
            net_wrapper_kt_pair_105_t_t data2;
            data2 = in2.read();
            if ((!data2.end_flag)) {
                if (((data2.data.key >> i) & 1)) {
                    out4.write(data2);
                } else {
                    out3.write(data2);
                }
            } else {
                in2_end_flag = true;
            }
        }
        if ((in1_end_flag & in2_end_flag)) {
            net_wrapper_kt_pair_105_t_t data;
            data.end_flag = true;
            out1.write(data);
            out2.write(data);
            out3.write(data);
            out4.write(data);
            in1_end_flag = false;
            in2_end_flag = false;
            break;
        }
    }
}

static void receiver_2(int32_t i, hls::stream<net_wrapper_kt_pair_105_t_t> &out1, hls::stream<net_wrapper_kt_pair_105_t_t> &out2, hls::stream<net_wrapper_kt_pair_105_t_t> &in1, hls::stream<net_wrapper_kt_pair_105_t_t> &in2, hls::stream<net_wrapper_kt_pair_105_t_t> &in3, hls::stream<net_wrapper_kt_pair_105_t_t> &in4) {
#pragma HLS function_instantiate variable=i
    bool in1_end_flag = false;
    bool in2_end_flag = false;
    bool in3_end_flag = false;
    bool in4_end_flag = false;
    LOOP_WHILE_24:
    while (true) {
#pragma HLS PIPELINE II=1
        if ((!in1.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in1.read();
            if ((!data.end_flag)) {
                out1.write(data);
            } else {
                in1_end_flag = true;
            }
        } else if ((!in3.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in3.read();
            if ((!data.end_flag)) {
                out1.write(data);
            } else {
                in3_end_flag = true;
            }
        }
        if ((!in2.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in2.read();
            if ((!data.end_flag)) {
                out2.write(data);
            } else {
                in2_end_flag = true;
            }
        } else if ((!in4.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in4.read();
            if ((!data.end_flag)) {
                out2.write(data);
            } else {
                in4_end_flag = true;
            }
        }
        if ((((in1_end_flag & in2_end_flag) & in3_end_flag) & in4_end_flag)) {
            net_wrapper_kt_pair_105_t_t data;
            data.end_flag = true;
            out1.write(data);
            out2.write(data);
            break;
        }
    }
}

static void switch2x2_2(int32_t i, hls::stream<net_wrapper_kt_pair_105_t_t> &in1, hls::stream<net_wrapper_kt_pair_105_t_t> &in2, hls::stream<net_wrapper_kt_pair_105_t_t> &out1, hls::stream<net_wrapper_kt_pair_105_t_t> &out2) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_1;
#pragma HLS STREAM variable=l1_1 depth=2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_2;
#pragma HLS STREAM variable=l1_2 depth=2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_3;
#pragma HLS STREAM variable=l1_3 depth=2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_4;
#pragma HLS STREAM variable=l1_4 depth=2
    sender_2(i, in1, in2, l1_1, l1_2, l1_3, l1_4);
    receiver_2(i, out1, out2, l1_1, l1_2, l1_3, l1_4);
}

static void omega_switch_2(hls::stream<net_wrapper_kt_pair_105_t_t> (&in_streams)[8], hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8]) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_0[8];
#pragma HLS STREAM variable=stream_stage_0 depth=2
    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_1[8];
#pragma HLS STREAM variable=stream_stage_1 depth=2
    switch2x2_2(2, in_streams[0], in_streams[1], stream_stage_0[0], stream_stage_0[1]);
    switch2x2_2(2, in_streams[2], in_streams[3], stream_stage_0[2], stream_stage_0[3]);
    switch2x2_2(2, in_streams[4], in_streams[5], stream_stage_0[4], stream_stage_0[5]);
    switch2x2_2(2, in_streams[6], in_streams[7], stream_stage_0[6], stream_stage_0[7]);
    switch2x2_2(1, stream_stage_0[0], stream_stage_0[4], stream_stage_1[0], stream_stage_1[1]);
    switch2x2_2(1, stream_stage_0[1], stream_stage_0[5], stream_stage_1[2], stream_stage_1[3]);
    switch2x2_2(1, stream_stage_0[2], stream_stage_0[6], stream_stage_1[4], stream_stage_1[5]);
    switch2x2_2(1, stream_stage_0[3], stream_stage_0[7], stream_stage_1[6], stream_stage_1[7]);
    switch2x2_2(0, stream_stage_1[0], stream_stage_1[4], out_streams[0], out_streams[1]);
    switch2x2_2(0, stream_stage_1[1], stream_stage_1[5], out_streams[2], out_streams[3]);
    switch2x2_2(0, stream_stage_1[2], stream_stage_1[6], out_streams[4], out_streams[5]);
    switch2x2_2(0, stream_stage_1[3], stream_stage_1[7], out_streams[6], out_streams[7]);
}

// --- 3. DFIR Component Functions ---
static void Reduc_105_pre_process(hls::stream<struct_ibu_14_t> &i_global_data_0, hls::stream<struct_nbu_11_t> &i_global_data_1, hls::stream<struct_abu_9_t> &i_global_data_2, hls::stream<struct_abu_9_t> &i_global_data_3, hls::stream<struct_ibu_14_t> &intermediate_key, hls::stream<internal_end_data_batch_t> &intermediate_transform) {
    struct_ibu_14_t in_batch_i_global_data_0;
    struct_nbu_11_t in_batch_i_global_data_1;
    struct_abu_9_t in_batch_i_global_data_2;
    struct_abu_9_t in_batch_i_global_data_3;
    struct_ibu_14_t out_batch_intermediate_key;
    internal_end_data_batch_t out_batch_intermediate_transform;
    bool end_flag;
    LOOP_WHILE_26:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_global_data_0 = i_global_data_0.read();
        in_batch_i_global_data_1 = i_global_data_1.read();
        in_batch_i_global_data_2 = i_global_data_2.read();
        in_batch_i_global_data_3 = i_global_data_3.read();
        LOOP_FOR_25:
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
            ap_fixed<32, 16> lhs_68 = *reinterpret_cast<ap_fixed<32, 16>*>(&in_batch_i_global_data_2.data[i]);
            ap_fixed<32, 16> rhs_68 = *reinterpret_cast<ap_fixed<32, 16>*>(&in_batch_i_global_data_3.data[i]);
            ap_fixed<32, 16> temp_BinOp_68_o_0_ap_result;
            temp_BinOp_68_o_0_ap_result = (lhs_68 + rhs_68);
            fused_temp_BinOp_68_o_0 = *reinterpret_cast<int32_t*>(&temp_BinOp_68_o_0_ap_result);
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
        out_batch_intermediate_transform.end_flag = in_batch_i_global_data_0.end_flag;
        out_batch_intermediate_transform.end_pos = in_batch_i_global_data_0.end_pos;
        intermediate_key.write(out_batch_intermediate_key);
        intermediate_transform.write(out_batch_intermediate_transform);
        end_flag = in_batch_i_global_data_0.end_flag;
        if (end_flag) {
            break;
        }
    }
}

static void Reduc_105_unit_reduce(hls::stream<net_wrapper_kt_pair_105_t_t> (&kt_wrap_item)[PE_NUM], hls::stream<internal_end_data_batch_t> &o_0) {
    // **KEY OPTIMIZATION**: Split struct into separate arrays
    // Valid flags use BRAM (lower latency)
    bool key_mem_valid[PE_NUM][MAX_NUM >> LOG_PE_NUM];
#pragma HLS dependence variable=key_mem_valid inter false direction=WAW
#pragma HLS dependence variable=key_mem_valid inter false direction=RAW
#pragma HLS BIND_STORAGE variable=key_mem_valid type=RAM_2P impl=BRAM latency=1
#pragma HLS ARRAY_PARTITION variable=key_mem_valid complete dim=1
    // Data uses URAM (large capacity)
    node_with_prop_t key_mem_data[PE_NUM][MAX_NUM >> LOG_PE_NUM];
#pragma HLS dependence variable=key_mem_data inter false direction=WAW
#pragma HLS dependence variable=key_mem_data inter false direction=RAW
#pragma HLS BIND_STORAGE variable=key_mem_data type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=key_mem_data complete dim=1
    // Buffers for latency hiding and data forwarding
    struct_nb_58_t key_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable=key_buffer complete dim=0
    struct_nb_58_t tmp_key_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable=tmp_key_buffer complete dim=0
    uint32_t i_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable=i_buffer complete dim=0
    uint32_t tmp_i_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable=tmp_i_buffer complete dim=0
    // 2. Memory initialization
    LOOP_FOR_28:
    for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
        LOOP_FOR_27:
        for (uint32_t i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
            i_buffer[pe][i] = (MAX_NUM + 1);
        }
    }
    // Initialize separated memories
    LOOP_FOR_30:
    for (uint32_t pe = 0; pe < PE_NUM; pe++) {
        LOOP_FOR_29:
        for (uint32_t i = 0; i < (MAX_NUM >> LOG_PE_NUM); i++) {
#pragma HLS PIPELINE II=1
            key_mem_valid[pe][i] = false;
            key_mem_data[pe][i].prop = 0;
            key_mem_data[pe][i].node_id = 0;
        }
    }
    // 3. Main processing loop for aggregation across PEs
    bool end_flag;
    bool all_end_flags[PE_NUM];
#pragma HLS ARRAY_PARTITION variable=all_end_flags complete dim=0
    LOOP_FOR_31:
    for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
        all_end_flags[i] = false;
    }
    LOOP_WHILE_37:
    while (true) {
#pragma HLS PIPELINE
        net_wrapper_kt_pair_105_t_t kt_elem;
        int32_t key_elem;
        node_with_prop_t transform_elem;
        LOOP_FOR_35:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if (((!all_end_flags[i]) & (!kt_wrap_item[i].empty()))) {
                kt_elem = kt_wrap_item[i].read();
                if (kt_elem.end_flag) {
                    all_end_flags[i] = kt_elem.end_flag;
                } else {
                    key_elem = kt_elem.data.key;
                    transform_elem = kt_elem.data.transform;
                    // Apply key partitioning
                    key_elem = (key_elem >> LOG_PE_NUM);
                    // **OPTIMIZED**: Separate reads from BRAM and URAM
                    bool old_valid;
                    node_with_prop_t old_data;
                    old_valid = key_mem_valid[i][key_elem];
                    old_data = key_mem_data[i][key_elem];
                    LOOP_FOR_32:
                    for (int32_t i_search = L; i_search >= 0; i_search--) {
#pragma HLS UNROLL
                        if ((key_elem == i_buffer[i][i_search])) {
                            old_valid = key_buffer[i][i_search].ele_1;
                            old_data = key_buffer[i][i_search].ele_0;
                            break;
                        }
                    }
                    LOOP_FOR_33:
                    for (uint32_t i_move = 0; i_move < L; i_move++) {
#pragma HLS UNROLL
                        tmp_i_buffer[i][i_move] = i_buffer[i][i_move + 1];
                        tmp_key_buffer[i][i_move] = key_buffer[i][i_move + 1];
                    }
                    LOOP_FOR_34:
                    for (uint32_t i_update = 0; i_update < L; i_update++) {
#pragma HLS UNROLL
                        i_buffer[i][i_update] = tmp_i_buffer[i][i_update];
                        key_buffer[i][i_update] = tmp_key_buffer[i][i_update];
                    }
                    // **OPTIMIZED**: Compute new value
                    node_with_prop_t new_data;
                    bool new_valid;
                    new_valid = true;
                    if (old_valid) {
                        // -- Inline sub graph --
                        // Inlining Scatt_220
                        ap_fixed_pod_t temp_Scatt_220_o_0;
                        node_id_t temp_Scatt_220_o_1;
                        temp_Scatt_220_o_0 = old_data.prop;
                        temp_Scatt_220_o_1 = old_data.node_id;
                        // Inlining Scatt_224
                        ap_fixed_pod_t temp_Scatt_224_o_0;
                        node_id_t temp_Scatt_224_o_1;
                        temp_Scatt_224_o_0 = transform_elem.prop;
                        // Inlining fused_op_214
                        // -- Begin Nested Inline for FusedOp fused_op_214 --
                        // Inlining BinOp_96
                        ap_fixed_pod_t fused_temp_BinOp_96_o_0;
                        ap_fixed<32, 16> lhs_96 = *reinterpret_cast<ap_fixed<32, 16>*>(&temp_Scatt_220_o_0);
                        ap_fixed<32, 16> rhs_96 = *reinterpret_cast<ap_fixed<32, 16>*>(&temp_Scatt_224_o_0);
                        ap_fixed<32, 16> temp_BinOp_96_o_0_ap_result;
                        temp_BinOp_96_o_0_ap_result = (((lhs_96) < (rhs_96) ? lhs_96 : rhs_96));
                        fused_temp_BinOp_96_o_0 = *reinterpret_cast<int32_t*>(&temp_BinOp_96_o_0_ap_result);
                        // Inlining Gathe_208
                        new_data.prop = fused_temp_BinOp_96_o_0;
                        new_data.node_id = temp_Scatt_220_o_1;
                        // -- End Nested Inline for FusedOp fused_op_214 --
                        // -- Inline sub graph end --
                    } else {
                        new_data = transform_elem;
                    }
                    // **OPTIMIZED**: Separate writes to BRAM and URAM
                    key_mem_valid[i][key_elem] = new_valid;
                    key_mem_data[i][key_elem] = new_data;
                    // Update buffer
                    key_buffer[i][L].ele_1 = new_valid;
                    key_buffer[i][L].ele_0 = new_data;
                    i_buffer[i][L] = key_elem;
                }
            }
        }
        end_flag = true;
        LOOP_FOR_36:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            end_flag = (end_flag & all_end_flags[i]);
        }
        if (end_flag) {
            break;
        }
    }
    // 4. Final drain loop (using optimized parallel drain logic)
    internal_end_data_batch_t data_pack;
#pragma HLS ARRAY_PARTITION variable=data_pack.data complete dim=0
    uint32_t write_positions[PE_NUM];
#pragma HLS ARRAY_PARTITION variable=write_positions complete dim=0
    node_with_prop_t tmp_data[PE_NUM];
#pragma HLS ARRAY_PARTITION variable=tmp_data complete dim=0
    bool tmp_data_valid[PE_NUM];
#pragma HLS ARRAY_PARTITION variable=tmp_data_valid complete dim=0
    data_pack.end_flag = false;
    uint32_t k = 0;
    k = 0;
    LOOP_WHILE_41:
    while ((k < (MAX_NUM >> LOG_PE_NUM))) {
#pragma HLS PIPELINE II=1
        LOOP_FOR_38:
        for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            tmp_data[pe] = key_mem_data[pe][k];
            tmp_data_valid[pe] = key_mem_valid[pe][k];
        }
        // Parallel prefix sum to find write positions for valid data
        uint32_t prefix_sum = 0;
        prefix_sum = 0;
        LOOP_FOR_39:
        for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            write_positions[pe] = prefix_sum;
            prefix_sum = (prefix_sum + (tmp_data_valid[pe] ? 1 : 0));
        }
        uint32_t data_cnt;
        data_cnt = prefix_sum;
        if ((data_cnt == 0)) {
            k = (k + 1);
            continue;
        }
        // Parallel write to pack the data
        LOOP_FOR_40:
        for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            if (tmp_data_valid[pe]) {
                data_pack.data[write_positions[pe]] = tmp_data[pe];
            }
        }
        k = (k + 1);
        data_pack.end_pos = data_cnt;
        o_0.write(data_pack);
    }
    // 5. Final batch to signal end of stream
    data_pack.end_flag = true;
    data_pack.end_pos = 0;
    o_0.write(data_pack);
}

static void Scatt_234(hls::stream<struct_sbu_7_t> &i_0, hls::stream<struct_abu_9_t> &o_0, hls::stream<struct_nbu_11_t> &o_1, hls::stream<struct_abu_9_t> &o_2) {
    struct_sbu_7_t in_batch_i_0;
    struct_abu_9_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    struct_abu_9_t out_batch_o_2;
    bool end_flag;
    uint8_t end_pos;
    LOOP_WHILE_43:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        LOOP_FOR_42:
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

static void Memor_231(hls::stream<struct_ibu_14_t> &o_0_node_id, hls::stream<struct_nbu_11_t> &i_0_node_id) {
    struct_nbu_11_t in_batch_i_0_node_id;
    struct_ibu_14_t out_batch_o_0_node_id;
    bool end_flag;
    uint8_t end_pos;
    LOOP_WHILE_45:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_node_id = i_0_node_id.read();
        LOOP_FOR_44:
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

static void CopyC_247(hls::stream<struct_nbu_11_t> &i_0, hls::stream<struct_nbu_11_t> &o_0, hls::stream<struct_nbu_11_t> &o_1) {
    struct_nbu_11_t in_batch_i_0;
    struct_nbu_11_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
    LOOP_WHILE_47:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        LOOP_FOR_46:
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

static void fused_op_269(hls::stream<struct_abu_9_t> &i_0, hls::stream<struct_nbu_11_t> &i_1, hls::stream<struct_abu_9_t> &i_2, hls::stream<struct_sbu_7_t> &o_0) {
    struct_abu_9_t in_batch_i_0;
    struct_nbu_11_t in_batch_i_1;
    struct_abu_9_t in_batch_i_2;
    struct_sbu_7_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    LOOP_WHILE_49:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
        LOOP_FOR_48:
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

static void Memor_274(hls::stream<edge_batch_t> &i_0_edge_id, hls::stream<struct_abu_9_t> &o_0_edge_src_distance, hls::stream<struct_nbu_11_t> &o_0_edge_dst, hls::stream<struct_abu_9_t> &o_0_edge_weight) {
    edge_batch_t in_batch_i_0_edge_id;
    struct_abu_9_t out_batch_o_0_edge_src_distance;
    struct_nbu_11_t out_batch_o_0_edge_dst;
    struct_abu_9_t out_batch_o_0_edge_weight;
    bool end_flag;
    uint8_t end_pos;
    LOOP_WHILE_51:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_edge_id = i_0_edge_id.read();
        LOOP_FOR_50:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0_edge_src_distance.data[i] = in_batch_i_0_edge_id.src_distances[i];
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

static void Memor_299(hls::stream<node_dist_batch_t> &i_all_node_distances, hls::stream<struct_abu_9_t> &o_0_node_distance, hls::stream<struct_nbu_11_t> &i_0_node_id) {
    // Efficiently filters a stream of all node distances against a stream of requested node IDs.
    struct_nbu_11_t in_node_id_batch;
    node_dist_batch_t in_dist_batch;
    struct_abu_9_t out_dist_batch;
#pragma HLS ARRAY_PARTITION variable=in_node_id_batch.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=in_dist_batch.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=out_dist_batch.data complete dim=0
    out_dist_batch.end_flag = false;
    // Initial reads to prime the pipeline
    in_node_id_batch = i_0_node_id.read();
    in_dist_batch = i_all_node_distances.read();
    uint32_t in_node_base_id = 0;
    uint32_t id_idx = 0;
    uint32_t in_node_end_id;
    in_node_end_id = in_dist_batch.end_pos;
#pragma HLS BIND_STORAGE variable=in_node_end_id type=register impl=srl
    LOOP_WHILE_52:
    while (true) {
#pragma HLS PIPELINE II=1
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
#pragma HLS BIND_OP variable=in_node_end_id op=add impl=fabric latency=0
            continue;
        }
        // Target found, calculate index and copy distance
        out_dist_batch.data[id_idx] = in_dist_batch.data[target_node_id - in_node_base_id];
        id_idx = (id_idx + 1);
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

static void Scatt_302(hls::stream<internal_end_data_batch_t> &i_0, hls::stream<struct_abu_9_t> &o_0, hls::stream<struct_nbu_11_t> &o_1) {
    internal_end_data_batch_t in_batch_i_0;
    struct_abu_9_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
    LOOP_WHILE_55:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        LOOP_FOR_54:
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

static void CopyC_306(hls::stream<struct_nbu_11_t> &i_0, hls::stream<struct_nbu_11_t> &o_0, hls::stream<struct_nbu_11_t> &o_1) {
    struct_nbu_11_t in_batch_i_0;
    struct_nbu_11_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
    LOOP_WHILE_57:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        LOOP_FOR_56:
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

static void fused_op_294(hls::stream<struct_abu_9_t> &i_0, hls::stream<struct_abu_9_t> &i_1, hls::stream<struct_nbu_11_t> &i_2, hls::stream<internal_end_data_batch_t> &o_0) {
    struct_abu_9_t in_batch_i_0;
    struct_abu_9_t in_batch_i_1;
    struct_nbu_11_t in_batch_i_2;
    internal_end_data_batch_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    LOOP_WHILE_59:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
        LOOP_FOR_58:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // -- Inlining FusedOp fused_op_294 --
            // Inlining BinOp_128
            ap_fixed_pod_t fused_temp_BinOp_128_o_0;
            ap_fixed<32, 16> lhs_128 = *reinterpret_cast<ap_fixed<32, 16>*>(&in_batch_i_0.data[i]);
            ap_fixed<32, 16> rhs_128 = *reinterpret_cast<ap_fixed<32, 16>*>(&in_batch_i_1.data[i]);
            ap_fixed<32, 16> temp_BinOp_128_o_0_ap_result;
            temp_BinOp_128_o_0_ap_result = (((lhs_128) < (rhs_128) ? lhs_128 : rhs_128));
            fused_temp_BinOp_128_o_0 = *reinterpret_cast<int32_t*>(&temp_BinOp_128_o_0_ap_result);
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
static void memory_loader(int32_t instantiate_idx, const int32_t* src_offsets, const edge_des_burst_t* edge_des_bursts, const int32_t* node_distances, int32_t num_nodes, int32_t num_edges, hls::stream<edge_batch_t> &response_to_318, hls::stream<node_dist_batch_t> &all_node_distances_to_343) {
#pragma HLS function_instantiate variable=instantiate_idx
#pragma HLS DATAFLOW
    hls::stream<node_distance_burst_t> node_distance_burst_stream_0;
#pragma HLS STREAM variable=node_distance_burst_stream_0 depth=12
    hls::stream<node_distance_burst_t> node_distance_burst_stream_1;
#pragma HLS STREAM variable=node_distance_burst_stream_1 depth=12
    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable=edge_stream depth=12
    hls::stream<int32_t> src_offsets_cache_stream;
#pragma HLS STREAM variable=src_offsets_cache_stream depth=32
    src_offset_loader(src_offsets, src_offsets_cache_stream, num_nodes);
    node_property_loader(node_distances, node_distance_burst_stream_0, node_distance_burst_stream_1, num_nodes);
    edge_descriptor_loader(edge_des_bursts, edge_stream, num_edges);
    edge_property_loader_and_dispatcher(src_offsets_cache_stream, edge_stream, node_distance_burst_stream_0, num_nodes, response_to_318);
    node_property_responder(node_distance_burst_stream_1, num_nodes, all_node_distances_to_343);
}

static void graphyflow_big_dataflow(hls::stream<edge_batch_t> &response_to_318, hls::stream<node_dist_batch_t> &all_node_distances_to_343, hls::stream<internal_end_data_batch_t> &internal_end_stream) {
#pragma HLS DATAFLOW
    hls::stream<struct_kbu_50_t> reduce_105_z2d_pair;
#pragma HLS STREAM variable=reduce_105_z2d_pair depth=4
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_d2o_pair[8];
#pragma HLS STREAM variable=reduce_105_d2o_pair depth=4
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_o2u_pair[8];
#pragma HLS STREAM variable=reduce_105_o2u_pair depth=4
    hls::stream<struct_ibu_14_t> intermediate_key;
#pragma HLS STREAM variable=intermediate_key depth=4
    hls::stream<internal_end_data_batch_t> intermediate_transform;
#pragma HLS STREAM variable=intermediate_transform depth=4
    hls::stream<struct_sbu_7_t> stream_o_0_273;
#pragma HLS STREAM variable=stream_o_0_273 depth=4
    hls::stream<struct_abu_9_t> stream_o_0_236;
#pragma HLS STREAM variable=stream_o_0_236 depth=4
    hls::stream<struct_nbu_11_t> stream_o_1_237;
#pragma HLS STREAM variable=stream_o_1_237 depth=4
    hls::stream<struct_abu_9_t> stream_o_2_238;
#pragma HLS STREAM variable=stream_o_2_238 depth=4
    hls::stream<struct_ibu_14_t> stream_o_0_node_id_232;
#pragma HLS STREAM variable=stream_o_0_node_id_232 depth=4
    hls::stream<struct_nbu_11_t> stream_o_1_250;
#pragma HLS STREAM variable=stream_o_1_250 depth=4
    hls::stream<internal_end_data_batch_t> stream_o_0_107;
#pragma HLS STREAM variable=stream_o_0_107 depth=4
    hls::stream<struct_nbu_11_t> stream_o_0_249;
#pragma HLS STREAM variable=stream_o_0_249 depth=4
    hls::stream<struct_abu_9_t> stream_o_0_edge_src_distance_275;
#pragma HLS STREAM variable=stream_o_0_edge_src_distance_275 depth=4
    hls::stream<struct_nbu_11_t> stream_o_0_edge_dst_277;
#pragma HLS STREAM variable=stream_o_0_edge_dst_277 depth=4
    hls::stream<struct_abu_9_t> stream_o_0_edge_weight_278;
#pragma HLS STREAM variable=stream_o_0_edge_weight_278 depth=4
    hls::stream<struct_abu_9_t> stream_o_0_node_distance_300;
#pragma HLS STREAM variable=stream_o_0_node_distance_300 depth=4
    hls::stream<struct_nbu_11_t> stream_o_1_309;
#pragma HLS STREAM variable=stream_o_1_309 depth=4
    hls::stream<struct_abu_9_t> stream_o_0_304;
#pragma HLS STREAM variable=stream_o_0_304 depth=4
    hls::stream<struct_nbu_11_t> stream_o_1_305;
#pragma HLS STREAM variable=stream_o_1_305 depth=4
    hls::stream<struct_nbu_11_t> stream_o_0_308;
#pragma HLS STREAM variable=stream_o_0_308 depth=4
    // --- Function Calls (in topological order) ---
    Memor_274(response_to_318, stream_o_0_edge_src_distance_275, stream_o_0_edge_dst_277, stream_o_0_edge_weight_278);
    fused_op_269(stream_o_0_edge_src_distance_275, stream_o_0_edge_dst_277, stream_o_0_edge_weight_278, stream_o_0_273);
    Scatt_234(stream_o_0_273, stream_o_0_236, stream_o_1_237, stream_o_2_238);
    CopyC_247(stream_o_1_237, stream_o_0_249, stream_o_1_250);
    Memor_231(stream_o_0_node_id_232, stream_o_1_250);
    // --- Start of Reduce Super-Block for Reduc_105 ---
    Reduc_105_pre_process(stream_o_0_node_id_232, stream_o_0_249, stream_o_0_236, stream_o_2_238, intermediate_key, intermediate_transform);
    stream_zipper_0(intermediate_key, intermediate_transform, reduce_105_z2d_pair);
    demux_1(reduce_105_z2d_pair, reduce_105_d2o_pair);
    omega_switch_2(reduce_105_d2o_pair, reduce_105_o2u_pair);
    Reduc_105_unit_reduce(reduce_105_o2u_pair, stream_o_0_107);
    // --- End of Reduce Super-Block for Reduc_105 ---
    Scatt_302(stream_o_0_107, stream_o_0_304, stream_o_1_305);
    CopyC_306(stream_o_1_305, stream_o_0_308, stream_o_1_309);
    Memor_299(all_node_distances_to_343, stream_o_0_node_distance_300, stream_o_1_309);
    fused_op_294(stream_o_0_304, stream_o_0_node_distance_300, stream_o_0_308, internal_end_stream);
}

static void final_writeback(int32_t instantiate_idx, hls::stream<internal_end_data_batch_t> &internal_end_stream, KernelOutputBatch* out_o_0_342) {
#pragma HLS function_instantiate variable=instantiate_idx
#pragma HLS DATAFLOW
    hls::stream<KernelOutputBatch> converted_stream;
#pragma HLS STREAM variable=converted_stream depth=12
    final_convert(internal_end_stream, converted_stream);
    final_write(converted_stream, out_o_0_342);
}

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void graphyflow_big(
    const int32_t* src_offsets,
    const edge_des_burst_t* edge_des_bursts,
    const int32_t* node_distances,
    KernelOutputBatch* o_0_342,
    int32_t num_nodes,
    int32_t num_edges
) {
#pragma HLS INTERFACE m_axi port=src_offsets offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=edge_des_bursts offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=node_distances offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=o_0_342 offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=src_offsets
#pragma HLS INTERFACE s_axilite port=edge_des_bursts
#pragma HLS INTERFACE s_axilite port=node_distances
#pragma HLS INTERFACE s_axilite port=o_0_342
#pragma HLS INTERFACE s_axilite port=num_nodes
#pragma HLS INTERFACE s_axilite port=num_edges
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW
    hls::stream<edge_batch_t> stream_edge_data;
#pragma HLS STREAM variable=stream_edge_data depth=4
    hls::stream<node_dist_batch_t> stream_node_dist_data;
#pragma HLS STREAM variable=stream_node_dist_data depth=4
    hls::stream<internal_end_data_batch_t> stream_result_data;
#pragma HLS STREAM variable=stream_result_data depth=4
    memory_loader(0, src_offsets, edge_des_bursts, node_distances, num_nodes, num_edges, stream_edge_data, stream_node_dist_data);
    graphyflow_big_dataflow(stream_edge_data, stream_node_dist_data, stream_result_data);
    final_writeback(0, stream_result_data, o_0_342);
}

#include "graphyflow.h"

// --- Utility Network Functions ---
void stream_zipper_0(hls::stream<struct_ibu_14_t> &in_key_batch_stream,
                     hls::stream<struct_sbu_19_t> &in_transform_batch_stream,
                     hls::stream<struct_kbu_30_t> &out_pair_batch_stream) {
    struct_ibu_14_t key_batch;
    struct_sbu_19_t transform_batch;
    struct_kbu_30_t out_batch;
    while (true) {
#pragma HLS PIPELINE
        key_batch = in_key_batch_stream.read();
        transform_batch = in_transform_batch_stream.read();
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

void demux_1(hls::stream<struct_kbu_30_t> &in_batch_stream,
             hls::stream<net_wrapper_kt_pair_141_t_t> (&out_streams)[8]) {
    struct_kbu_30_t in_batch;
    while (true) {
#pragma HLS PIPELINE
        in_batch = in_batch_stream.read();
        net_wrapper_kt_pair_141_t_t wrapper_data;
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
    net_wrapper_kt_pair_141_t_t end_wrapper;
    end_wrapper.end_flag = true;
    for (uint32_t i = 0; i < 8; i++) {
#pragma HLS UNROLL
        out_streams[i].write(end_wrapper);
    }
}

void sender_2(int32_t i, hls::stream<net_wrapper_kt_pair_141_t_t> &in1,
              hls::stream<net_wrapper_kt_pair_141_t_t> &in2,
              hls::stream<net_wrapper_kt_pair_141_t_t> &out1,
              hls::stream<net_wrapper_kt_pair_141_t_t> &out2,
              hls::stream<net_wrapper_kt_pair_141_t_t> &out3,
              hls::stream<net_wrapper_kt_pair_141_t_t> &out4) {
#pragma HLS function_instantiate variable = i
    bool in1_end_flag = false;
    bool in2_end_flag = false;
    while (true) {
#pragma HLS PIPELINE II = 1
        if ((!in1.empty())) {
            net_wrapper_kt_pair_141_t_t data1;
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
            net_wrapper_kt_pair_141_t_t data2;
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
            net_wrapper_kt_pair_141_t_t data;
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

void receiver_2(int32_t i, hls::stream<net_wrapper_kt_pair_141_t_t> &out1,
                hls::stream<net_wrapper_kt_pair_141_t_t> &out2,
                hls::stream<net_wrapper_kt_pair_141_t_t> &in1,
                hls::stream<net_wrapper_kt_pair_141_t_t> &in2,
                hls::stream<net_wrapper_kt_pair_141_t_t> &in3,
                hls::stream<net_wrapper_kt_pair_141_t_t> &in4) {
#pragma HLS function_instantiate variable = i
    bool in1_end_flag = false;
    bool in2_end_flag = false;
    bool in3_end_flag = false;
    bool in4_end_flag = false;
    while (true) {
#pragma HLS PIPELINE II = 1
        if ((!in1.empty())) {
            net_wrapper_kt_pair_141_t_t data;
            data = in1.read();
            if ((!data.end_flag)) {
                out1.write(data);
            } else {
                in1_end_flag = true;
            }
        } else if ((!in3.empty())) {
            net_wrapper_kt_pair_141_t_t data;
            data = in3.read();
            if ((!data.end_flag)) {
                out1.write(data);
            } else {
                in3_end_flag = true;
            }
        }
        if ((!in2.empty())) {
            net_wrapper_kt_pair_141_t_t data;
            data = in2.read();
            if ((!data.end_flag)) {
                out2.write(data);
            } else {
                in2_end_flag = true;
            }
        } else if ((!in4.empty())) {
            net_wrapper_kt_pair_141_t_t data;
            data = in4.read();
            if ((!data.end_flag)) {
                out2.write(data);
            } else {
                in4_end_flag = true;
            }
        }
        if ((((in1_end_flag & in2_end_flag) & in3_end_flag) & in4_end_flag)) {
            net_wrapper_kt_pair_141_t_t data;
            data.end_flag = true;
            out1.write(data);
            out2.write(data);
            break;
        }
    }
}

void switch2x2_2(int32_t i, hls::stream<net_wrapper_kt_pair_141_t_t> &in1,
                 hls::stream<net_wrapper_kt_pair_141_t_t> &in2,
                 hls::stream<net_wrapper_kt_pair_141_t_t> &out1,
                 hls::stream<net_wrapper_kt_pair_141_t_t> &out2) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_141_t_t> l1_1;
#pragma HLS STREAM variable = l1_1 depth = 2
    hls::stream<net_wrapper_kt_pair_141_t_t> l1_2;
#pragma HLS STREAM variable = l1_2 depth = 2
    hls::stream<net_wrapper_kt_pair_141_t_t> l1_3;
#pragma HLS STREAM variable = l1_3 depth = 2
    hls::stream<net_wrapper_kt_pair_141_t_t> l1_4;
#pragma HLS STREAM variable = l1_4 depth = 2
    sender_2(i, in1, in2, l1_1, l1_2, l1_3, l1_4);
    receiver_2(i, out1, out2, l1_1, l1_2, l1_3, l1_4);
}

void omega_switch_2(
    hls::stream<net_wrapper_kt_pair_141_t_t> (&in_streams)[8],
    hls::stream<net_wrapper_kt_pair_141_t_t> (&out_streams)[8]) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_141_t_t> stream_stage_0[8];
#pragma HLS STREAM variable = stream_stage_0 depth = 2
    hls::stream<net_wrapper_kt_pair_141_t_t> stream_stage_1[8];
#pragma HLS STREAM variable = stream_stage_1 depth = 2
    switch2x2_2(2, in_streams[0], in_streams[1], stream_stage_0[0],
                stream_stage_0[1]);
    switch2x2_2(2, in_streams[2], in_streams[3], stream_stage_0[2],
                stream_stage_0[3]);
    switch2x2_2(2, in_streams[4], in_streams[5], stream_stage_0[4],
                stream_stage_0[5]);
    switch2x2_2(2, in_streams[6], in_streams[7], stream_stage_0[6],
                stream_stage_0[7]);
    switch2x2_2(1, stream_stage_0[0], stream_stage_0[4], stream_stage_1[0],
                stream_stage_1[1]);
    switch2x2_2(1, stream_stage_0[1], stream_stage_0[5], stream_stage_1[2],
                stream_stage_1[3]);
    switch2x2_2(1, stream_stage_0[2], stream_stage_0[6], stream_stage_1[4],
                stream_stage_1[5]);
    switch2x2_2(1, stream_stage_0[3], stream_stage_0[7], stream_stage_1[6],
                stream_stage_1[7]);
    switch2x2_2(0, stream_stage_1[0], stream_stage_1[4], out_streams[0],
                out_streams[1]);
    switch2x2_2(0, stream_stage_1[1], stream_stage_1[5], out_streams[2],
                out_streams[3]);
    switch2x2_2(0, stream_stage_1[2], stream_stage_1[6], out_streams[4],
                out_streams[5]);
    switch2x2_2(0, stream_stage_1[3], stream_stage_1[7], out_streams[6],
                out_streams[7]);
}

// --- DFIR Component Functions ---
void Reduc_141_pre_process(
    hls::stream<struct_ibu_14_t> &i_global_data_0,
    hls::stream<struct_nbu_16_t> &i_global_data_1,
    hls::stream<struct_ibu_14_t> &i_global_data_2,
    hls::stream<struct_ibu_14_t> &i_global_data_3,
    hls::stream<struct_ibu_14_t> &intermediate_key,
    hls::stream<struct_sbu_19_t> &intermediate_transform) {
    struct_ibu_14_t in_batch_i_global_data_0;
    struct_nbu_16_t in_batch_i_global_data_1;
    struct_ibu_14_t in_batch_i_global_data_2;
    struct_ibu_14_t in_batch_i_global_data_3;
    struct_ibu_14_t out_batch_intermediate_key;
    struct_sbu_19_t out_batch_intermediate_transform;
    bool end_flag;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_global_data_0 = i_global_data_0.read();
        in_batch_i_global_data_1 = i_global_data_1.read();
        in_batch_i_global_data_2 = i_global_data_2.read();
        in_batch_i_global_data_3 = i_global_data_3.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            int32_t key_out_elem;
            struct_in_17_t transform_out_elem;
            // -- Inline sub graph --
            // Inlining fused_op_191
            // -- Begin Nested Inline for FusedOp fused_op_191 --
            // Inlining Place_187
            key_out_elem = in_batch_i_global_data_0.data[i];
            // -- End Nested Inline for FusedOp fused_op_191 --
            // -- Inline sub graph end --
            // -- Inline sub graph --
            // Inlining fused_op_221
            // -- Begin Nested Inline for FusedOp fused_op_221 --
            // Inlining BinOp_104
            int32_t fused_temp_BinOp_104_o_0;
            fused_temp_BinOp_104_o_0 = (in_batch_i_global_data_1.data[i] +
                                        in_batch_i_global_data_2.data[i]);
            // Inlining Gathe_215
            transform_out_elem.ele_0 = fused_temp_BinOp_104_o_0;
            transform_out_elem.ele_1 = in_batch_i_global_data_0.data[i];
            // -- End Nested Inline for FusedOp fused_op_221 --
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

void Reduc_141_unit_reduce(
    hls::stream<net_wrapper_kt_pair_141_t_t> (&kt_wrap_item)[PE_NUM],
    hls::stream<struct_sbu_19_t> &o_0) {
    // 1. Stateful memories for PE_NUM parallel reduction units
    struct_sb_38_t key_mem[PE_NUM][MAX_NUM];
#pragma HLS BIND_STORAGE variable = key_mem type = RAM_2P impl = URAM
#pragma HLS ARRAY_PARTITION variable = key_mem complete dim = 1
    struct_sb_38_t key_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = key_buffer complete dim = 0
    uint32_t i_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = i_buffer complete dim = 0
    // 2. Memory initialization for all PEs
    for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
        for (uint32_t i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
            i_buffer[pe][i] = (MAX_NUM + 1);
        }
    }
    memset(key_mem, 0, sizeof(key_mem));
    // 3. Main processing loop for aggregation across PEs
    bool end_flag;
    bool all_end_flags[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = all_end_flags complete dim = 0
    for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
        all_end_flags[i] = false;
    }
    while (true) {
#pragma HLS PIPELINE
        net_wrapper_kt_pair_141_t_t kt_elem;
        int32_t key_elem;
        struct_in_17_t transform_elem;
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if (((!all_end_flags[i]) & (!kt_wrap_item[i].empty()))) {
                kt_elem = kt_wrap_item[i].read();
                if (kt_elem.end_flag) {
                    all_end_flags[i] = kt_elem.end_flag;
                } else {
                    key_elem = kt_elem.data.key;
                    transform_elem = kt_elem.data.transform;
                    struct_sb_38_t old_ele;
                    old_ele = key_mem[i][key_elem];
                    for (uint32_t i_search = 0; i_search < L + 1; i_search++) {
#pragma HLS UNROLL
                        if ((key_elem == i_buffer[i][i_search])) {
                            old_ele = key_buffer[i][i_search];
                        }
                    }
                    for (uint32_t i_move = 0; i_move < L; i_move++) {
#pragma HLS UNROLL
                        {
                            i_buffer[i][i_move] = i_buffer[i][i_move + 1];
                            key_buffer[i][i_move] = key_buffer[i][i_move + 1];
                        }
                    }
                    struct_sb_38_t new_ele;
                    if (old_ele.ele_1) {
                        struct_in_17_t old_data;
                        old_data = old_ele.ele_0;
                        // -- Inline sub graph --
                        // Inlining Scatt_256
                        int32_t temp_Scatt_256_o_0;
                        node_id_t temp_Scatt_256_o_1;
                        temp_Scatt_256_o_0 = old_data.ele_0;
                        temp_Scatt_256_o_1 = old_data.ele_1;
                        // Inlining Scatt_260
                        int32_t temp_Scatt_260_o_0;
                        node_id_t temp_Scatt_260_o_1;
                        temp_Scatt_260_o_0 = transform_elem.ele_0;
                        // Inlining fused_op_250
                        // -- Begin Nested Inline for FusedOp fused_op_250 --
                        // Inlining BinOp_132
                        int32_t fused_temp_BinOp_132_o_0;
                        ap_fixed<32, 16> lhs_132 =
                            *reinterpret_cast<ap_fixed<32, 16> *>(
                                &temp_Scatt_256_o_0);
                        ap_fixed<32, 16> rhs_132 =
                            *reinterpret_cast<ap_fixed<32, 16> *>(
                                &temp_Scatt_260_o_0);
                        ap_fixed<32, 16> temp_BinOp_132_o_0_ap_result;
                        temp_BinOp_132_o_0_ap_result =
                            (((lhs_132) < (rhs_132) ? lhs_132 : rhs_132));
                        fused_temp_BinOp_132_o_0 = *reinterpret_cast<int32_t *>(
                            &temp_BinOp_132_o_0_ap_result);
                        // Inlining Gathe_244
                        new_ele.ele_0.ele_0 = fused_temp_BinOp_132_o_0;
                        new_ele.ele_0.ele_1 = temp_Scatt_256_o_1;
                        // -- End Nested Inline for FusedOp fused_op_250 --
                        // -- Inline sub graph end --
                        new_ele.ele_1 = true;
                    } else {
                        new_ele.ele_1 = true;
                        new_ele.ele_0 = transform_elem;
                    }
                    key_mem[i][key_elem] = new_ele;
                    key_buffer[i][L] = new_ele;
                    i_buffer[i][L] = key_elem;
                }
            }
        }
        end_flag = true;
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            end_flag = (end_flag & all_end_flags[i]);
        }
        if (end_flag) {
            break;
        }
    }
    // 4. Final output loop to drain all PE memories with swapped loops
    uint32_t data_cnt = 0;
    data_cnt = 0;
    uint32_t start_pos = 0;
    start_pos = 0;
    struct_sbu_19_t data_pack;
    data_pack.end_flag = false;
    struct_in_17_t data_to_write[((PE_NUM << 1))];
#pragma HLS ARRAY_PARTITION variable = data_to_write complete dim = 0
    uint32_t k = 0;
    k = 0;
    while ((k < MAX_NUM)) {
#pragma HLS PIPELINE
        for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            if (key_mem[pe][(k + pe)].ele_1) {
                data_to_write[(start_pos % ((PE_NUM << 1)))] =
                    key_mem[pe][(k + pe)].ele_0;
                data_cnt = (data_cnt + 1);
                start_pos = (start_pos + 1);
            }
        }
        if ((data_cnt >= PE_NUM)) {
            data_pack.end_pos = PE_NUM;
            for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
                data_pack.data[i] = data_to_write[(
                    ((start_pos - data_cnt) + i) % (PE_NUM << 1))];
            }
            o_0.write(data_pack);
            data_cnt = (data_cnt - PE_NUM);
        }
        k = (k + PE_NUM);
    }
    // 5. Drain any remaining data and send final batch with end_flag
    data_pack.end_flag = true;
    data_pack.end_pos = data_cnt;
    for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
        if ((i < data_cnt)) {
            data_pack.data[i] =
                data_to_write[(((start_pos - data_cnt) + i) % (PE_NUM << 1))];
        }
    }
    o_0.write(data_pack);
}

void Colle_65(hls::stream<struct_obu_10_t> &i_0,
              hls::stream<struct_sbu_12_t> &o_0) {
    struct_obu_10_t in_batch_i_0;
    struct_sbu_12_t out_batch_o_0;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        uint8_t out_idx = 0;
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if ((in_batch_i_0.data[i].valid & (i < in_batch_i_0.end_pos))) {
                out_batch_o_0.data[out_idx] = in_batch_i_0.data[i].data;
                out_idx = (out_idx + 1);
            }
        }
        out_batch_o_0.end_pos = out_idx;
        out_batch_o_0.end_flag = in_batch_i_0.end_flag;
        o_0.write(out_batch_o_0);
        if (in_batch_i_0.end_flag) {
            break;
        }
    }
}

void Scatt_270(hls::stream<struct_sbu_12_t> &i_0,
               hls::stream<struct_ibu_14_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1,
               hls::stream<struct_ibu_14_t> &o_2) {
    struct_sbu_12_t in_batch_i_0;
    struct_ibu_14_t out_batch_o_0;
    struct_nbu_16_t out_batch_o_1;
    struct_ibu_14_t out_batch_o_2;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
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

void Memor_267(hls::stream<struct_ibu_14_t> &o_0_node_id,
               hls::stream<struct_nbu_16_t> &i_0_node_id) {
    struct_nbu_16_t in_batch_i_0_node_id;
    struct_ibu_14_t out_batch_o_0_node_id;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_node_id = i_0_node_id.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // Memory read from path: (0, 'node', ['id'])
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

void CopyC_283(hls::stream<struct_nbu_16_t> &i_0,
               hls::stream<struct_nbu_16_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1) {
    struct_nbu_16_t in_batch_i_0;
    struct_nbu_16_t out_batch_o_0;
    struct_nbu_16_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
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

void Condi_61(hls::stream<struct_sbu_12_t> &i_data,
              hls::stream<struct_bbu_21_t> &i_cond,
              hls::stream<struct_obu_10_t> &o_0) {
    struct_sbu_12_t in_batch_i_data;
    struct_bbu_21_t in_batch_i_cond;
    struct_obu_10_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_data = i_data.read();
        in_batch_i_cond = i_cond.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i].data = in_batch_i_data.data[i];
            out_batch_o_0.data[i].valid = in_batch_i_cond.data[i];
        }
        end_flag = in_batch_i_data.end_flag;
        end_pos = in_batch_i_data.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        if (end_flag) {
            break;
        }
    }
}

void fused_op_312(hls::stream<struct_ibu_14_t> &i_0,
                  hls::stream<struct_nbu_16_t> &i_1,
                  hls::stream<struct_ibu_14_t> &i_2,
                  hls::stream<struct_bbu_21_t> &o_0,
                  hls::stream<struct_sbu_12_t> &o_1) {
    struct_ibu_14_t in_batch_i_0;
    struct_nbu_16_t in_batch_i_1;
    struct_ibu_14_t in_batch_i_2;
    struct_bbu_21_t out_batch_o_0;
    struct_sbu_12_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // -- Inlining FusedOp fused_op_312 --
            // Inlining Const_37
            int32_t fused_temp_Const_37_o_0;
            // Inlining CopyC_306
            int32_t fused_temp_CopyC_306_o_0;
            int32_t fused_temp_CopyC_306_o_1;
            fused_temp_CopyC_306_o_0 = in_batch_i_2.data[i];
            fused_temp_CopyC_306_o_1 = in_batch_i_2.data[i];
            // Inlining BinOp_48
            ap_fixed<32, 16> lhs_48 = *reinterpret_cast<ap_fixed<32, 16> *>(
                &fused_temp_CopyC_306_o_1);
            out_batch_o_0.data[i] =
                (lhs_48 >= ((ap_fixed<32, 16>)(((ap_fixed<32, 16>)0.0))));
            // Inlining Gathe_301
            out_batch_o_1.data[i].ele_0 = in_batch_i_0.data[i];
            out_batch_o_1.data[i].ele_1 = in_batch_i_1.data[i];
            out_batch_o_1.data[i].ele_2 = fused_temp_CopyC_306_o_0;
            // -- End Inlining FusedOp fused_op_312 --
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

void Memor_318(hls::stream<struct_ibu_14_t> &o_0_edge_weight,
               hls::stream<struct_ebu_4_t> &i_0_edge_id,
               hls::stream<struct_ibu_14_t> &o_0_edge_src_distance,
               hls::stream<struct_nbu_16_t> &o_0_edge_dst) {
    struct_ebu_4_t in_batch_i_0_edge_id;
    struct_ibu_14_t out_batch_o_0_edge_weight;
    struct_ibu_14_t out_batch_o_0_edge_src_distance;
    struct_nbu_16_t out_batch_o_0_edge_dst;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_edge_id = i_0_edge_id.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // Memory read from path: (0, 'edge', ('weight',))
            // Memory read from path: (0, 'edge', ('src', 'distance'))
            // Memory read from path: (0, 'edge', ('dst',))
        }
        end_flag = in_batch_i_0_edge_id.end_flag;
        end_pos = in_batch_i_0_edge_id.end_pos;
        out_batch_o_0_edge_weight.end_flag = end_flag;
        out_batch_o_0_edge_weight.end_pos = end_pos;
        o_0_edge_weight.write(out_batch_o_0_edge_weight);
        out_batch_o_0_edge_src_distance.end_flag = end_flag;
        out_batch_o_0_edge_src_distance.end_pos = end_pos;
        o_0_edge_src_distance.write(out_batch_o_0_edge_src_distance);
        out_batch_o_0_edge_dst.end_flag = end_flag;
        out_batch_o_0_edge_dst.end_pos = end_pos;
        o_0_edge_dst.write(out_batch_o_0_edge_dst);
        if (end_flag) {
            break;
        }
    }
}

void CopyC_350(hls::stream<struct_nbu_16_t> &i_0,
               hls::stream<struct_nbu_16_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1) {
    struct_nbu_16_t in_batch_i_0;
    struct_nbu_16_t out_batch_o_0;
    struct_nbu_16_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
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

void Memor_343(hls::stream<struct_ibu_14_t> &o_0_node_distance,
               hls::stream<struct_nbu_16_t> &i_0_node_id) {
    struct_nbu_16_t in_batch_i_0_node_id;
    struct_ibu_14_t out_batch_o_0_node_distance;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_node_id = i_0_node_id.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // Memory read from path: (0, 'node', ('distance',))
        }
        end_flag = in_batch_i_0_node_id.end_flag;
        end_pos = in_batch_i_0_node_id.end_pos;
        out_batch_o_0_node_distance.end_flag = end_flag;
        out_batch_o_0_node_distance.end_pos = end_pos;
        o_0_node_distance.write(out_batch_o_0_node_distance);
        if (end_flag) {
            break;
        }
    }
}

void fused_op_338(hls::stream<struct_ibu_14_t> &i_0,
                  hls::stream<struct_ibu_14_t> &i_1,
                  hls::stream<struct_nbu_16_t> &i_2,
                  hls::stream<struct_sbu_19_t> &o_0) {
    struct_ibu_14_t in_batch_i_0;
    struct_ibu_14_t in_batch_i_1;
    struct_nbu_16_t in_batch_i_2;
    struct_sbu_19_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // -- Inlining FusedOp fused_op_338 --
            // Inlining BinOp_164
            int32_t fused_temp_BinOp_164_o_0;
            ap_fixed<32, 16> lhs_164 =
                *reinterpret_cast<ap_fixed<32, 16> *>(&in_batch_i_0.data[i]);
            ap_fixed<32, 16> rhs_164 =
                *reinterpret_cast<ap_fixed<32, 16> *>(&in_batch_i_1.data[i]);
            ap_fixed<32, 16> temp_BinOp_164_o_0_ap_result;
            temp_BinOp_164_o_0_ap_result =
                (((lhs_164) < (rhs_164) ? lhs_164 : rhs_164));
            fused_temp_BinOp_164_o_0 =
                *reinterpret_cast<int32_t *>(&temp_BinOp_164_o_0_ap_result);
            // Inlining Gathe_332
            out_batch_o_0.data[i].ele_0 = fused_temp_BinOp_164_o_0;
            out_batch_o_0.data[i].ele_1 = in_batch_i_2.data[i];
            // -- End Inlining FusedOp fused_op_338 --
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

void Scatt_346(hls::stream<struct_sbu_19_t> &i_0,
               hls::stream<struct_ibu_14_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1) {
    struct_sbu_19_t in_batch_i_0;
    struct_ibu_14_t out_batch_o_0;
    struct_nbu_16_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].ele_0;
            out_batch_o_1.data[i] = in_batch_i_0.data[i].ele_1;
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

static void
mem_to_stream_func(const struct_ebu_4_t *in_i_0_edge_id_320,
                   hls::stream<struct_ebu_4_t> &out_i_0_edge_id_320_stream,
                   uint16_t num_batches) {
    for (uint32_t i = 0; i < num_batches; i++) {
#pragma HLS PIPELINE
        out_i_0_edge_id_320_stream.write(in_i_0_edge_id_320[i]);
    }
}

static void stream_to_mem_func(hls::stream<struct_sbu_19_t> &in_o_0_342_stream,
                               KernelOutputBatch *out_o_0_342) {
    int32_t i = 0;
    while (true) {
#pragma HLS PIPELINE
        struct_sbu_19_t internal_batch;
        internal_batch = in_o_0_342_stream.read();
        KernelOutputBatch output_batch;
        for (uint32_t k = 0; k < PE_NUM; k++) {
#pragma HLS UNROLL
            ap_fixed<32, 16> final_dist_fp;
            final_dist_fp = *reinterpret_cast<ap_fixed<32, 16> *>(
                &internal_batch.data[k].ele_0);
            output_batch.data[k].distance = (float)final_dist_fp;
            output_batch.data[k].id = internal_batch.data[k].ele_1.id;
        }
        output_batch.end_flag = internal_batch.end_flag;
        output_batch.end_pos = internal_batch.end_pos;
        out_o_0_342[i] = output_batch;
        if (out_o_0_342[i].end_flag) {
            break;
        }
        i = (i + 1);
    }
}

static void
graphyflow_dataflow(hls::stream<struct_ebu_4_t> &i_0_edge_id_320_stream,
                    hls::stream<struct_sbu_19_t> &o_0_342_stream) {
#pragma HLS DATAFLOW
    hls::stream<struct_kbu_30_t> reduce_141_z2d_pair;
#pragma HLS STREAM variable = reduce_141_z2d_pair depth = 4
    hls::stream<net_wrapper_kt_pair_141_t_t> reduce_141_d2o_pair[8];
#pragma HLS STREAM variable = reduce_141_d2o_pair depth = 4
    hls::stream<net_wrapper_kt_pair_141_t_t> reduce_141_o2u_pair[8];
#pragma HLS STREAM variable = reduce_141_o2u_pair depth = 4
    hls::stream<struct_ibu_14_t> intermediate_key;
#pragma HLS STREAM variable = intermediate_key depth = 4
    hls::stream<struct_sbu_19_t> intermediate_transform;
#pragma HLS STREAM variable = intermediate_transform depth = 4
    hls::stream<struct_obu_10_t> stream_o_0_64;
#pragma HLS STREAM variable = stream_o_0_64 depth = 4
    hls::stream<struct_sbu_12_t> stream_o_0_67;
#pragma HLS STREAM variable = stream_o_0_67 depth = 4
    hls::stream<struct_ibu_14_t> stream_o_0_272;
#pragma HLS STREAM variable = stream_o_0_272 depth = 4
    hls::stream<struct_nbu_16_t> stream_o_1_273;
#pragma HLS STREAM variable = stream_o_1_273 depth = 4
    hls::stream<struct_ibu_14_t> stream_o_2_274;
#pragma HLS STREAM variable = stream_o_2_274 depth = 4
    hls::stream<struct_ibu_14_t> stream_o_0_node_id_268;
#pragma HLS STREAM variable = stream_o_0_node_id_268 depth = 4
    hls::stream<struct_nbu_16_t> stream_o_1_286;
#pragma HLS STREAM variable = stream_o_1_286 depth = 4
    hls::stream<struct_sbu_19_t> stream_o_0_143;
#pragma HLS STREAM variable = stream_o_0_143 depth = 4
    hls::stream<struct_nbu_16_t> stream_o_0_285;
#pragma HLS STREAM variable = stream_o_0_285 depth = 4
    hls::stream<struct_sbu_12_t> stream_o_1_317;
#pragma HLS STREAM variable = stream_o_1_317 depth = 4
    hls::stream<struct_bbu_21_t> stream_o_0_316;
#pragma HLS STREAM variable = stream_o_0_316 depth = 4
    hls::stream<struct_ibu_14_t> stream_o_0_edge_src_distance_321;
#pragma HLS STREAM variable = stream_o_0_edge_src_distance_321 depth = 4
    hls::stream<struct_nbu_16_t> stream_o_0_edge_dst_322;
#pragma HLS STREAM variable = stream_o_0_edge_dst_322 depth = 4
    hls::stream<struct_ibu_14_t> stream_o_0_edge_weight_319;
#pragma HLS STREAM variable = stream_o_0_edge_weight_319 depth = 4
    hls::stream<struct_nbu_16_t> stream_o_1_349;
#pragma HLS STREAM variable = stream_o_1_349 depth = 4
    hls::stream<struct_nbu_16_t> stream_o_0_352;
#pragma HLS STREAM variable = stream_o_0_352 depth = 4
    hls::stream<struct_nbu_16_t> stream_o_1_353;
#pragma HLS STREAM variable = stream_o_1_353 depth = 4
    hls::stream<struct_ibu_14_t> stream_o_0_node_distance_344;
#pragma HLS STREAM variable = stream_o_0_node_distance_344 depth = 4
    hls::stream<struct_ibu_14_t> stream_o_0_348;
#pragma HLS STREAM variable = stream_o_0_348 depth = 4
    // --- Function Calls (in topological order) ---
    Memor_318(stream_o_0_edge_weight_319, i_0_edge_id_320_stream,
              stream_o_0_edge_src_distance_321, stream_o_0_edge_dst_322);
    fused_op_312(stream_o_0_edge_src_distance_321, stream_o_0_edge_dst_322,
                 stream_o_0_edge_weight_319, stream_o_0_316, stream_o_1_317);
    Condi_61(stream_o_1_317, stream_o_0_316, stream_o_0_64);
    Colle_65(stream_o_0_64, stream_o_0_67);
    Scatt_270(stream_o_0_67, stream_o_0_272, stream_o_1_273, stream_o_2_274);
    CopyC_283(stream_o_1_273, stream_o_0_285, stream_o_1_286);
    Memor_267(stream_o_0_node_id_268, stream_o_1_286);
    // --- Start of Reduce Super-Block for Reduc_141 ---
    Reduc_141_pre_process(stream_o_0_node_id_268, stream_o_0_285,
                          stream_o_0_272, stream_o_2_274, intermediate_key,
                          intermediate_transform);
    stream_zipper_0(intermediate_key, intermediate_transform,
                    reduce_141_z2d_pair);
    demux_1(reduce_141_z2d_pair, reduce_141_d2o_pair);
    omega_switch_2(reduce_141_d2o_pair, reduce_141_o2u_pair);
    Reduc_141_unit_reduce(reduce_141_o2u_pair, stream_o_0_143);
    // --- End of Reduce Super-Block for Reduc_141 ---
    Scatt_346(stream_o_0_143, stream_o_0_348, stream_o_1_349);
    CopyC_350(stream_o_1_349, stream_o_0_352, stream_o_1_353);
    Memor_343(stream_o_0_node_distance_344, stream_o_1_353);
    fused_op_338(stream_o_0_348, stream_o_0_node_distance_344, stream_o_0_352,
                 o_0_342_stream);
}

extern "C" void graphyflow(const struct_ebu_4_t *i_0_edge_id_320,
                           KernelOutputBatch *o_0_342, int *stop_flag,
                           uint16_t input_length_in_batches) {
#pragma HLS INTERFACE m_axi port = i_0_edge_id_320 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = o_0_342 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = stop_flag offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = i_0_edge_id_320
#pragma HLS INTERFACE s_axilite port = o_0_342
#pragma HLS INTERFACE s_axilite port = stop_flag
#pragma HLS INTERFACE s_axilite port = input_length_in_batches
#pragma HLS INTERFACE s_axilite port = return
    static hls::stream<struct_ebu_4_t> i_0_edge_id_320_internal_stream;
#pragma HLS STREAM variable = i_0_edge_id_320_internal_stream depth = 4
    static hls::stream<struct_sbu_19_t> o_0_342_internal_stream;
#pragma HLS STREAM variable = o_0_342_internal_stream depth = 4
#pragma HLS DATAFLOW
    mem_to_stream_func(i_0_edge_id_320, i_0_edge_id_320_internal_stream,
                       input_length_in_batches);
    graphyflow_dataflow(i_0_edge_id_320_internal_stream,
                        o_0_342_internal_stream);
    stream_to_mem_func(o_0_342_internal_stream, o_0_342);
}

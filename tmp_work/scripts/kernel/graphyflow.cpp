#include "graphyflow.h"

// --- Utility Network Functions ---
void stream_zipper_0(hls::stream<struct_ibu_24_t> &in_key_batch_stream,
                     hls::stream<struct_sbu_22_t> &in_transform_batch_stream,
                     hls::stream<struct_kbu_33_t> &out_pair_batch_stream) {
    struct_ibu_24_t key_batch;
    struct_sbu_22_t transform_batch;
    struct_kbu_33_t out_batch;
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

void demux_1(hls::stream<struct_kbu_33_t> &in_batch_stream,
             hls::stream<net_wrapper_kt_pair_141_t_t> (&out_streams)[8]) {
    struct_kbu_33_t in_batch;
    while (true) {
#pragma HLS PIPELINE
        // printf("DEBUG: --- reading one in_batch ---\n");
        // fflush(stdout);
        in_batch = in_batch_stream.read();
        net_wrapper_kt_pair_141_t_t wrapper_data;
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if (i < in_batch.end_pos) {
                // printf("DEBUG: --- pos i = %d < end_pos %d ---\n", i,
                // in_batch.end_pos); fflush(stdout);
                wrapper_data.end_flag = false;
                wrapper_data.data = in_batch.data[i];
                out_streams[i].write(wrapper_data);
            } else {
                // printf("DEBUG: --- pos i = %d > end_pos %d, end ---\n", i,
                // in_batch.end_pos); fflush(stdout);
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
    hls::stream<struct_sbu_14_t> &i_0,
    hls::stream<struct_ibu_24_t> &intermediate_key,
    hls::stream<struct_sbu_22_t> &intermediate_transform) {
    struct_sbu_14_t in_batch_i_0;
    struct_ibu_24_t out_batch_intermediate_key;
    struct_sbu_22_t out_batch_intermediate_transform;
    bool end_flag;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            int32_t key_out_elem;
            struct_an_20_t transform_out_elem;
            // -- Inline sub graph --
            // Starting for comp Scatt_68
            ap_fixed<32, 16> temp_Scatt_68_o_0;
            node_t temp_Scatt_68_o_1;
            ap_fixed<32, 16> temp_Scatt_68_o_2;
            temp_Scatt_68_o_0 = in_batch_i_0.data[i].ele_0;
            temp_Scatt_68_o_1 = in_batch_i_0.data[i].ele_1;
            temp_Scatt_68_o_2 = in_batch_i_0.data[i].ele_2;
            // Starting for comp Unary_82
            key_out_elem = temp_Scatt_68_o_1.id;
            // -- Inline sub graph end --
            // -- Inline sub graph --
            // Starting for comp Scatt_90
            ap_fixed<32, 16> temp_Scatt_90_o_0;
            node_t temp_Scatt_90_o_1;
            ap_fixed<32, 16> temp_Scatt_90_o_2;
            temp_Scatt_90_o_0 = in_batch_i_0.data[i].ele_0;
            temp_Scatt_90_o_1 = in_batch_i_0.data[i].ele_1;
            temp_Scatt_90_o_2 = in_batch_i_0.data[i].ele_2;
            // Starting for comp BinOp_104
            ap_fixed<32, 16> temp_BinOp_104_o_0;
            temp_BinOp_104_o_0 = (temp_Scatt_90_o_0 + temp_Scatt_90_o_2);
            // Starting for comp Gathe_108
            transform_out_elem.ele_0 = temp_BinOp_104_o_0;
            transform_out_elem.ele_1 = temp_Scatt_90_o_1;
            // -- Inline sub graph end --
            out_batch_intermediate_key.data[i] = key_out_elem;
            out_batch_intermediate_transform.data[i] = transform_out_elem;
            // printf("DEBUG: --- preprocess: batch data[%d] src_dist = %.2f,
            // dst.id = %d, edge_w = %.2f ---\n", i,
            // (float)in_batch_i_0.data[i].ele_0, in_batch_i_0.data[i].ele_1.id,
            // (float)in_batch_i_0.data[i].ele_2);
            // fflush(stdout);
        }
        // printf("DEBUG: --- preprocess: batch end_flag = %d, end_pos = %d
        // ---\n", in_batch_i_0.end_flag, in_batch_i_0.end_pos); fflush(stdout);
        out_batch_intermediate_key.end_flag = in_batch_i_0.end_flag;
        out_batch_intermediate_key.end_pos = in_batch_i_0.end_pos;
        out_batch_intermediate_transform.end_flag = in_batch_i_0.end_flag;
        out_batch_intermediate_transform.end_pos = in_batch_i_0.end_pos;
        intermediate_key.write(out_batch_intermediate_key);
        intermediate_transform.write(out_batch_intermediate_transform);
        end_flag = in_batch_i_0.end_flag;
        if (end_flag) {
            break;
        }
    }
}

void Reduc_141_unit_reduce(
    hls::stream<net_wrapper_kt_pair_141_t_t> (&kt_wrap_item)[PE_NUM],
    hls::stream<struct_sbu_22_t> &o_0) {
    // 1. Stateful memories for PE_NUM parallel reduction units
    struct_sb_41_t key_mem[PE_NUM][MAX_NUM];
#pragma HLS BIND_STORAGE variable = key_mem type = RAM_2P impl = URAM
#pragma HLS ARRAY_PARTITION variable = key_mem complete dim = 1
    struct_sb_41_t key_buffer[PE_NUM][L + 1];
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
    for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
        for (uint32_t i = 0; i < MAX_NUM; i++) {
#pragma HLS UNROLL
            key_mem[pe][i].ele_1 = false;
        }
    }
    // 3. Main processing loop for aggregation across PEs
    bool end_flag;
    bool all_end_flags[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = all_end_flags complete dim = 0
    for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
        all_end_flags[i] = false;
    }
    while (true) {
        // printf("DEBUG: --- Reduce unit while loop ---\n");
        // fflush(stdout);
#pragma HLS PIPELINE
        net_wrapper_kt_pair_141_t_t kt_elem;
        int32_t key_elem;
        struct_an_20_t transform_elem;
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // printf("DEBUG: --- Reduce unit PE for loop ---\n");
            // fflush(stdout);
            if (((!all_end_flags[i]) & (!kt_wrap_item[i].empty()))) {
                kt_elem = kt_wrap_item[i].read();
                if (kt_elem.end_flag) {
                    // printf("DEBUG: --- end flag detected in for loop ---\n");
                    // fflush(stdout);
                    all_end_flags[i] = kt_elem.end_flag;
                } else {
                    key_elem = kt_elem.data.key;
                    transform_elem = kt_elem.data.transform;
                    struct_sb_41_t old_ele;
                    // printf("DEBUG: --- key_elem = %d data = %.2f ---\n",
                    // key_elem, (float)transform_elem.ele_0); fflush(stdout);
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
                    struct_sb_41_t new_ele;
                    if (old_ele.ele_1) {
                        struct_an_20_t old_data;
                        old_data = old_ele.ele_0;
                        // -- Inline sub graph --
                        // Starting for comp CopyC_125
                        struct_an_20_t temp_CopyC_125_o_0;
                        struct_an_20_t temp_CopyC_125_o_1;
                        temp_CopyC_125_o_0 = old_data;
                        temp_CopyC_125_o_1 = old_data;
                        // Starting for comp Unary_129
                        ap_fixed<32, 16> temp_Unary_129_o_0;
                        temp_Unary_129_o_0 = transform_elem.ele_0;
                        // Starting for comp Unary_119
                        ap_fixed<32, 16> temp_Unary_119_o_0;
                        temp_Unary_119_o_0 = temp_CopyC_125_o_0.ele_0;
                        // Starting for comp Unary_122
                        node_t temp_Unary_122_o_0;
                        temp_Unary_122_o_0 = temp_CopyC_125_o_1.ele_1;
                        // Starting for comp BinOp_132
                        ap_fixed<32, 16> temp_BinOp_132_o_0;
                        temp_BinOp_132_o_0 =
                            (((temp_Unary_119_o_0) < (temp_Unary_129_o_0)
                                  ? temp_Unary_119_o_0
                                  : temp_Unary_129_o_0));
                        // Starting for comp Gathe_136
                        new_ele.ele_0.ele_0 = temp_BinOp_132_o_0;
                        new_ele.ele_0.ele_1 = temp_Unary_122_o_0;
                        // -- Inline sub graph end --
                        new_ele.ele_1 = true;
                    } else {
                        new_ele.ele_1 = true;
                        new_ele.ele_0 = transform_elem;
                    }
                    // printf("DEBUG: --- new_data = %.2f ---\n",
                    // (float)new_ele.ele_0.ele_0);
                    fflush(stdout);
                    key_mem[i][key_elem] = new_ele;
                    key_buffer[i][L] = new_ele;
                    i_buffer[i][L] = key_elem;
                }
            }
        }
        // // printf("DEBUG: --- Reduce unit PE for loop done ---\n");
        // fflush(stdout);
        end_flag = true;
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // // printf("DEBUG: --- setting end flag ---\n");
            // fflush(stdout);
            end_flag = (end_flag & all_end_flags[i]);
        }
        if (end_flag) {
            // // printf("DEBUG: --- end flag detected, breaking ---\n");
            // fflush(stdout);
            break;
        }
    }
    // 4. Final output loop to drain all PE memories with swapped loops
    uint32_t data_cnt;
    data_cnt = 0;
    uint32_t start_pos;
    start_pos = 0;
    struct_sbu_22_t data_pack;
    data_pack.end_flag = false;
    struct_an_20_t data_to_write[(PE_NUM << 1)];
#pragma HLS ARRAY_PARTITION variable = data_to_write complete dim = 0
    uint32_t k;
    k = 0;
    while ((k < MAX_NUM)) {
#pragma HLS PIPELINE
        for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            if (key_mem[pe][(k + pe)].ele_1) {
                data_to_write[(start_pos % (PE_NUM << 1))] =
                    key_mem[pe][(k + pe)].ele_0;
                data_cnt = (data_cnt + 1);
                start_pos = (start_pos + 1);
                // printf("DEBUG: --- mem data key = %d, elem = %.2f ---\n", k +
                // pe, (float)key_mem[pe][(k + pe)].ele_0.ele_0);
            }
        }
        if ((data_cnt >= PE_NUM)) {
            data_pack.end_pos = PE_NUM;
            for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
                data_pack.data[i] = data_to_write[(
                    ((start_pos - data_cnt) + i) % (PE_NUM << 1))];
                // printf("DEBUG: --- write data key = %d, elem = %.2f pos = %d
                // ---\n", data_pack.data[i].ele_1.id,
                // (float)data_pack.data[i].ele_0, (((start_pos - data_cnt) + i)
                // % (PE_NUM << 1)));
            }
            o_0.write(data_pack);
            data_cnt = (data_cnt - PE_NUM);
        }
        k = (k + PE_NUM);
    }
    // 5. Drain any remaining data and send final batch with end_flag
    data_pack.end_flag = true;
    data_pack.end_pos = data_cnt;
    // printf("DEBUG: --- data_cnt = %d ---\n", data_cnt);
    for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
        if ((i < data_cnt)) {
            data_pack.data[i] =
                data_to_write[(((start_pos - data_cnt) + i) % (PE_NUM << 1))];
            // printf("DEBUG: --- write data key = %d, elem = %.2f pos = %d
            // ---\n", data_pack.data[i].ele_1.id,
            // (float)data_pack.data[i].ele_0, (((start_pos - data_cnt) + i) %
            // (PE_NUM << 1)));
        }
    }
    o_0.write(data_pack);
}

void Unary_6(hls::stream<struct_ebu_7_t> &i_0,
             hls::stream<struct_nbu_9_t> &o_0) {
    struct_ebu_7_t in_batch_i_0;
    struct_nbu_9_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].src;
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

void Unary_9(hls::stream<struct_ebu_7_t> &i_0,
             hls::stream<struct_nbu_9_t> &o_0) {
    struct_ebu_7_t in_batch_i_0;
    struct_nbu_9_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].dst;
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

void CopyC_12(hls::stream<struct_ebu_7_t> &i_0,
              hls::stream<struct_ebu_7_t> &o_0,
              hls::stream<struct_ebu_7_t> &o_1) {
    struct_ebu_7_t in_batch_i_0;
    struct_ebu_7_t out_batch_o_0;
    struct_ebu_7_t out_batch_o_1;
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

void Unary_16(hls::stream<struct_ebu_7_t> &i_0,
              hls::stream<struct_abu_11_t> &o_0) {
    struct_ebu_7_t in_batch_i_0;
    struct_abu_11_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].weight;
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

void CopyC_19(hls::stream<struct_ebu_7_t> &i_0,
              hls::stream<struct_ebu_7_t> &o_0,
              hls::stream<struct_ebu_7_t> &o_1) {
    struct_ebu_7_t in_batch_i_0;
    struct_ebu_7_t out_batch_o_0;
    struct_ebu_7_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i];
            out_batch_o_1.data[i] = in_batch_i_0.data[i];
            // printf("DEBUG: --- batch data[%d] src.id = %d, dst.id = %d
            // ---\n", i, in_batch_i_0.data[i].src.id,
            // in_batch_i_0.data[i].dst.id); fflush(stdout);
        }
        // printf("DEBUG: --- batch end_flag = %d, end_pos = %d ---\n",
        // in_batch_i_0.end_flag, in_batch_i_0.end_pos); fflush(stdout);
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

void Unary_23(hls::stream<struct_nbu_9_t> &i_0,
              hls::stream<struct_abu_11_t> &o_0) {
    struct_nbu_9_t in_batch_i_0;
    struct_abu_11_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].distance;
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

void Gathe_27(hls::stream<struct_abu_11_t> &i_0,
              hls::stream<struct_nbu_9_t> &i_1,
              hls::stream<struct_abu_11_t> &i_2,
              hls::stream<struct_sbu_14_t> &o_0) {
    struct_abu_11_t in_batch_i_0;
    struct_nbu_9_t in_batch_i_1;
    struct_abu_11_t in_batch_i_2;
    struct_sbu_14_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i].ele_0 = in_batch_i_0.data[i];
            out_batch_o_0.data[i].ele_1 = in_batch_i_1.data[i];
            out_batch_o_0.data[i].ele_2 = in_batch_i_2.data[i];
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

void CopyC_57(hls::stream<struct_sbu_14_t> &i_0,
              hls::stream<struct_sbu_14_t> &o_0,
              hls::stream<struct_sbu_14_t> &o_1) {
    struct_sbu_14_t in_batch_i_0;
    struct_sbu_14_t out_batch_o_0;
    struct_sbu_14_t out_batch_o_1;
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

void Scatt_32(hls::stream<struct_sbu_14_t> &i_0,
              hls::stream<struct_abu_11_t> &o_2) {
    struct_sbu_14_t in_batch_i_0;
    struct_abu_11_t out_batch_o_2;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_2.data[i] = in_batch_i_0.data[i].ele_2;
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_2.end_flag = end_flag;
        out_batch_o_2.end_pos = end_pos;
        o_2.write(out_batch_o_2);
        if (end_flag) {
            break;
        }
    }
}

void BinOp_48(hls::stream<struct_abu_11_t> &i_0,
              hls::stream<struct_bbu_16_t> &o_0) {
    struct_abu_11_t in_batch_i_0;
    struct_bbu_16_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] =
                (in_batch_i_0.data[i] >= ((ap_fixed<32, 16>)0.0));
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

void Condi_61(hls::stream<struct_sbu_14_t> &i_data,
              hls::stream<struct_bbu_16_t> &i_cond,
              hls::stream<struct_obu_19_t> &o_0) {
    struct_sbu_14_t in_batch_i_data;
    struct_bbu_16_t in_batch_i_cond;
    struct_obu_19_t out_batch_o_0;
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

void Colle_65(hls::stream<struct_obu_19_t> &i_0,
              hls::stream<struct_sbu_14_t> &o_0) {
    struct_obu_19_t in_batch_i_0;
    struct_sbu_14_t out_batch_o_0;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        uint8_t out_idx;
        out_idx = 0;
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if (in_batch_i_0.data[i].valid && i < in_batch_i_0.end_pos) {
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

void Scatt_151(hls::stream<struct_sbu_22_t> &i_0,
               hls::stream<struct_abu_11_t> &o_0,
               hls::stream<struct_nbu_9_t> &o_1) {
    struct_sbu_22_t in_batch_i_0;
    struct_abu_11_t out_batch_o_0;
    struct_nbu_9_t out_batch_o_1;
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

void Unary_161(hls::stream<struct_nbu_9_t> &i_0,
               hls::stream<struct_abu_11_t> &o_0) {
    struct_nbu_9_t in_batch_i_0;
    struct_abu_11_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].distance;
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

void BinOp_164(hls::stream<struct_abu_11_t> &i_0,
               hls::stream<struct_abu_11_t> &i_1,
               hls::stream<struct_abu_11_t> &o_0) {
    struct_abu_11_t in_batch_i_0;
    struct_abu_11_t in_batch_i_1;
    struct_abu_11_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] =
                (((in_batch_i_0.data[i]) < (in_batch_i_1.data[i])
                      ? in_batch_i_0.data[i]
                      : in_batch_i_1.data[i]));
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

void CopyC_168(hls::stream<struct_nbu_9_t> &i_0,
               hls::stream<struct_nbu_9_t> &o_0,
               hls::stream<struct_nbu_9_t> &o_1) {
    struct_nbu_9_t in_batch_i_0;
    struct_nbu_9_t out_batch_o_0;
    struct_nbu_9_t out_batch_o_1;
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

void Gathe_173(hls::stream<struct_abu_11_t> &i_0,
               hls::stream<struct_nbu_9_t> &i_1,
               hls::stream<struct_sbu_22_t> &o_0) {
    struct_abu_11_t in_batch_i_0;
    struct_nbu_9_t in_batch_i_1;
    struct_sbu_22_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i].ele_0 = in_batch_i_0.data[i];
            out_batch_o_0.data[i].ele_1 = in_batch_i_1.data[i];
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

static void graphyflow_dataflow(hls::stream<struct_ebu_7_t> &i_0_20_stream,
                                hls::stream<struct_sbu_22_t> &o_0_176_stream) {
#pragma HLS DATAFLOW
    hls::stream<struct_kbu_33_t> reduce_141_z2d_pair;
#pragma HLS STREAM variable = reduce_141_z2d_pair depth = 4
    hls::stream<net_wrapper_kt_pair_141_t_t> reduce_141_d2o_pair[8];
#pragma HLS STREAM variable = reduce_141_d2o_pair depth = 4
    hls::stream<net_wrapper_kt_pair_141_t_t> reduce_141_o2u_pair[8];
#pragma HLS STREAM variable = reduce_141_o2u_pair depth = 4
    hls::stream<struct_sbu_22_t> reduce_141_uout_streams;
#pragma HLS STREAM variable = reduce_141_uout_streams depth = 4
    hls::stream<struct_ibu_24_t> reduce_141_intermediate_key;
#pragma HLS STREAM variable = reduce_141_intermediate_key depth = 4
    hls::stream<struct_sbu_22_t> reduce_141_intermediate_transform;
#pragma HLS STREAM variable = reduce_141_intermediate_transform depth = 4
    hls::stream<struct_ebu_7_t> stream_o_0_14;
#pragma HLS STREAM variable = stream_o_0_14 depth = 4
    hls::stream<struct_nbu_9_t> stream_o_0_8;
#pragma HLS STREAM variable = stream_o_0_8 depth = 4
    hls::stream<struct_ebu_7_t> stream_o_1_15;
#pragma HLS STREAM variable = stream_o_1_15 depth = 4
    hls::stream<struct_nbu_9_t> stream_o_0_11;
#pragma HLS STREAM variable = stream_o_0_11 depth = 4
    hls::stream<struct_ebu_7_t> stream_o_0_21;
#pragma HLS STREAM variable = stream_o_0_21 depth = 4
    hls::stream<struct_ebu_7_t> stream_o_1_22;
#pragma HLS STREAM variable = stream_o_1_22 depth = 4
    hls::stream<struct_abu_11_t> stream_o_0_18;
#pragma HLS STREAM variable = stream_o_0_18 depth = 4
    hls::stream<struct_abu_11_t> stream_o_0_25;
#pragma HLS STREAM variable = stream_o_0_25 depth = 4
    hls::stream<struct_sbu_14_t> stream_o_0_31;
#pragma HLS STREAM variable = stream_o_0_31 depth = 4
    hls::stream<struct_sbu_14_t> stream_o_0_59;
#pragma HLS STREAM variable = stream_o_0_59 depth = 4
    hls::stream<struct_sbu_14_t> stream_o_1_60;
#pragma HLS STREAM variable = stream_o_1_60 depth = 4
    hls::stream<struct_abu_11_t> stream_o_2_36;
#pragma HLS STREAM variable = stream_o_2_36 depth = 4
    hls::stream<struct_bbu_16_t> stream_o_0_51;
#pragma HLS STREAM variable = stream_o_0_51 depth = 4
    hls::stream<struct_obu_19_t> stream_o_0_64;
#pragma HLS STREAM variable = stream_o_0_64 depth = 4
    hls::stream<struct_sbu_14_t> stream_o_0_67;
#pragma HLS STREAM variable = stream_o_0_67 depth = 4
    hls::stream<struct_sbu_22_t> stream_o_0_143;
#pragma HLS STREAM variable = stream_o_0_143 depth = 4
    hls::stream<struct_abu_11_t> stream_o_0_153;
#pragma HLS STREAM variable = stream_o_0_153 depth = 4
    hls::stream<struct_nbu_9_t> stream_o_1_154;
#pragma HLS STREAM variable = stream_o_1_154 depth = 4
    hls::stream<struct_nbu_9_t> stream_o_0_170;
#pragma HLS STREAM variable = stream_o_0_170 depth = 4
    hls::stream<struct_abu_11_t> stream_o_0_163;
#pragma HLS STREAM variable = stream_o_0_163 depth = 4
    hls::stream<struct_abu_11_t> stream_o_0_167;
#pragma HLS STREAM variable = stream_o_0_167 depth = 4
    hls::stream<struct_nbu_9_t> stream_o_1_171;
#pragma HLS STREAM variable = stream_o_1_171 depth = 4
    // // printf("DEBUG: --- Starting Execution ---\n");
    // fflush(stdout);

    // --- Function Calls (in topological order) ---
    CopyC_19(i_0_20_stream, stream_o_0_21, stream_o_1_22);
    // // printf("DEBUG: After CopyC_19\n");
    // fflush(stdout);

    CopyC_12(stream_o_0_21, stream_o_0_14, stream_o_1_15);
    // // printf("DEBUG: After CopyC_12\n");
    // fflush(stdout);

    Unary_16(stream_o_1_22, stream_o_0_18);
    // // printf("DEBUG: After Unary_16\n");
    // fflush(stdout);

    Unary_6(stream_o_0_14, stream_o_0_8);
    // // printf("DEBUG: After Unary_6\n");
    // fflush(stdout);

    Unary_9(stream_o_1_15, stream_o_0_11);
    // // printf("DEBUG: After Unary_9\n");
    // fflush(stdout);

    Unary_23(stream_o_0_8, stream_o_0_25);
    // // printf("DEBUG: After Unary_23\n");
    // fflush(stdout);

    Gathe_27(stream_o_0_25, stream_o_0_11, stream_o_0_18, stream_o_0_31);
    // // printf("DEBUG: After Gathe_27\n");
    // fflush(stdout);

    CopyC_57(stream_o_0_31, stream_o_0_59, stream_o_1_60);
    // // printf("DEBUG: After CopyC_57\n");
    // fflush(stdout);

    Scatt_32(stream_o_0_59, stream_o_2_36);
    // // printf("DEBUG: After Scatt_32\n");
    // fflush(stdout);

    BinOp_48(stream_o_2_36, stream_o_0_51);
    // // printf("DEBUG: After BinOp_48\n");
    // fflush(stdout);

    Condi_61(stream_o_1_60, stream_o_0_51, stream_o_0_64);
    // // printf("DEBUG: After Condi_61\n");
    // fflush(stdout);

    Colle_65(stream_o_0_64, stream_o_0_67);
    // // printf("DEBUG: After Colle_65\n");
    // fflush(stdout);

    // --- Start of Reduce Super-Block for Reduc_141 ---
    Reduc_141_pre_process(stream_o_0_67, reduce_141_intermediate_key,
                          reduce_141_intermediate_transform);
    // // printf("DEBUG: After Reduc_141_pre_process\n");
    // fflush(stdout);

    stream_zipper_0(reduce_141_intermediate_key,
                    reduce_141_intermediate_transform, reduce_141_z2d_pair);
    // // printf("DEBUG: After stream_zipper_0\n");
    // fflush(stdout);

    demux_1(reduce_141_z2d_pair, reduce_141_d2o_pair);
    // // printf("DEBUG: After demux_1\n");
    // fflush(stdout);

    omega_switch_2(reduce_141_d2o_pair, reduce_141_o2u_pair);
    // // printf("DEBUG: After omega_switch_2\n");
    // fflush(stdout);

    Reduc_141_unit_reduce(reduce_141_o2u_pair, reduce_141_uout_streams);
    // // printf("DEBUG: After Reduc_141_unit_reduce\n");
    // fflush(stdout);
    // --- End of Reduce Super-Block for Reduc_141 ---

    Scatt_151(reduce_141_uout_streams, stream_o_0_153, stream_o_1_154);
    // // printf("DEBUG: After Scatt_151\n");
    // fflush(stdout);

    CopyC_168(stream_o_1_154, stream_o_0_170, stream_o_1_171);
    // // printf("DEBUG: After CopyC_168\n");
    // fflush(stdout);

    Unary_161(stream_o_0_170, stream_o_0_163);
    // // printf("DEBUG: After Unary_161\n");
    // fflush(stdout);

    BinOp_164(stream_o_0_153, stream_o_0_163, stream_o_0_167);
    // // printf("DEBUG: After BinOp_164\n");
    // fflush(stdout);

    Gathe_173(stream_o_0_167, stream_o_1_171, o_0_176_stream);
    // // printf("DEBUG: After Gathe_173\n");
    // fflush(stdout);

    // // printf("DEBUG: --- Execution Finished Successfully ---\n");
    // fflush(stdout);
}

// Memory-to-Stream Function
static void mem_to_stream_func(const struct_ebu_7_t *in,
                               hls::stream<struct_ebu_7_t> &out_stream,
                               uint16_t num_batches) {
mem_to_stream_loop:
    for (uint16_t i = 0; i < num_batches; ++i) {
#pragma HLS PIPELINE
        out_stream.write(in[i]);
    }
}

// Stream-to-Memory Function
static void stream_to_mem_func(
    hls::stream<struct_sbu_22_t> &in_stream,
    KernelOutputBatch *out) { // <--- 修改1: 类型变为 KernelOutputBatch*
stream_to_mem_loop:
    int i = 0;
    while (true) {
#pragma HLS PIPELINE
        if (!in_stream.empty()) {
            struct_sbu_22_t internal_batch = in_stream.read();
            KernelOutputBatch output_batch; // <--- 修改2: 创建新的简单结构体

            // --- 修改3: 循环转换数据 ---
            for (int k = 0; k < PE_NUM; k++) {
#pragma HLS UNROLL
                output_batch.data[k].distance =
                    (float)internal_batch.data[k]
                        .ele_0; // ap_fixed 自动转为 float
                output_batch.data[k].id = internal_batch.data[k].ele_1.id;
                // printf("stream_to_mem_func: node_id = %d, dist = %.2f\n",
                // output_batch.data[k].id, output_batch.data[k].distance);
            }
            output_batch.end_flag = internal_batch.end_flag;
            output_batch.end_pos = internal_batch.end_pos;

            out[i] = output_batch; // 写入到全局内存

            // printf("DEBUG: --- end pos = %d ---\n", out[i].end_pos);

            if (out[i].end_flag) {
                break;
            }
            i++;
        }
    }
}

extern "C" void graphyflow(const struct_ebu_7_t *i_0_20,
                           KernelOutputBatch *o_0_176, int *stop_flag,
                           uint16_t input_length_in_batches) {
// AXI Interface Pragmas
#pragma HLS INTERFACE m_axi port = i_0_20 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = o_0_176 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = stop_flag offset = slave bundle = gmem2

#pragma HLS INTERFACE s_axilite port = i_0_20
#pragma HLS INTERFACE s_axilite port = o_0_176
#pragma HLS INTERFACE s_axilite port = stop_flag
#pragma HLS INTERFACE s_axilite port = input_length_in_batches
#pragma HLS INTERFACE s_axilite port = return

    // Internal streams for connecting the modules
    static hls::stream<struct_ebu_7_t> i_stream("input_stream");
    static hls::stream<struct_sbu_22_t> o_stream("output_stream");
#pragma HLS STREAM variable = i_stream depth = 4
#pragma HLS STREAM variable = o_stream depth = 4

#pragma HLS DATAFLOW
    // printf("Info: mem_to_stream_func started...\n");
    mem_to_stream_func(i_0_20, i_stream, input_length_in_batches);
    // printf("Info: GraphyFlow Kernel started...\n");
    graphyflow_dataflow(i_stream, o_stream);
    // printf("Info: stream_to_mem_func started...\n");
    stream_to_mem_func(o_stream, o_0_176);
}

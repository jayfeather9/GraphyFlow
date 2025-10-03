#include "graphyflow.h"

// --- Utility Network Functions ---
void stream_zipper_0(hls::stream<struct_ibu_14_t> &in_key_batch_stream,
                     hls::stream<struct_sbu_19_t> &in_transform_batch_stream,
                     hls::stream<struct_kbu_30_t> &out_pair_batch_stream) {
    struct_ibu_14_t key_batch;
    struct_sbu_19_t transform_batch;
    struct_kbu_30_t out_batch;
LOOP_ZIPPER_STREAM_9:
    while (true) {
#pragma HLS PIPELINE
        key_batch = in_key_batch_stream.read();
        transform_batch = in_transform_batch_stream.read();
    LOOP_ZIPPER_UNROLL_13:
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
LOOP_DEMUX_BATCH_26:
    while (true) {
#pragma HLS PIPELINE
        in_batch = in_batch_stream.read();
        net_wrapper_kt_pair_141_t_t wrapper_data;
    LOOP_DEMUX_UNROLL_30:
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
LOOP_DEMUX_PROPAGATE_42:
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
LOOP_SENDER_PIPELINE_56:
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
LOOP_RECEIVER_PIPELINE_120:
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
LOOP_REDUC_PRE_PROCESS_268:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_global_data_0 = i_global_data_0.read();
        in_batch_i_global_data_1 = i_global_data_1.read();
        in_batch_i_global_data_2 = i_global_data_2.read();
        in_batch_i_global_data_3 = i_global_data_3.read();
    LOOP_REDUC_PRE_PROCESS_UNROLL_274:
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
            // ap_fixed<32, 16> data_1_fp =
            //     *reinterpret_cast<ap_fixed<32, 16> *>(
            //         &in_batch_i_global_data_1.data[i]);
            // ap_fixed<32, 16> data_2_fp =
            //     *reinterpret_cast<ap_fixed<32, 16> *>(
            //         &in_batch_i_global_data_2.data[i]);
            // printf("PreProcessor input value1: %.2f, value2: %.2f\n",
            //        (float)data_1_fp,
            //        (float)data_2_fp);
            ap_fixed_pod_t fused_temp_BinOp_104_o_0;
            ap_fixed<32, 16> lhs_104 = *reinterpret_cast<ap_fixed<32, 16> *>(
                &in_batch_i_global_data_2.data[i]);
            ap_fixed<32, 16> rhs_104 = *reinterpret_cast<ap_fixed<32, 16> *>(
                &in_batch_i_global_data_3.data[i]);
            ap_fixed<32, 16> temp_BinOp_104_o_0_ap_result;
            temp_BinOp_104_o_0_ap_result = (lhs_104 + rhs_104);
            fused_temp_BinOp_104_o_0 =
                *reinterpret_cast<int32_t *>(&temp_BinOp_104_o_0_ap_result);
            // Inlining Gathe_215
            transform_out_elem.ele_0 = fused_temp_BinOp_104_o_0;
            transform_out_elem.ele_1 = in_batch_i_global_data_0.data[i];
            // -- End Nested Inline for FusedOp fused_op_221 --
            // -- Inline sub graph end --
            out_batch_intermediate_key.data[i] = key_out_elem;
            out_batch_intermediate_transform.data[i] = transform_out_elem;

            // ap_fixed<32, 16> dist_fp =
            //     *reinterpret_cast<ap_fixed<32, 16> *>(
            //         &transform_out_elem.ele_0);
            // printf("PreProcessor output key: %d, value: %.2f, node: %d\n",
            //        key_out_elem,
            //        (float)dist_fp,
            //        transform_out_elem.ele_1);
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
    
    // **KEY OPTIMIZATION**: Split struct into separate arrays
    // Valid flags use BRAM (lower latency ~0.5-1ns vs URAM 2.189ns)
    bool key_mem_valid[PE_NUM][MAX_NUM >> LOG_PE_NUM];
#pragma HLS dependence variable = key_mem_valid inter false direction = WAW
#pragma HLS dependence variable = key_mem_valid inter false direction = RAW
#pragma HLS BIND_STORAGE variable = key_mem_valid type = RAM_2P impl = BRAM latency=1
#pragma HLS ARRAY_PARTITION variable = key_mem_valid complete dim = 1

    // Data uses URAM (large capacity)
    struct_in_17_t key_mem_data[PE_NUM][MAX_NUM >> LOG_PE_NUM];
#pragma HLS dependence variable = key_mem_data inter false direction = WAW
#pragma HLS dependence variable = key_mem_data inter false direction = RAW
#pragma HLS BIND_STORAGE variable = key_mem_data type = RAM_2P impl = URAM
#pragma HLS ARRAY_PARTITION variable = key_mem_data complete dim = 1

    // Buffers remain the same structure for simplicity
    struct_sb_38_t key_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = key_buffer complete dim = 0
    uint32_t i_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = i_buffer complete dim = 0
    struct_sb_38_t tmp_key_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable = tmp_key_buffer complete dim = 0
    uint32_t tmp_i_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable = tmp_i_buffer complete dim = 0

    // 2. Memory initialization
LOOP_REDUC_UNIT_INIT_PE_361:
    for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
        for (uint32_t i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
            i_buffer[pe][i] = (MAX_NUM + 1);
        }
    }
    
    // Initialize separated memories
LOOP_REDUC_UNIT_INIT_MEM:
    for (uint32_t pe = 0; pe < PE_NUM; pe++) {
        for (uint32_t i = 0; i < (MAX_NUM >> LOG_PE_NUM); i++) {
#pragma HLS PIPELINE II=1
            key_mem_valid[pe][i] = false;
            key_mem_data[pe][i].ele_0 = 0;
            key_mem_data[pe][i].ele_1 = 0;
        }
    }

    // 3. Main processing loop
    bool end_flag;
    bool all_end_flags[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = all_end_flags complete dim = 0
LOOP_REDUC_UNIT_INIT_FLAGS_373:
    for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
        all_end_flags[i] = false;
    }

LOOP_REDUC_UNIT_AGGREGATE_377:
    while (true) {
#pragma HLS PIPELINE II = 1
        net_wrapper_kt_pair_141_t_t kt_elem;
        int32_t key_elem;
        struct_in_17_t transform_elem;
        
    LOOP_REDUC_UNIT_AGGREGATE_PES_382:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if (((!all_end_flags[i]) & (!kt_wrap_item[i].empty()))) {
                kt_elem = kt_wrap_item[i].read();
                if (kt_elem.end_flag) {
                    all_end_flags[i] = kt_elem.end_flag;
                } else {
                    key_elem = kt_elem.data.key >> LOG_PE_NUM;
                    transform_elem = kt_elem.data.transform;
                    
                    // **OPTIMIZED**: Separate reads with different latencies
                    bool old_valid = key_mem_valid[i][key_elem];
                    struct_in_17_t old_data = key_mem_data[i][key_elem];
                    
                LOOP_REDUC_UNIT_SEARCH_BUFFER_401:
                    for (int32_t i_search = L; i_search >= 0; i_search--) {
#pragma HLS UNROLL
                        if ((key_elem == i_buffer[i][i_search])) {
                            old_valid = key_buffer[i][i_search].ele_1;
                            old_data = key_buffer[i][i_search].ele_0;
                            break;
                        }
                    }
                    
                LOOP_REDUC_UNIT_MOVE_BUFFER_407:
                    for (uint32_t i_move = 0; i_move < L; i_move++) {
#pragma HLS UNROLL
                        tmp_i_buffer[i][i_move] = i_buffer[i][i_move + 1];
                        tmp_key_buffer[i][i_move] = key_buffer[i][i_move + 1];
                    }
                    
                LOOP_REDUC_UNIT_UPDATE_BUFFER_412:
                    for (uint32_t i_update = 0; i_update < L; i_update++) {
#pragma HLS UNROLL
                        i_buffer[i][i_update] = tmp_i_buffer[i][i_update];
                        key_buffer[i][i_update] = tmp_key_buffer[i][i_update];
                    }
                    
                    // **OPTIMIZED**: Compute new value
                    struct_in_17_t new_data;
                    bool new_valid = true;
                    
                    if (old_valid) {
                        // Min reduction logic
                        int32_t temp_Scatt_256_o_0 = old_data.ele_0;
                        node_id_t temp_Scatt_256_o_1 = old_data.ele_1;
                        int32_t temp_Scatt_260_o_0 = transform_elem.ele_0;
                        
                        ap_fixed<32, 16> lhs_132 =
                            *reinterpret_cast<ap_fixed<32, 16> *>(&temp_Scatt_256_o_0);
                        ap_fixed<32, 16> rhs_132 =
                            *reinterpret_cast<ap_fixed<32, 16> *>(&temp_Scatt_260_o_0);
                        
                            // printf("Reducer input old: %.2f, new: %.2f, node: %d\n",
                            //        (float)lhs_132,
                            //        (float)rhs_132,
                            //           transform_elem.ele_1);
                        ap_fixed<32, 16> temp_BinOp_132_o_0_ap_result;
                        temp_BinOp_132_o_0_ap_result =
                            (((lhs_132) < (rhs_132) ? lhs_132 : rhs_132));
                        
                            // printf("Reducer output min: %.2f\n",
                                //    (float)temp_BinOp_132_o_0_ap_result);
                                               
                        new_data.ele_0 = *reinterpret_cast<int32_t *>(
                            &temp_BinOp_132_o_0_ap_result);
                        new_data.ele_1 = temp_Scatt_256_o_1;
                    } else {
                        new_data = transform_elem;
                        // printf("Reducer new entry: %.2f, node: %d\n",
                        //        (float)(*reinterpret_cast<ap_fixed<32, 16> *>(
                        //            &transform_elem.ele_0)),
                        //        transform_elem.ele_1);
                    }
                    
                    // **OPTIMIZED**: Separate writes
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
    LOOP_REDUC_UNIT_CHECK_END_450:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            end_flag = (end_flag & all_end_flags[i]);
        }
        if (end_flag) {
            break;
        }
    }
    
    // 4. Final drain loop (修改为使用分离的存储)
    struct_sbu_19_t data_pack;
#pragma HLS ARRAY_PARTITION variable = data_pack.data complete dim = 0
    uint32_t write_positions[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = write_positions complete dim = 0
    struct_in_17_t tmp_data[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = tmp_data complete dim = 0
    bool tmp_data_valid[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = tmp_data_valid complete dim = 0
    
    data_pack.end_flag = false;
    uint32_t k = 0;
    
LOOP_REDUC_UNIT_FINAL_DRAIN_481:
    while ((k < (MAX_NUM >> LOG_PE_NUM))) {
#pragma HLS PIPELINE II = 1
        
        // Load from separated arrays
        for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            tmp_data[pe] = key_mem_data[pe][k];
            tmp_data_valid[pe] = key_mem_valid[pe][k];
        }
        
        // Parallel prefix sum
        uint32_t prefix_sum = 0;
        for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            write_positions[pe] = prefix_sum;
            prefix_sum += (tmp_data_valid[pe] ? 1 : 0);
        }
        uint32_t data_cnt = prefix_sum;

        if (data_cnt == 0) {
            k = k + 1;
            continue;
        }
        // if (data_cnt > 0) {
        //     printf("Reducer final drain round %d, valid count: %d\n", k, data_cnt);
        // }
        // Parallel write
        for (uint32_t pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            if (tmp_data_valid[pe]) {
                data_pack.data[write_positions[pe]] = tmp_data[pe];
                // printf("Reducer final output: %.2f, node: %d\n",
                //        (float)(*reinterpret_cast<ap_fixed<32, 16> *>(
                //            &tmp_data[pe].ele_0)),
                //        tmp_data[pe].ele_1);
            }
        }
        
        k = (k + 1);
        
        data_pack.end_pos = data_cnt;
        o_0.write(data_pack);
    }
    
    // 5. Final batch
    data_pack.end_flag = true;
    data_pack.end_pos = 0;
    o_0.write(data_pack);
}


void Colle_65(hls::stream<struct_obu_10_t> &i_0,
              hls::stream<struct_sbu_12_t> &o_0) {
    struct_obu_10_t in_batch_i_0;
    struct_sbu_12_t out_batch_o_0;
#pragma HLS dependence variable = out_batch_o_0 inter false direction = WAW
#pragma HLS ARRAY_PARTITION variable = out_batch_o_0.data complete dim = 0
LOOP_COLLECT_500:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        uint8_t out_idx = 0;
    LOOP_COLLECT_UNROLL_504:
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
    // printf("Collect finished\n");
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
LOOP_SCATTER_270_520:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_SCATTER_270_UNROLL_524:
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
    /**
     * @brief Pass-through module that casts node_id_t batches to int32_t
     * batches. In the new design, this module no longer accesses memory. It
     * simply performs a data type conversion as required by the downstream
     * reduce component.
     */
    bool end_flag;
LOOP_MEMORY_267_544:
    do {
#pragma HLS PIPELINE
        struct_nbu_16_t in_batch = i_0_node_id.read();
        struct_ibu_14_t out_batch;

        out_batch.end_pos = in_batch.end_pos;
        out_batch.end_flag = in_batch.end_flag;
        end_flag = in_batch.end_flag;

    LOOP_MEMORY_267_UNROLL_553:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // Cast node_id_t (uint16_t) to int32_t
            out_batch.data[i] = static_cast<int32_t>(in_batch.data[i]);
        }
        o_0_node_id.write(out_batch);
    } while (!end_flag);
}

void CopyC_283(hls::stream<struct_nbu_16_t> &i_0,
               hls::stream<struct_nbu_16_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1) {
    struct_nbu_16_t in_batch_i_0;
    struct_nbu_16_t out_batch_o_0;
    struct_nbu_16_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_COPYC_283_562:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_COPYC_283_UNROLL_566:
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
LOOP_CONDI_61_584:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_data = i_data.read();
        in_batch_i_cond = i_cond.read();
    LOOP_CONDI_61_UNROLL_588:
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
                  //   hls::stream<struct_bbu_21_t> &o_0,
                  hls::stream<struct_sbu_12_t> &o_1) {
    struct_ibu_14_t in_batch_i_0;
    struct_nbu_16_t in_batch_i_1;
    struct_ibu_14_t in_batch_i_2;
    // struct_bbu_21_t out_batch_o_0;
    struct_sbu_12_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_FUSED_OP_312_606:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
    LOOP_FUSED_OP_312_UNROLL_612:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // -- Inlining FusedOp fused_op_312 --
            // Inlining Const_37
            // int32_t fused_temp_Const_37_o_0;
            // Inlining CopyC_306
            // int32_t fused_temp_CopyC_306_o_0;
            // int32_t fused_temp_CopyC_306_o_1;
            // fused_temp_CopyC_306_o_0 = in_batch_i_2.data[i];
            // fused_temp_CopyC_306_o_1 = in_batch_i_2.data[i];
            // Inlining BinOp_48
            // ap_fixed<32, 16> lhs_48 = *reinterpret_cast<ap_fixed<32, 16> *>(
            //     &fused_temp_CopyC_306_o_1);
            // out_batch_o_0.data[i] =
            //     (lhs_48 >= ((ap_fixed<32, 16>)(((ap_fixed<32, 16>)0.0))));
            // Inlining Gathe_301
            out_batch_o_1.data[i].ele_0 = in_batch_i_0.data[i];
            out_batch_o_1.data[i].ele_1 = in_batch_i_1.data[i];
            out_batch_o_1.data[i].ele_2 = in_batch_i_2.data[i];
            // -- End Inlining FusedOp fused_op_312 --
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        // out_batch_o_0.end_flag = end_flag;
        // out_batch_o_0.end_pos = end_pos;
        // o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        if (end_flag) {
            break;
        }
    }
}

void Memor_318(hls::stream<struct_ibu_14_t> &o_0_edge_weight,
               hls::stream<struct_ibu_14_t> &o_0_edge_src_distance,
               hls::stream<struct_nbu_16_t> &o_0_edge_dst,
               hls::stream<edge_batch_t> &response_from_umc) {
    /**
     * @brief Unpacks edge data batches from the UMC.
     * In this version, it's a pure consumer with no backpressure mechanism.
     */
    bool done = false;
LOOP_MEMORY_318_641:
    do {
#pragma HLS PIPELINE
        edge_batch_t in_batch = response_from_umc.read();
        done = in_batch.end_flag;

        struct_ibu_14_t weight_batch;
        struct_ibu_14_t src_dist_batch;
        struct_nbu_16_t dst_id_batch;

        weight_batch.end_pos = in_batch.end_pos;
        src_dist_batch.end_pos = in_batch.end_pos;
        dst_id_batch.end_pos = in_batch.end_pos;

        weight_batch.end_flag = done;
        src_dist_batch.end_flag = done;
        dst_id_batch.end_flag = done;

    LOOP_MEMORY_318_UNROLL_660:
        for (int i = 0; i < PE_NUM; ++i) {
#pragma HLS UNROLL
            weight_batch.data[i] = in_batch.weights[i];
            src_dist_batch.data[i] = in_batch.src_distances[i];
            dst_id_batch.data[i] = in_batch.dst_ids[i];
            // printf("Edge weight: %.2f, src dist: %.2f, dst id: %d\n",
            //        (float)reinterpret_cast<ap_fixed<32, 16> *>(
            //            &in_batch.weights[i])[0],
            //        (float)reinterpret_cast<ap_fixed<32, 16> *>(
            //            &in_batch.src_distances[i])[0],
            //        in_batch.dst_ids[i]);
        }

        o_0_edge_weight.write(weight_batch);
        o_0_edge_src_distance.write(src_dist_batch);
        o_0_edge_dst.write(dst_id_batch);
    } while (!done);
}

void CopyC_350(hls::stream<struct_nbu_16_t> &i_0,
               hls::stream<struct_nbu_16_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1) {
    struct_nbu_16_t in_batch_i_0;
    struct_nbu_16_t out_batch_o_0;
    struct_nbu_16_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_COPYC_350_674:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_COPYC_350_UNROLL_678:
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
               hls::stream<struct_nbu_16_t> &i_0_node_id,
               hls::stream<struct_ibu_14_t> &all_node_distances_from_umc) {
    /**
     * @brief Efficiently filters a stream of all node distances against a
     * stream of requested node IDs.
     * Assumes both input streams are sorted by node ID.
     */

    struct_nbu_16_t in_node_id_batch;
    struct_ibu_14_t in_dist_batch;
    struct_ibu_14_t out_dist_batch;
#pragma HLS ARRAY_PARTITION variable = in_node_id_batch.data complete dim = 0
#pragma HLS ARRAY_PARTITION variable = in_dist_batch.data complete dim = 0
#pragma HLS ARRAY_PARTITION variable = out_dist_batch.data complete dim = 0

    out_dist_batch.end_flag = false;

    // Initial reads
    in_node_id_batch = i_0_node_id.read();
    in_dist_batch = all_node_distances_from_umc.read();

    uint32_t in_node_base_id = 0;
    uint32_t id_idx = 0;
    
    // **CRITICAL OPTIMIZATION**: Use end_id instead of upper_bound to reduce arithmetic
    // This changes: base + len - 1 (2 ops) → base + len (1 op)
    uint32_t in_node_end_id = in_dist_batch.end_pos;
#pragma HLS BIND_STORAGE variable=in_node_end_id type=register impl=srl

LOOP_MEMORY_343_FILTER_715:
    while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS expression_balance
        
        uint32_t current_batch_len = in_node_id_batch.end_pos;
        if (current_batch_len == 0 || id_idx >= current_batch_len) {
            if (id_idx > 0){
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
        
        // **OPTIMIZED**: Use >= with end_id instead of < with upper_bound
        // Equivalent logic: (target >= end) ⇔ (target > end-1) ⇔ (upper_bound < target)
        if (target_node_id >= in_node_end_id) {
            // Update base to current end position
            in_node_base_id = in_node_end_id;
            in_dist_batch = all_node_distances_from_umc.read();
            
            // **OPTIMIZED**: Single add operation (reduced from base + len - 1)
            uint32_t batch_len = in_dist_batch.end_pos;
            in_node_end_id = in_node_base_id + batch_len;
#pragma HLS BIND_OP variable=in_node_end_id op=add impl=fabric latency=0
            continue;
        }
        
        // Index calculation remains the same
        out_dist_batch.data[id_idx] = in_dist_batch.data[target_node_id - in_node_base_id];
        id_idx++;
    }

    // Send the final output batch with end_flag
    struct_ibu_14_t final_batch;
    final_batch.end_flag = true;
    final_batch.end_pos = 0;
    o_0_node_distance.write(final_batch);

    while (in_dist_batch.end_flag == false) {
        in_dist_batch = all_node_distances_from_umc.read();
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
LOOP_FUSED_OP_338_771:
    while (true) {
#pragma HLS PIPELINE
        // printf("DEBUG: Reading new batch\n");
        // fflush(stdout);
        in_batch_i_0 = i_0.read();
        // printf("DEBUG: Read batch with end_flag=%d, end_pos=%d\n",
        //        in_batch_i_0.end_flag, in_batch_i_0.end_pos);
        in_batch_i_1 = i_1.read();
        // printf("DEBUG: Read batch with end_flag=%d, end_pos=%d\n",
        //        in_batch_i_1.end_flag, in_batch_i_1.end_pos);
        in_batch_i_2 = i_2.read();
        // printf("DEBUG: Read batch with end_flag=%d, end_pos=%d\n",
        //        in_batch_i_2.end_flag, in_batch_i_2.end_pos);
        // fflush(stdout);
    LOOP_FUSED_OP_338_UNROLL_777:
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
            // printf("DEBUG: node_id=%d lhs_164=%.2f, rhs_164=%.2f, min=%.2f\n",
            //     in_batch_i_2.data[i],
            // (float)lhs_164, (float)rhs_164,
            // (float)temp_BinOp_164_o_0_ap_result);
            // fflush(stdout);
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
        // printf("DEBUG: Writing batch with end_flag=%d, end_pos=%d\n",
        //        out_batch_o_0.end_flag, out_batch_o_0.end_pos);
        // fflush(stdout);
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
LOOP_SCATTER_346_810:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_SCATTER_346_UNROLL_814:
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

// --- PHASE 2.1: UMC Sub-module: Node Property Loader ---

/**
 * @brief Loads all node distances from DDR into a URAM cache.
 * It reads data in wide AXI bus bursts for efficiency.
 *
 * @param node_distances_ddr  DDR pointer for node distances.
 * @param node_distance_cache On-chip URAM cache for node distances
 * (write-only).
 * @param num_nodes           Total number of nodes.
 * @param load_finished_signal Stream to signal completion to the edge loader.
 */
static void
node_property_loader(const int *node_distances_ddr,
                     hls::stream<node_distance_burst_t> &node_distance_burst_stream_0,
                     hls::stream<node_distance_burst_t> &node_distance_burst_stream_1,
                     int num_nodes) {
    // Use ap_uint for wide bus access
    const ap_uint<AXI_BUS_WIDTH> *wide_bus_ptr =
        reinterpret_cast<const ap_uint<AXI_BUS_WIDTH> *>(node_distances_ddr);
    const int num_wide_reads =
        (num_nodes + NUM_WORDS_PER_BUS - 1) / NUM_WORDS_PER_BUS;
    int sent_pack_cnt = 0;
    int total_pack_cnt = (num_nodes + PE_NUM - 1) / PE_NUM;

LOAD_NODES_LOOP:
LOOP_NODE_PROP_LOADER_848:
    for (int i = 0; i < num_wide_reads; ++i) {
#pragma HLS PIPELINE II = 1
        ap_uint<AXI_BUS_WIDTH> wide_word = wide_bus_ptr[i];
        node_distance_burst_t burst;
        int batch_idx = i * NUM_WORDS_PER_BUS / PE_NUM;
    UNPACK_NODES_LOOP:
    LOOP_NODE_PROP_LOADER_UNPACK_852:
        for (int j = 0; j < NUM_WORDS_PER_BUS; j += PE_NUM) {
#pragma HLS UNROLL
            for (int pe = 0; pe < PE_NUM; ++pe) {
#pragma HLS UNROLL
                int node_idx = i * NUM_WORDS_PER_BUS + j + pe;
                if (node_idx < num_nodes) {
                    burst.data[pe] = wide_word.range((j + pe + 1) * DATA_TYPE_WIDTH - 1,
                                                     (j + pe) * DATA_TYPE_WIDTH);
                }
            }
            if (sent_pack_cnt < total_pack_cnt) {
                node_distance_burst_stream_0.write(burst);
                // node_distance_burst_stream_1.write(burst);
                sent_pack_cnt++;
            }
        }
    }

    sent_pack_cnt = 0;
    LOAD_NODES_LOOP_2:
LOOP_NODE_PROP_LOADER_848_2:
    for (int i = 0; i < num_wide_reads; ++i) {
#pragma HLS PIPELINE II = 1
        ap_uint<AXI_BUS_WIDTH> wide_word = wide_bus_ptr[i];
        node_distance_burst_t burst;
        int batch_idx = i * NUM_WORDS_PER_BUS / PE_NUM;
    UNPACK_NODES_LOOP_2:
    LOOP_NODE_PROP_LOADER_UNPACK_852_2:
        for (int j = 0; j < NUM_WORDS_PER_BUS; j += PE_NUM) {
#pragma HLS UNROLL
            for (int pe = 0; pe < PE_NUM; ++pe) {
#pragma HLS UNROLL
                int node_idx = i * NUM_WORDS_PER_BUS + j + pe;
                if (node_idx < num_nodes) {
                    burst.data[pe] = wide_word.range((j + pe + 1) * DATA_TYPE_WIDTH - 1,
                                                     (j + pe) * DATA_TYPE_WIDTH);
                }
            }
            if (sent_pack_cnt < total_pack_cnt) {
                // node_distance_burst_stream_0.write(burst);
                node_distance_burst_stream_1.write(burst);
                sent_pack_cnt++;
            }
        }
    }

}

// --- PHASE 2.2: UMC Sub-module: Edge Loader and Dispatcher ---

// edge_prop_loader
/**
 * @brief Load edge properties from DDR and dispatch to downstream modules.
 *
 * @param edge_descriptors DDR pointer for edge descriptors.
 * @param edge_stream     Stream to output edge descriptors.
 * @param num_edges       Total number of edges.
 */
static void
edge_descriptor_loader(const edge_des_burst_t *edge_des_bursts,
                       hls::stream<edge_descriptor_batch_t> &edge_stream,
                       int num_edges) {
#pragma HLS dependence variable = edge_des_bursts inter false
    // read 8 edges at one time and push
    const int num_batches = (num_edges + PE_NUM - 1) / PE_NUM;
LOAD_EDGES_LOOP:
LOOP_EDGE_DESC_LOADER_872:
    for (int i = 0; i < num_batches; ++i) {
#pragma HLS PIPELINE II = 1
        edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
#pragma HLS dependence variable = edge_batch inter false direction = WAW
        edge_des_burst_t burst = edge_des_bursts[i];
        edge_batch.end_pos = 0;
    LOAD_EDGE_BATCH_LOOP:
    LOOP_EDGE_DESC_LOADER_LOAD_BATCH_877:
        for (int j = 0; j < PE_NUM; ++j) {
#pragma HLS UNROLL
            int edge_idx = i * PE_NUM + j;
            if (edge_idx < num_edges) {
                edge_batch.edges[edge_batch.end_pos] = burst.edges[j];
                edge_batch.end_pos++;
            }
        }
        edge_stream.write(edge_batch);
    }
}

static void src_offset_loader(const int *src_offsets_ddr,
                               hls::stream<int> &src_offsets_stream,
                               int num_nodes) {
    const ap_uint<AXI_BUS_WIDTH> *wide_bus_ptr =
        reinterpret_cast<const ap_uint<AXI_BUS_WIDTH> *>(src_offsets_ddr);
    const int num_wide_reads =
        (num_nodes + 1 + NUM_WORDS_PER_BUS - 1) / NUM_WORDS_PER_BUS;
#pragma HLS dependence variable = wide_bus_ptr inter false

    // read src offsets and push to stream
LOAD_SRC_OFFSETS_LOOP:
LOOP_SRC_OFFSET_LOADER_881:
    for (int i = 0; i <= num_wide_reads; ++i) {
#pragma HLS PIPELINE II = 1
        ap_uint<AXI_BUS_WIDTH> wide_word = wide_bus_ptr[i];
        // src_offsets_stream.write(src_offsets_ddr[i]);
        for (int j = 0; j < NUM_WORDS_PER_BUS; j++) {
#pragma HLS UNROLL
            int node_idx = i * NUM_WORDS_PER_BUS + j;
            int cur_data;
            if (node_idx <= num_nodes) {
                cur_data = wide_word.range((j + 1) * DATA_TYPE_WIDTH - 1,
                                           j * DATA_TYPE_WIDTH);
                src_offsets_stream.write(cur_data);
            }
        }
    }
}

/**
 * @brief Reads CSR graph data, fetches corresponding source node distances from
 * the URAM cache, and serves batches of processed edges to downstream modules.
 */
static void edge_property_loader_and_dispatcher(
    hls::stream<int> &src_offsets_cache_stream,
    hls::stream<edge_descriptor_batch_t> &edge_stream,
    hls::stream<node_distance_burst_t> &node_distance_burst_stream,
    int num_nodes,
    hls::stream<edge_batch_t> &response_stream) {
    // // --- PHASE A: Wait for node properties to be fully cached on-chip ---
    // (void)load_finished_signal.read();
    // --- PHASE B: Cache src_offsets on-chip for fast access ---
//     int src_offsets_cache[MAX_NUM + 1];
// #pragma HLS BIND_STORAGE variable = src_offsets_cache type = RAM_1P impl = BRAM

// CACHE_OFFSETS_LOOP:
// LOOP_EDGE_PROP_LOADER_CACHE_OFFSETS_884:
//     for (int i = 0; i <= num_nodes; ++i) {
// #pragma HLS PIPELINE II = 1
//         // src_offsets_cache[i] = src_offsets_ddr[i];
//         src_offsets_cache_stream.write(src_offsets_ddr[i]);
//     }

    // --- PHASE C: Process and dispatch all edges unconditionally ---
    edge_batch_t current_batch;
#pragma HLS ARRAY_PARTITION variable = current_batch.weights complete dim = 0
#pragma HLS ARRAY_PARTITION variable =                                         \
    current_batch.src_distances complete dim = 0
#pragma HLS ARRAY_PARTITION variable = current_batch.dst_ids complete dim = 0
#pragma HLS dependence variable = current_batch inter false direction = WAW
    current_batch.end_pos = 0;
    current_batch.end_flag = false;

    edge_descriptor_batch_t edge_batch;
    edge_batch.end_pos = 0;
    int edge_batch_pos = 0;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
#pragma HLS dependence variable = edge_batch inter false

    node_distance_burst_t node_distance_burst;
#pragma HLS ARRAY_PARTITION variable = node_distance_burst.data complete dim = 0
#pragma HLS dependence variable = node_distance_burst inter false

    int start_edge_idx, end_edge_idx;
    start_edge_idx = src_offsets_cache_stream.read();

    int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
LOOP_EDGE_PROP_LOADER_MAX_BURST_891:
    for (int node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         ++node_burst_idx) {
        node_distance_burst = node_distance_burst_stream.read();
        int32_t base_idx = node_burst_idx << LOG_PE_NUM;
    LOOP_EDGE_PROP_LOADER_PROCESS_NODES_894:
        for (int pe_idx = 0; pe_idx < PE_NUM; ++pe_idx) {
            int32_t u = base_idx + pe_idx;
            if (u >= num_nodes) {
                break;
            }
            int32_t src_dist = node_distance_burst.data[pe_idx];
            // int start_edge_idx = src_offsets_cache[u];
            // int end_edge_idx = src_offsets_cache[u + 1];
            end_edge_idx = src_offsets_cache_stream.read();

        LOOP_EDGE_PROP_LOADER_PROCESS_EDGES_900:
            for (int e_idx = start_edge_idx; e_idx < end_edge_idx; ++e_idx) {
    #pragma HLS PIPELINE II = 1
                if (edge_batch_pos == edge_batch.end_pos) {
                    edge_batch = edge_stream.read();
                    edge_batch_pos = 0;
                }
                edge_descriptor_t edge = edge_batch.edges[edge_batch_pos++];
                int batch_idx = current_batch.end_pos;
                current_batch.weights[batch_idx] = edge.weight;
                current_batch.src_distances[batch_idx] = src_dist;
                current_batch.dst_ids[batch_idx] = edge.dst_id;
                current_batch.end_pos++;

                // ap_fixed<32, 16> src_dist_fp = *reinterpret_cast<ap_fixed<32, 16>
                // *>(&src_dist); ap_fixed<32, 16> weight_fp =
                // *reinterpret_cast<ap_fixed<32, 16> *>(&edge.weight);

                // printf("DEBUG: Edge (src=%d, dst=%d, weight=%.2f) with src_dist=%.2f\n",
                //        u, edge.dst_id, (float)weight_fp, (float)src_dist_fp);

                if (current_batch.end_pos == PE_NUM) {
                    response_stream.write(current_batch);
                    current_batch.end_pos = 0;
                }
            }

            start_edge_idx = end_edge_idx;
        }
    }

    // --- PHASE D: Send the final batch (can be partial) with end flag ---
    current_batch.end_flag = true;
    response_stream.write(current_batch);
}

// --- PHASE 2.3: UMC Sub-module: Node Property Responder ---

/**
 * @brief Responds to requests for node distances from the URAM cache.
 *
 * @param node_distance_cache On-chip URAM cache (read-only).
 * @param num_nodes           Total number of nodes.
 * @param response_stream     Stream of batched node distance responses.
 */
static void
node_property_responder(hls::stream<node_distance_burst_t> &node_distance_burst_stream,
                        int num_nodes,
                        hls::stream<struct_ibu_14_t> &all_distances_stream) {
    /**
     * @brief Proactively broadcasts all node distances in ascending order of
     * node ID.
     */
    struct_ibu_14_t dist_batch;
    dist_batch.end_pos = 0;
    dist_batch.end_flag = false;

// BROADCAST_LOOP:
// LOOP_NODE_PROP_RESPONDER_BROADCAST_936:
//     for (int i = 0; i < num_nodes; i++) {
// #pragma HLS PIPELINE II = 1
//         dist_batch.data[dist_batch.end_pos] = node_distance_cache[i];
//         dist_batch.end_pos++;
//         if (dist_batch.end_pos == PE_NUM) {
//             all_distances_stream.write(dist_batch);
//             dist_batch.end_pos = 0;
//         }
//     }
    int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
LOOP_NODE_PROP_RESPONDER_MAX_BURST_940:
    for (int node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         ++node_burst_idx) {
#pragma HLS PIPELINE II = 1
        node_distance_burst_t node_distance_burst = node_distance_burst_stream.read();
        int32_t base_idx = node_burst_idx << LOG_PE_NUM;
    LOOP_NODE_PROP_RESPONDER_PROCESS_NODES_943:
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; ++pe_idx) {
#pragma HLS UNROLL
            int32_t node_idx = base_idx + pe_idx;
            if (node_idx >= num_nodes) {
                break;
            }
            dist_batch.data[dist_batch.end_pos] = node_distance_burst.data[pe_idx];
            dist_batch.end_pos++;
        }
        all_distances_stream.write(dist_batch);
        dist_batch.end_pos = 0;
    }

    // Send the final batch (can be partial) with the end flag set.
    dist_batch.end_flag = true;
    all_distances_stream.write(dist_batch);
}

// --- PHASE 2.4: UMC Top-Level Dataflow Function ---

void UnifiedMemoryController(
    const int *src_offsets, const edge_des_burst_t *edge_des_bursts,
    const int *node_distances, int num_nodes, int num_edges,
    hls::stream<edge_batch_t> &response_to_318,
    hls::stream<struct_ibu_14_t> &all_node_distances_to_343) {
#pragma HLS DATAFLOW

    // On-chip caches
//     static int32_t node_distance_cache_for_edge_loader[MAX_NUM];
// #pragma HLS BIND_STORAGE variable = node_distance_cache_for_edge_loader type = \
//     RAM_T2P impl = URAM
//     static int32_t node_distance_cache_for_responder[MAX_NUM];
// #pragma HLS BIND_STORAGE variable = node_distance_cache_for_responder type =   \
//     RAM_T2P impl = URAM
    hls::stream<node_distance_burst_t> node_distance_burst_stream_0;
#pragma HLS STREAM variable = node_distance_burst_stream_0 depth = 12
    hls::stream<node_distance_burst_t> node_distance_burst_stream_1;
#pragma HLS STREAM variable = node_distance_burst_stream_1 depth = 12

    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 12

    hls::stream<int> src_offsets_cache_stream;
#pragma HLS STREAM variable = src_offsets_cache_stream depth = 32

    // Internal Signal Stream
//     static hls::stream<bool> node_loader_finished;
// #pragma HLS STREAM variable = node_loader_finished depth = 2

    src_offset_loader(src_offsets, src_offsets_cache_stream, num_nodes);

    node_property_loader(node_distances, node_distance_burst_stream_0,
                         node_distance_burst_stream_1, num_nodes);

    edge_descriptor_loader(edge_des_bursts, edge_stream, num_edges);

    edge_property_loader_and_dispatcher(
        src_offsets_cache_stream, edge_stream, node_distance_burst_stream_0,
        num_nodes, response_to_318);

    node_property_responder(node_distance_burst_stream_1, num_nodes,
                            all_node_distances_to_343);
}

// static void
// mem_to_stream_func(const struct_ebu_4_t *in_i_0_edge_id_320,
//                    hls::stream<struct_ebu_4_t> &out_i_0_edge_id_320_stream,
//                    uint16_t num_batches) {
// LOOP_MEM_TO_STREAM_1000:
//     for (uint32_t i = 0; i < num_batches; i++) {
// #pragma HLS PIPELINE
//         out_i_0_edge_id_320_stream.write(in_i_0_edge_id_320[i]);
//     }
// }

// static void stream_to_mem_func(hls::stream<struct_sbu_19_t> &in_o_0_342_stream,
//                                KernelOutputBatch *out_o_0_342) {
//     int32_t i = 0;
// LOOP_STREAM_TO_MEM_1008:
//     while (true) {
// #pragma HLS PIPELINE
//         struct_sbu_19_t internal_batch;
//         internal_batch = in_o_0_342_stream.read();
//         KernelOutputBatch output_batch;
//     LOOP_STREAM_TO_MEM_UNROLL_1013:
//         for (uint32_t k = 0; k < PE_NUM; k++) {
// #pragma HLS UNROLL
//             ap_fixed<32, 16> final_dist_fp;
//             final_dist_fp = *reinterpret_cast<ap_fixed<32, 16> *>(
//                 &internal_batch.data[k].ele_0);
//             output_batch.data[k].distance = (float)final_dist_fp;
//             output_batch.data[k].id = internal_batch.data[k].ele_1;
//         }
//         output_batch.end_flag = internal_batch.end_flag;
//         output_batch.end_pos = internal_batch.end_pos;
//         out_o_0_342[i] = output_batch;
//         if (out_o_0_342[i].end_flag) {
//             break;
//         }
//         i = (i + 1);
//     }
// }

static void final_convert(
    hls::stream<struct_sbu_19_t> &in_stream,
    hls::stream<KernelOutputBatch> &converted_stream) {
LOOP_FINAL_CONVERT_1033:
    while (true) {
#pragma HLS PIPELINE II=1
        struct_sbu_19_t in_batch = in_stream.read();
        KernelOutputBatch out_batch;
        out_batch.end_flag = in_batch.end_flag;
        out_batch.end_pos = in_batch.end_pos;
    LOOP_FINAL_CONVERT_UNROLL_1038:
        for (int i = 0; i < PE_NUM; ++i) {
#pragma HLS UNROLL
            ap_fixed<32, 16> dist_fp =
                *reinterpret_cast<ap_fixed<32, 16> *>(&in_batch.data[i].ele_0);
            out_batch.data[i].distance = (float)dist_fp;
            out_batch.data[i].id = in_batch.data[i].ele_1;
        }
        converted_stream.write(out_batch);
        if (in_batch.end_flag) {
            break;
        }
    }
}

static void final_write(
    hls::stream<KernelOutputBatch> &converted_stream,
    KernelOutputBatch *out_o_0_342) {
    #pragma HLS dependence variable=out_o_0_342 inter false
    int32_t i = 0;
LOOP_FINAL_WRITE_1055:
    while (true) {
#pragma HLS PIPELINE II=1
        KernelOutputBatch out_batch = converted_stream.read();
        out_o_0_342[i] = out_batch;
        if (out_batch.end_flag) {
            break;
        }
        i = (i + 1);
    }
}

static void graphyflow_dataflow(
    // UMC inputs from DDR
    const int *src_offsets, const edge_des_burst_t *edge_des_bursts,
    const int *node_distances, int num_nodes, int num_edges,
    // Final output stream
    hls::stream<struct_sbu_19_t> &o_0_342_stream) {
#pragma HLS DATAFLOW

    // --- PHASE 1: Streams for UMC Communication ---
    // Streams for edge data requests (Memor_318 <-> UMC)
    static hls::stream<edge_batch_t> umc_edge_resp_stream;
#pragma HLS STREAM variable = umc_edge_resp_stream depth = 4
    static hls::stream<struct_ibu_14_t> umc_all_node_distances_stream;
#pragma HLS STREAM variable = umc_all_node_distances_stream depth = 4

    // --- PHASE 2: Instantiate the Unified Memory Controller ---
    UnifiedMemoryController(src_offsets, edge_des_bursts, node_distances,
                            num_nodes, num_edges, umc_edge_resp_stream,
                            umc_all_node_distances_stream);

    // --- PHASE 3: Instantiate the original DFIR dataflow graph ---
    // Note: The original stream declarations are preserved, but the ones
    // related to the old memory inputs are no longer fed by a memory reader.
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

    // --- PHASE 4: Connect all components in topological order ---

    // Memor_318 now initiates the pipeline by communicating with the UMC
    Memor_318(stream_o_0_edge_weight_319, stream_o_0_edge_src_distance_321,
              stream_o_0_edge_dst_322, umc_edge_resp_stream);

    // The rest of the original dataflow graph connects as before
    fused_op_312(stream_o_0_edge_src_distance_321, stream_o_0_edge_dst_322,
                 stream_o_0_edge_weight_319, stream_o_1_317);
    // Condi_61(stream_o_1_317, stream_o_0_316, stream_o_0_64);
    // Colle_65(stream_o_0_64, stream_o_0_67);
    Scatt_270(stream_o_1_317, stream_o_0_272, stream_o_1_273, stream_o_2_274);
    CopyC_283(stream_o_1_273, stream_o_0_285, stream_o_1_286);

    // Memor_267 is now a simple pass-through/type-caster
    Memor_267(stream_o_0_node_id_268, stream_o_1_286);

    Reduc_141_pre_process(stream_o_0_node_id_268, stream_o_0_285,
                          stream_o_0_272, stream_o_2_274, intermediate_key,
                          intermediate_transform);
    stream_zipper_0(intermediate_key, intermediate_transform,
                    reduce_141_z2d_pair);
    demux_1(reduce_141_z2d_pair, reduce_141_d2o_pair);
    omega_switch_2(reduce_141_d2o_pair, reduce_141_o2u_pair);
    Reduc_141_unit_reduce(reduce_141_o2u_pair, stream_o_0_143);

    Scatt_346(stream_o_0_143, stream_o_0_348, stream_o_1_349);
    CopyC_350(stream_o_1_349, stream_o_0_352, stream_o_1_353);

    // printf("DEBUG: Entering Memor_343 as UMC proxy...\n");
    // fflush(stdout);
    // Memor_343 now acts as a proxy to the UMC
    Memor_343(stream_o_0_node_distance_344, stream_o_1_353,
              umc_all_node_distances_stream);

    // printf("DEBUG: Exited Memor_343 UMC proxy.\n");
    // fflush(stdout);
    // The final operation which writes to the output stream
    fused_op_338(stream_o_0_348, stream_o_0_node_distance_344, stream_o_0_352,
                 o_0_342_stream);
    // printf("DEBUG: Exited fused_op_338.\n");
    // fflush(stdout);
}

/**
 * @brief Top-level kernel function for the graph processing accelerator.
 * * This function orchestrates the dataflow between global memory and the
 * processing elements. It now uses a CSR graph representation.
 * * @param src_offsets       Pointer to the CSR offsets array for source nodes.
 * @param edge_descriptors  Pointer to the array of edge data (destination and
 * weight).
 * @param node_distances    Pointer to the array of node distances (readable and
 * writable).
 * @param num_nodes         The total number of nodes in the graph.
 */
extern "C" void graphyflow(
    const int *src_offsets, const edge_des_burst_t *edge_des_bursts,
    int *node_distances, int num_nodes, int num_edges,
    // Note: The original output 'o_0_342' is still needed for the final
    // results, but its role inside the kernel logic needs clarification.
    // Assuming it is an output for a different data path, we will leave it for
    // now. Let's assume the Bellman-Ford output is now written back via the
    // 'node_distances' pointer by the host. The kernel's output stream
    // 'o_0_342' will produce the results as per the dataflow logic.
    KernelOutputBatch *o_0_342) {
#pragma HLS INTERFACE m_axi port = src_offsets offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = edge_des_bursts offset = slave bundle =    \
    gmem1
#pragma HLS INTERFACE m_axi port = node_distances offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = o_0_342 offset = slave bundle = gmem3

#pragma HLS INTERFACE s_axilite port = src_offsets
#pragma HLS INTERFACE s_axilite port = edge_des_bursts
#pragma HLS INTERFACE s_axilite port = node_distances
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = o_0_342
#pragma HLS INTERFACE s_axilite port = return

    // This stream connects the dataflow output to the memory writer.
    static hls::stream<struct_sbu_19_t> o_0_342_internal_stream;
#pragma HLS STREAM variable = o_0_342_internal_stream depth = 32

#pragma HLS DATAFLOW

    // The main dataflow function, now driven by the new CSR inputs.
    graphyflow_dataflow(src_offsets, edge_des_bursts, node_distances,
                        num_nodes, num_edges, o_0_342_internal_stream);

    // Writes the final result from the dataflow to global memory.
    // stream_to_mem_func(o_0_342_internal_stream, o_0_342);
    hls::stream<KernelOutputBatch> converted_stream;
#pragma HLS STREAM variable = converted_stream depth = 32
    final_convert(o_0_342_internal_stream, converted_stream);
    final_write(converted_stream, o_0_342);
}

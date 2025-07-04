#include "graphyflow.h"

using namespace hls;

static void CopyC_19(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_edge__t> &o_0,
    stream<outer_basic_edge__t> &o_1,
    uint16_t input_length
);

static void CopyC_12(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_edge__t> &o_0,
    stream<outer_basic_edge__t> &o_1,
    uint16_t input_length
);

static void Unary_16(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
);

static void Unary_6(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_node__t> &o_0,
    uint16_t input_length
);

static void Unary_9(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_node__t> &o_0,
    uint16_t input_length
);

static void Unary_23(
    stream<outer_basic_node__t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
);

static void Gathe_27(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_node__t> &i_1,
    stream<outer_basic_ap_fi_t> &i_2,
    stream<outer_tuple_bbb_1_t> &o_0,
    uint16_t input_length
);

static void CopyC_57(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_tuple_bbb_1_t> &o_0,
    stream<outer_tuple_bbb_1_t> &o_1,
    uint16_t input_length
);

static void Scatt_32(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_2,
    uint16_t input_length
);

static void BinOp_48(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_bool_t> &o_0,
    uint16_t input_length
);

static void Condi_61(
    stream<outer_tuple_bbb_1_t> &i_data,
    stream<outer_basic_bool_t> &i_cond,
    stream<outer_opt__of_tup_t> &o_0,
    uint16_t input_length
);

static void Colle_65(
    stream<outer_opt__of_tup_t> &i_0,
    stream<outer_tuple_bbb_1_t> &o_0,
    uint16_t input_length
);

static void Scatt_87(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    stream<outer_basic_node__t> &o_1,
    stream<outer_basic_ap_fi_t> &o_2,
    uint16_t input_length
);

static void CopyC_122(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_tuple_bb_2_t> &o_0,
    stream<outer_tuple_bb_2_t> &o_1,
    uint16_t input_length
);

static void Unary_126(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
);

static void Scatt_68(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_basic_node__t> &o_1,
    uint16_t input_length
);

static void Scatt_148(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    stream<outer_basic_node__t> &o_1,
    uint16_t input_length
);

static void BinOp_101(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_ap_fi_t> &i_1,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
);

static void Unary_116(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
);

static void Unary_119(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_basic_node__t> &o_0,
    uint16_t input_length
);

static void CopyC_165(
    stream<outer_basic_node__t> &i_0,
    stream<outer_basic_node__t> &o_0,
    stream<outer_basic_node__t> &o_1,
    uint16_t input_length
);

static void Gathe_105(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_node__t> &i_1,
    stream<outer_tuple_bb_2_t> &o_0,
    uint16_t input_length
);

static void BinOp_129(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_ap_fi_t> &i_1,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
);

static void Unary_158(
    stream<outer_basic_node__t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
);

static void Gathe_133(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_node__t> &i_1,
    stream<outer_tuple_bb_2_t> &o_0,
    uint16_t input_length
);

static void BinOp_161(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_ap_fi_t> &i_1,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
);

static void Gathe_170(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_node__t> &i_1,
    stream<outer_tuple_bb_2_t> &o_0,
    uint16_t input_length
);

static void Reduc_138_key_sub_func(
    stream<outer_tuple_bbb_1_t> &i_0_69,
    stream<outer_basic_node__t> &o_1_71,
    uint16_t input_length
) {
    uint16_t Scatt_68_input_len = input_length;
    Scatt_68(
        i_0_69,
        o_1_71,
        Scatt_68_input_len
    );
}


static void Reduc_138_transform_sub_func(
    stream<outer_tuple_bbb_1_t> &i_0_88,
    stream<outer_tuple_bb_2_t> &o_0_108,
    uint16_t input_length
) {
    uint16_t Scatt_87_input_len = input_length;
    stream<outer_basic_ap_fi_t> Scatt_87_o_0_89;
    #pragma HLS STREAM variable=Scatt_87_o_0_89 depth=4
    stream<outer_basic_node__t> Scatt_87_o_1_90;
    #pragma HLS STREAM variable=Scatt_87_o_1_90 depth=4
    stream<outer_basic_ap_fi_t> Scatt_87_o_2_91;
    #pragma HLS STREAM variable=Scatt_87_o_2_91 depth=4
    Scatt_87(
        i_0_88,
        Scatt_87_o_0_89,
        Scatt_87_o_1_90,
        Scatt_87_o_2_91,
        Scatt_87_input_len
    );
    uint16_t BinOp_101_input_len = input_length;
    stream<outer_basic_ap_fi_t> BinOp_101_o_0_104;
    #pragma HLS STREAM variable=BinOp_101_o_0_104 depth=4
    BinOp_101(
        Scatt_87_o_0_89,
        Scatt_87_o_2_91,
        BinOp_101_o_0_104,
        BinOp_101_input_len
    );
    uint16_t Gathe_105_input_len = input_length;
    Gathe_105(
        BinOp_101_o_0_104,
        Scatt_87_o_1_90,
        o_0_108,
        Gathe_105_input_len
    );
}


static void Reduc_138_unit_sub_func(
    stream<outer_tuple_bb_2_t> &i_0_123,
    stream<outer_tuple_bb_2_t> &i_0_127,
    stream<outer_tuple_bb_2_t> &o_0_136,
    uint16_t input_length
) {
    uint16_t CopyC_122_input_len = input_length;
    stream<outer_tuple_bb_2_t> CopyC_122_o_0_124;
    #pragma HLS STREAM variable=CopyC_122_o_0_124 depth=4
    stream<outer_tuple_bb_2_t> CopyC_122_o_1_125;
    #pragma HLS STREAM variable=CopyC_122_o_1_125 depth=4
    CopyC_122(
        i_0_123,
        CopyC_122_o_0_124,
        CopyC_122_o_1_125,
        CopyC_122_input_len
    );
    uint16_t Unary_126_input_len = input_length;
    stream<outer_basic_ap_fi_t> Unary_126_o_0_128;
    #pragma HLS STREAM variable=Unary_126_o_0_128 depth=4
    Unary_126(
        i_0_127,
        Unary_126_o_0_128,
        Unary_126_input_len
    );
    uint16_t Unary_116_input_len = input_length;
    stream<outer_basic_ap_fi_t> Unary_116_o_0_118;
    #pragma HLS STREAM variable=Unary_116_o_0_118 depth=4
    Unary_116(
        CopyC_122_o_0_124,
        Unary_116_o_0_118,
        Unary_116_input_len
    );
    uint16_t Unary_119_input_len = input_length;
    stream<outer_basic_node__t> Unary_119_o_0_121;
    #pragma HLS STREAM variable=Unary_119_o_0_121 depth=4
    Unary_119(
        CopyC_122_o_1_125,
        Unary_119_o_0_121,
        Unary_119_input_len
    );
    uint16_t BinOp_129_input_len = input_length;
    stream<outer_basic_ap_fi_t> BinOp_129_o_0_132;
    #pragma HLS STREAM variable=BinOp_129_o_0_132 depth=4
    BinOp_129(
        Unary_116_o_0_118,
        Unary_126_o_0_128,
        BinOp_129_o_0_132,
        BinOp_129_input_len
    );
    uint16_t Gathe_133_input_len = input_length;
    Gathe_133(
        BinOp_129_o_0_132,
        Unary_119_o_0_121,
        o_0_136,
        Gathe_133_input_len
    );
}


static void CopyC_19(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_edge__t> &o_0,
    stream<outer_basic_edge__t> &o_1,
    uint16_t input_length
) {
    LOOP_CopyC_19:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_edge__t copy_src = i_0.read();
        bool end_flag_val = copy_src.end_flag;
        o_0.write(copy_src);
        o_1.write(copy_src);
        if (end_flag_val) break;
    }
}


static void CopyC_12(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_edge__t> &o_0,
    stream<outer_basic_edge__t> &o_1,
    uint16_t input_length
) {
    LOOP_CopyC_12:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_edge__t copy_src = i_0.read();
        bool end_flag_val = copy_src.end_flag;
        o_0.write(copy_src);
        o_1.write(copy_src);
        if (end_flag_val) break;
    }
}


static void Unary_16(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
) {
    LOOP_Unary_16:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_edge__t unary_src = i_0.read();
        bool end_flag_val = unary_src.end_flag;
        basic_ap_fi_t unary_out = unary_src.weight;
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(unary_out, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Unary_6(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_node__t> &o_0,
    uint16_t input_length
) {
    LOOP_Unary_6:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_edge__t unary_src = i_0.read();
        bool end_flag_val = unary_src.end_flag;
        basic_node__t unary_out = unary_src.src;
        {
            basic_node__t_to_outer_basic_node__t(unary_out, tmp_outer_basic_node__t_var, end_flag_val);
            o_0.write(tmp_outer_basic_node__t_var);
        }
        if (end_flag_val) break;
    }
}


static void Unary_9(
    stream<outer_basic_edge__t> &i_0,
    stream<outer_basic_node__t> &o_0,
    uint16_t input_length
) {
    LOOP_Unary_9:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_edge__t unary_src = i_0.read();
        bool end_flag_val = unary_src.end_flag;
        basic_node__t unary_out = unary_src.dst;
        {
            basic_node__t_to_outer_basic_node__t(unary_out, tmp_outer_basic_node__t_var, end_flag_val);
            o_0.write(tmp_outer_basic_node__t_var);
        }
        if (end_flag_val) break;
    }
}


static void Unary_23(
    stream<outer_basic_node__t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
) {
    LOOP_Unary_23:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_node__t unary_src = i_0.read();
        bool end_flag_val = unary_src.end_flag;
        basic_ap_fi_t unary_out = unary_src.distance;
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(unary_out, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Gathe_27(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_node__t> &i_1,
    stream<outer_basic_ap_fi_t> &i_2,
    stream<outer_tuple_bbb_1_t> &o_0,
    uint16_t input_length
) {
    LOOP_Gathe_27:
    while (true) {
#pragma HLS PIPELINE
        bool end_flag_val = false;
        outer_basic_ap_fi_t gather_src_0 = i_0.read();
        end_flag_val |= gather_src_0.end_flag;
        outer_basic_ap_fi_t_to_basic_ap_fi_t(gather_src_0, real_gather_src_0);
        outer_basic_node__t gather_src_1 = i_1.read();
        end_flag_val |= gather_src_1.end_flag;
        outer_basic_node__t_to_basic_node__t(gather_src_1, real_gather_src_1);
        outer_basic_ap_fi_t gather_src_2 = i_2.read();
        end_flag_val |= gather_src_2.end_flag;
        outer_basic_ap_fi_t_to_basic_ap_fi_t(gather_src_2, real_gather_src_2);
        outer_tuple_bbb_1_t gather_result = {real_gather_src_0, real_gather_src_1, real_gather_src_2, end_flag_val};
        o_0.write(gather_result);
        if (end_flag_val) break;
    }
}


static void CopyC_57(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_tuple_bbb_1_t> &o_0,
    stream<outer_tuple_bbb_1_t> &o_1,
    uint16_t input_length
) {
    LOOP_CopyC_57:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bbb_1_t copy_src = i_0.read();
        printf("CopyC_57:\n");
        print_outer_tuple_bbb_1_t(copy_src);
        bool end_flag_val = copy_src.end_flag;
        o_0.write(copy_src);
        o_1.write(copy_src);
        if (end_flag_val) break;
    }
}


static void Scatt_32(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_2,
    uint16_t input_length
) {
    LOOP_Scatt_32:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bbb_1_t scatter_src = i_0.read();
        bool end_flag_val = scatter_src.end_flag;
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(scatter_src.ele_2, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_2.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void BinOp_48(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_bool_t> &o_0,
    uint16_t input_length
) {
    LOOP_BinOp_48:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_ap_fi_t binop_src_0 = i_0.read();
        outer_basic_ap_fi_t binop_src_1 = { 0.0, false };
        bool end_flag_val = binop_src_0.end_flag | binop_src_1.end_flag;
        basic_bool_t binop_out = { binop_src_0.ele >= binop_src_1.ele };
        {
            basic_bool_t_to_outer_basic_bool_t(binop_out, tmp_outer_basic_bool_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_bool_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Condi_61(
    stream<outer_tuple_bbb_1_t> &i_data,
    stream<outer_basic_bool_t> &i_cond,
    stream<outer_opt__of_tup_t> &o_0,
    uint16_t input_length
) {
    LOOP_Condi_61:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bbb_1_t cond_data = i_data.read();
        outer_basic_bool_t cond = i_cond.read();
        bool end_flag_val = cond_data.end_flag | cond.end_flag;
        outer_tuple_bbb_1_t_to_tuple_bbb_1_t(cond_data, real_cond_data);
        outer_basic_bool_t_to_basic_bool_t(cond, real_cond);
        opt__of_tup_t cond_result = {real_cond_data, real_cond};
        {
            opt__of_tup_t_to_outer_opt__of_tup_t(cond_result, tmp_outer_opt__of_tup_t_var, end_flag_val);
            o_0.write(tmp_outer_opt__of_tup_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Colle_65(
    stream<outer_opt__of_tup_t> &i_0,
    stream<outer_tuple_bbb_1_t> &o_0,
    uint16_t input_length
) {
    LOOP_Colle_65:
    while (true) {
#pragma HLS PIPELINE
        outer_opt__of_tup_t collect_src = i_0.read();
        printf("Colle_65:\n");
        print_outer_opt__of_tup_t(collect_src);
        bool end_flag_val = collect_src.end_flag;
        if (collect_src.valid.ele || end_flag_val) {
            {
                tuple_bbb_1_t_to_outer_tuple_bbb_1_t(collect_src.data, tmp_outer_tuple_bbb_1_t_var, end_flag_val);
                o_0.write(tmp_outer_tuple_bbb_1_t_var);
            }
        }
        if (end_flag_val) break;
    }
}


static void Reduc_138_pre_process(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_basic_node__t> &intermediate_key,
    stream<outer_tuple_bb_2_t> &intermediate_transform,
    uint16_t input_length
) {
    LOOP_Reduc_138_pre_process:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bbb_1_t reduce_src = i_0.read();
        bool end_flag_val = reduce_src.end_flag;
        hls::stream<outer_tuple_bbb_1_t> reduce_key_in_stream("reduce_key_in_stream");
        #pragma HLS STREAM variable=reduce_key_in_stream depth=4
        hls::stream<outer_tuple_bbb_1_t> reduce_transform_in_stream("reduce_transform_in_stream");
        #pragma HLS STREAM variable=reduce_transform_in_stream depth=4
        reduce_src.end_flag = true;reduce_key_in_stream.write(reduce_src);
        reduce_transform_in_stream.write(reduce_src);
        hls::stream<outer_basic_node__t> reduce_key_out_stream("reduce_key_out_stream");
        #pragma HLS STREAM variable=reduce_key_out_stream depth=4
        hls::stream<outer_tuple_bb_2_t> reduce_transform_out_stream("reduce_transform_out_stream");
        #pragma HLS STREAM variable=reduce_transform_out_stream depth=4
        Reduc_138_key_sub_func(reduce_key_in_stream, reduce_key_out_stream, 1);
        Reduc_138_transform_sub_func(reduce_transform_in_stream, reduce_transform_out_stream, 1);
        outer_basic_node__t reduce_key_out = reduce_key_out_stream.read();
        reduce_key_out.end_flag = end_flag_val;
        intermediate_key.write(reduce_key_out);
        outer_tuple_bb_2_t reduce_transform_out = reduce_transform_out_stream.read();
        reduce_transform_out.end_flag = end_flag_val;
        intermediate_transform.write(reduce_transform_out);
        if (end_flag_val) break;
    }
}


static void Reduc_138_unit_reduce(
    stream<outer_basic_node__t> &intermediate_key,
    stream<outer_tuple_bb_2_t> &intermediate_transform,
    stream<outer_tuple_bb_2_t> &o_0,
    uint16_t input_length
) {
    static outer_tuple_btb_3_t key_mem[MAX_NUM];
    #pragma HLS ARRAY_PARTITION variable=key_mem complete dim=0
    CLEAR_REDUCE_VALID: for (int i_reduce_clear = 0; i_reduce_clear < MAX_NUM; i_reduce_clear++) {
    #pragma UNROLL
        key_mem[i_reduce_clear].valid.ele = 0;
    }
    LOOP_Reduc_138_unit_reduce:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_node__t reduce_key_out = intermediate_key.read();
        outer_tuple_bb_2_t reduce_transform_out = intermediate_transform.read();
        printf("Reduc_138_unit_reduce:\n");
        print_outer_basic_node__t(reduce_key_out);
        print_outer_tuple_bb_2_t(reduce_transform_out);
        bool end_flag_val = reduce_key_out.end_flag | reduce_transform_out.end_flag;
        bool merged = false;
        outer_basic_node__t_to_basic_node__t(reduce_key_out, real_reduce_key_out);
        outer_tuple_bb_2_t_to_tuple_bb_2_t(reduce_transform_out, real_reduce_transform_out);
        SCAN_BRAM_INTER_LOOP: for (int i_in_reduce = 0; i_in_reduce < MAX_NUM; i_in_reduce++) {
        #pragma HLS PIPELINE
            outer_tuple_btb_3_t cur_ele = key_mem[i_in_reduce];
            if (!merged && !cur_ele.valid.ele) {
                key_mem[i_in_reduce].valid.ele = 1;
                key_mem[i_in_reduce].key = real_reduce_key_out;
                key_mem[i_in_reduce].data = real_reduce_transform_out;
                merged = true;
            } else if (!merged && cur_ele.valid.ele && cur_ele.key.id.ele == real_reduce_key_out.id.ele) {
                hls::stream<outer_tuple_bb_2_t> reduce_unit_stream_0("reduce_unit_stream_0");
        #pragma HLS STREAM variable=reduce_unit_stream_0 depth=4
                hls::stream<outer_tuple_bb_2_t> reduce_unit_stream_1("reduce_unit_stream_1");
        #pragma HLS STREAM variable=reduce_unit_stream_1 depth=4
                hls::stream<outer_tuple_bb_2_t> reduce_unit_stream_out("reduce_unit_stream_out");
        #pragma HLS STREAM variable=reduce_unit_stream_out depth=4
                {
                    tuple_bb_2_t_to_outer_tuple_bb_2_t(cur_ele.data, tmp_outer_tuple_bb_2_t_var, true);
                    reduce_unit_stream_0.write(tmp_outer_tuple_bb_2_t_var);
                }
                {
                    tuple_bb_2_t_to_outer_tuple_bb_2_t(real_reduce_transform_out, tmp_outer_tuple_bb_2_t_var, true);
                    reduce_unit_stream_1.write(tmp_outer_tuple_bb_2_t_var);
                }
                Reduc_138_unit_sub_func(reduce_unit_stream_0, reduce_unit_stream_1, reduce_unit_stream_out, 1);
                outer_tuple_bb_2_t reduce_unit_out = reduce_unit_stream_out.read();
                outer_tuple_bb_2_t_to_tuple_bb_2_t(reduce_unit_out, real_reduce_unit_out);
                key_mem[i_in_reduce].data = real_reduce_unit_out;
                merged = true;
            }
        }
        if (end_flag_val) break;
    }
    WRITE_KEY_MEM_LOOP: for (int i_write_key_mem = 0; i_write_key_mem < MAX_NUM; i_write_key_mem++) {
    #pragma HLS PIPELINE
        if (key_mem[i_write_key_mem].valid.ele) {
            {
                tuple_bb_2_t_to_outer_tuple_bb_2_t(key_mem[i_write_key_mem].data, tmp_outer_tuple_bb_2_t_var, true);
                printf("Reduc_138_unit_reduce[end]:\n");
                print_outer_tuple_bb_2_t(tmp_outer_tuple_bb_2_t_var);
                o_0.write(tmp_outer_tuple_bb_2_t_var);
            }
        }
    }
}


static void Scatt_87(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    stream<outer_basic_node__t> &o_1,
    stream<outer_basic_ap_fi_t> &o_2,
    uint16_t input_length
) {
    LOOP_Scatt_87:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bbb_1_t scatter_src = i_0.read();
        bool end_flag_val = scatter_src.end_flag;
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(scatter_src.ele_0, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        {
            basic_node__t_to_outer_basic_node__t(scatter_src.ele_1, tmp_outer_basic_node__t_var, end_flag_val);
            o_1.write(tmp_outer_basic_node__t_var);
        }
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(scatter_src.ele_2, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_2.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void CopyC_122(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_tuple_bb_2_t> &o_0,
    stream<outer_tuple_bb_2_t> &o_1,
    uint16_t input_length
) {
    LOOP_CopyC_122:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bb_2_t copy_src = i_0.read();
        bool end_flag_val = copy_src.end_flag;
        o_0.write(copy_src);
        o_1.write(copy_src);
        if (end_flag_val) break;
    }
}


static void Unary_126(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
) {
    LOOP_Unary_126:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bb_2_t unary_src = i_0.read();
        bool end_flag_val = unary_src.end_flag;
        basic_ap_fi_t unary_out = unary_src.ele_0;
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(unary_out, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Scatt_68(
    stream<outer_tuple_bbb_1_t> &i_0,
    stream<outer_basic_node__t> &o_1,
    uint16_t input_length
) {
    LOOP_Scatt_68:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bbb_1_t scatter_src = i_0.read();
        bool end_flag_val = scatter_src.end_flag;
        {
            basic_node__t_to_outer_basic_node__t(scatter_src.ele_1, tmp_outer_basic_node__t_var, end_flag_val);
            o_1.write(tmp_outer_basic_node__t_var);
        }
        if (end_flag_val) break;
    }
}


static void Scatt_148(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    stream<outer_basic_node__t> &o_1,
    uint16_t input_length
) {
    LOOP_Scatt_148:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bb_2_t scatter_src = i_0.read();
        bool end_flag_val = scatter_src.end_flag;
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(scatter_src.ele_0, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        {
            basic_node__t_to_outer_basic_node__t(scatter_src.ele_1, tmp_outer_basic_node__t_var, end_flag_val);
            o_1.write(tmp_outer_basic_node__t_var);
        }
        if (end_flag_val) break;
    }
}


static void BinOp_101(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_ap_fi_t> &i_1,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
) {
    LOOP_BinOp_101:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_ap_fi_t binop_src_0 = i_0.read();
        outer_basic_ap_fi_t binop_src_1 = i_1.read();
        bool end_flag_val = binop_src_0.end_flag | binop_src_1.end_flag;
        basic_ap_fi_t binop_out = { binop_src_0.ele + binop_src_1.ele };
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(binop_out, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Unary_116(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
) {
    LOOP_Unary_116:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bb_2_t unary_src = i_0.read();
        bool end_flag_val = unary_src.end_flag;
        basic_ap_fi_t unary_out = unary_src.ele_0;
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(unary_out, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Unary_119(
    stream<outer_tuple_bb_2_t> &i_0,
    stream<outer_basic_node__t> &o_0,
    uint16_t input_length
) {
    LOOP_Unary_119:
    while (true) {
#pragma HLS PIPELINE
        outer_tuple_bb_2_t unary_src = i_0.read();
        bool end_flag_val = unary_src.end_flag;
        basic_node__t unary_out = unary_src.ele_1;
        {
            basic_node__t_to_outer_basic_node__t(unary_out, tmp_outer_basic_node__t_var, end_flag_val);
            o_0.write(tmp_outer_basic_node__t_var);
        }
        if (end_flag_val) break;
    }
}


static void CopyC_165(
    stream<outer_basic_node__t> &i_0,
    stream<outer_basic_node__t> &o_0,
    stream<outer_basic_node__t> &o_1,
    uint16_t input_length
) {
    LOOP_CopyC_165:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_node__t copy_src = i_0.read();
        bool end_flag_val = copy_src.end_flag;
        o_0.write(copy_src);
        o_1.write(copy_src);
        if (end_flag_val) break;
    }
}


static void Gathe_105(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_node__t> &i_1,
    stream<outer_tuple_bb_2_t> &o_0,
    uint16_t input_length
) {
    LOOP_Gathe_105:
    while (true) {
#pragma HLS PIPELINE
        bool end_flag_val = false;
        outer_basic_ap_fi_t gather_src_0 = i_0.read();
        end_flag_val |= gather_src_0.end_flag;
        outer_basic_ap_fi_t_to_basic_ap_fi_t(gather_src_0, real_gather_src_0);
        outer_basic_node__t gather_src_1 = i_1.read();
        end_flag_val |= gather_src_1.end_flag;
        outer_basic_node__t_to_basic_node__t(gather_src_1, real_gather_src_1);
        outer_tuple_bb_2_t gather_result = {real_gather_src_0, real_gather_src_1, end_flag_val};
        o_0.write(gather_result);
        if (end_flag_val) break;
    }
}


static void BinOp_129(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_ap_fi_t> &i_1,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
) {
    LOOP_BinOp_129:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_ap_fi_t binop_src_0 = i_0.read();
        outer_basic_ap_fi_t binop_src_1 = i_1.read();
        bool end_flag_val = binop_src_0.end_flag | binop_src_1.end_flag;
        basic_ap_fi_t binop_out = { ((binop_src_0.ele) < (binop_src_1.ele) ? binop_src_0.ele : binop_src_1.ele) };
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(binop_out, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Unary_158(
    stream<outer_basic_node__t> &i_0,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
) {
    LOOP_Unary_158:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_node__t unary_src = i_0.read();
        bool end_flag_val = unary_src.end_flag;
        basic_ap_fi_t unary_out = unary_src.distance;
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(unary_out, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Gathe_133(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_node__t> &i_1,
    stream<outer_tuple_bb_2_t> &o_0,
    uint16_t input_length
) {
    LOOP_Gathe_133:
    while (true) {
#pragma HLS PIPELINE
        bool end_flag_val = false;
        outer_basic_ap_fi_t gather_src_0 = i_0.read();
        end_flag_val |= gather_src_0.end_flag;
        outer_basic_ap_fi_t_to_basic_ap_fi_t(gather_src_0, real_gather_src_0);
        outer_basic_node__t gather_src_1 = i_1.read();
        end_flag_val |= gather_src_1.end_flag;
        outer_basic_node__t_to_basic_node__t(gather_src_1, real_gather_src_1);
        outer_tuple_bb_2_t gather_result = {real_gather_src_0, real_gather_src_1, end_flag_val};
        o_0.write(gather_result);
        if (end_flag_val) break;
    }
}


static void BinOp_161(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_ap_fi_t> &i_1,
    stream<outer_basic_ap_fi_t> &o_0,
    uint16_t input_length
) {
    LOOP_BinOp_161:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_ap_fi_t binop_src_0 = i_0.read();
        outer_basic_ap_fi_t binop_src_1 = i_1.read();
        bool end_flag_val = binop_src_0.end_flag | binop_src_1.end_flag;
        basic_ap_fi_t binop_out = { ((binop_src_0.ele) < (binop_src_1.ele) ? binop_src_0.ele : binop_src_1.ele) };
        {
            basic_ap_fi_t_to_outer_basic_ap_fi_t(binop_out, tmp_outer_basic_ap_fi_t_var, end_flag_val);
            o_0.write(tmp_outer_basic_ap_fi_t_var);
        }
        if (end_flag_val) break;
    }
}


static void Gathe_170(
    stream<outer_basic_ap_fi_t> &i_0,
    stream<outer_basic_node__t> &i_1,
    stream<outer_tuple_bb_2_t> &o_0,
    uint16_t input_length
) {
    LOOP_Gathe_170:
    while (true) {
#pragma HLS PIPELINE
        bool end_flag_val = false;
        outer_basic_ap_fi_t gather_src_0 = i_0.read();
        end_flag_val |= gather_src_0.end_flag;
        outer_basic_ap_fi_t_to_basic_ap_fi_t(gather_src_0, real_gather_src_0);
        outer_basic_node__t gather_src_1 = i_1.read();
        end_flag_val |= gather_src_1.end_flag;
        outer_basic_node__t_to_basic_node__t(gather_src_1, real_gather_src_1);
        outer_tuple_bb_2_t gather_result = {real_gather_src_0, real_gather_src_1, end_flag_val};
        o_0.write(gather_result);
        if (end_flag_val) break;
    }
}


void graphyflow(
    stream<outer_basic_edge__t> &i_0_20,
    stream<outer_tuple_bb_2_t> &o_0_173,
    uint16_t input_length
) {
#pragma HLS dataflow
    uint16_t CopyC_19_input_len = input_length;
    stream<outer_basic_edge__t> CopyC_19_o_0_21;
    #pragma HLS STREAM variable=CopyC_19_o_0_21 depth=4
    stream<outer_basic_edge__t> CopyC_19_o_1_22;
    #pragma HLS STREAM variable=CopyC_19_o_1_22 depth=4
    CopyC_19(
        i_0_20,
        CopyC_19_o_0_21,
        CopyC_19_o_1_22,
        CopyC_19_input_len
    );
    uint16_t CopyC_12_input_len = input_length;
    stream<outer_basic_edge__t> CopyC_12_o_0_14;
    #pragma HLS STREAM variable=CopyC_12_o_0_14 depth=4
    stream<outer_basic_edge__t> CopyC_12_o_1_15;
    #pragma HLS STREAM variable=CopyC_12_o_1_15 depth=4
    CopyC_12(
        CopyC_19_o_0_21,
        CopyC_12_o_0_14,
        CopyC_12_o_1_15,
        CopyC_12_input_len
    );
    uint16_t Unary_16_input_len = input_length;
    stream<outer_basic_ap_fi_t> Unary_16_o_0_18;
    #pragma HLS STREAM variable=Unary_16_o_0_18 depth=4
    Unary_16(
        CopyC_19_o_1_22,
        Unary_16_o_0_18,
        Unary_16_input_len
    );
    uint16_t Unary_6_input_len = input_length;
    stream<outer_basic_node__t> Unary_6_o_0_8;
    #pragma HLS STREAM variable=Unary_6_o_0_8 depth=4
    Unary_6(
        CopyC_12_o_0_14,
        Unary_6_o_0_8,
        Unary_6_input_len
    );
    uint16_t Unary_9_input_len = input_length;
    stream<outer_basic_node__t> Unary_9_o_0_11;
    #pragma HLS STREAM variable=Unary_9_o_0_11 depth=4
    Unary_9(
        CopyC_12_o_1_15,
        Unary_9_o_0_11,
        Unary_9_input_len
    );
    uint16_t Unary_23_input_len = input_length;
    stream<outer_basic_ap_fi_t> Unary_23_o_0_25;
    #pragma HLS STREAM variable=Unary_23_o_0_25 depth=4
    Unary_23(
        Unary_6_o_0_8,
        Unary_23_o_0_25,
        Unary_23_input_len
    );
    uint16_t Gathe_27_input_len = input_length;
    stream<outer_tuple_bbb_1_t> Gathe_27_o_0_31;
    #pragma HLS STREAM variable=Gathe_27_o_0_31 depth=4
    Gathe_27(
        Unary_23_o_0_25,
        Unary_9_o_0_11,
        Unary_16_o_0_18,
        Gathe_27_o_0_31,
        Gathe_27_input_len
    );
    uint16_t CopyC_57_input_len = input_length;
    stream<outer_tuple_bbb_1_t> CopyC_57_o_0_59;
    #pragma HLS STREAM variable=CopyC_57_o_0_59 depth=4
    stream<outer_tuple_bbb_1_t> CopyC_57_o_1_60;
    #pragma HLS STREAM variable=CopyC_57_o_1_60 depth=4
    CopyC_57(
        Gathe_27_o_0_31,
        CopyC_57_o_0_59,
        CopyC_57_o_1_60,
        CopyC_57_input_len
    );
    uint16_t Scatt_32_input_len = input_length;
    stream<outer_basic_ap_fi_t> Scatt_32_o_2_36;
    #pragma HLS STREAM variable=Scatt_32_o_2_36 depth=4
    Scatt_32(
        CopyC_57_o_0_59,
        Scatt_32_o_2_36,
        Scatt_32_input_len
    );
    uint16_t BinOp_48_input_len = input_length;
    stream<outer_basic_bool_t> BinOp_48_o_0_51;
    #pragma HLS STREAM variable=BinOp_48_o_0_51 depth=4
    BinOp_48(
        Scatt_32_o_2_36,
        BinOp_48_o_0_51,
        BinOp_48_input_len
    );
    uint16_t Condi_61_input_len = input_length;
    stream<outer_opt__of_tup_t> Condi_61_o_0_64;
    #pragma HLS STREAM variable=Condi_61_o_0_64 depth=4
    Condi_61(
        CopyC_57_o_1_60,
        BinOp_48_o_0_51,
        Condi_61_o_0_64,
        Condi_61_input_len
    );
    uint16_t Colle_65_input_len = input_length;
    stream<outer_tuple_bbb_1_t> Colle_65_o_0_67;
    #pragma HLS STREAM variable=Colle_65_o_0_67 depth=4
    Colle_65(
        Condi_61_o_0_64,
        Colle_65_o_0_67,
        Colle_65_input_len
    );
    uint16_t Reduc_138_pre_process_input_len = input_length;
    stream<outer_basic_node__t> Reduc_138_pre_process_o_intermediate_key_176;
    #pragma HLS STREAM variable=Reduc_138_pre_process_o_intermediate_key_176 depth=4
    stream<outer_tuple_bb_2_t> Reduc_138_pre_process_o_intermediate_transform_180;
    #pragma HLS STREAM variable=Reduc_138_pre_process_o_intermediate_transform_180 depth=4
    Reduc_138_pre_process(
        Colle_65_o_0_67,
        Reduc_138_pre_process_o_intermediate_key_176,
        Reduc_138_pre_process_o_intermediate_transform_180,
        Reduc_138_pre_process_input_len
    );
    uint16_t Reduc_138_unit_reduce_input_len = input_length;
    stream<outer_tuple_bb_2_t> Reduc_138_unit_reduce_o_0_140;
    #pragma HLS STREAM variable=Reduc_138_unit_reduce_o_0_140 depth=4
    Reduc_138_unit_reduce(
        Reduc_138_pre_process_o_intermediate_key_176,
        Reduc_138_pre_process_o_intermediate_transform_180,
        Reduc_138_unit_reduce_o_0_140,
        Reduc_138_unit_reduce_input_len
    );
    uint16_t Scatt_148_input_len = input_length;
    stream<outer_basic_ap_fi_t> Scatt_148_o_0_150;
    #pragma HLS STREAM variable=Scatt_148_o_0_150 depth=4
    stream<outer_basic_node__t> Scatt_148_o_1_151;
    #pragma HLS STREAM variable=Scatt_148_o_1_151 depth=4
    Scatt_148(
        Reduc_138_unit_reduce_o_0_140,
        Scatt_148_o_0_150,
        Scatt_148_o_1_151,
        Scatt_148_input_len
    );
    uint16_t CopyC_165_input_len = input_length;
    stream<outer_basic_node__t> CopyC_165_o_0_167;
    #pragma HLS STREAM variable=CopyC_165_o_0_167 depth=4
    stream<outer_basic_node__t> CopyC_165_o_1_168;
    #pragma HLS STREAM variable=CopyC_165_o_1_168 depth=4
    CopyC_165(
        Scatt_148_o_1_151,
        CopyC_165_o_0_167,
        CopyC_165_o_1_168,
        CopyC_165_input_len
    );
    uint16_t Unary_158_input_len = input_length;
    stream<outer_basic_ap_fi_t> Unary_158_o_0_160;
    #pragma HLS STREAM variable=Unary_158_o_0_160 depth=4
    Unary_158(
        CopyC_165_o_0_167,
        Unary_158_o_0_160,
        Unary_158_input_len
    );
    uint16_t BinOp_161_input_len = input_length;
    stream<outer_basic_ap_fi_t> BinOp_161_o_0_164;
    #pragma HLS STREAM variable=BinOp_161_o_0_164 depth=4
    BinOp_161(
        Scatt_148_o_0_150,
        Unary_158_o_0_160,
        BinOp_161_o_0_164,
        BinOp_161_input_len
    );
    uint16_t Gathe_170_input_len = input_length;
    Gathe_170(
        BinOp_161_o_0_164,
        CopyC_165_o_1_168,
        o_0_173,
        Gathe_170_input_len
    );
}



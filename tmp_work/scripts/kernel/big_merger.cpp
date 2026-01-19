#include "shared_kernel_params.h"

void merge_big_kernels(hls::stream<write_burst_pkt_t> &big_kernel_1_out_stream,
hls::stream<write_burst_pkt_t> &big_kernel_2_out_stream,
hls::stream<write_burst_pkt_t> &big_kernel_3_out_stream,
hls::stream<write_burst_pkt_t> &kernel_out_stream) {

 write_burst_pkt_t tmp_prop_pkt[BIG_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = tmp_prop_pkt dim = 0 complete

    bool process_flag[BIG_MERGER_LENGTH];
#pragma HLS ARRAY_PARTITION variable = process_flag dim = 0 complete

    for (int i = 0; i < BIG_MERGER_LENGTH; i++) {
#pragma HLS unroll
        process_flag[i] = 0;
    }

    bus_word_t merged_write_burst;

    write_burst_pkt_t one_write_burst;

    uint32_t outer_idx = 0;

    ap_fixed_pod_t tmp_dist_array[8];
#pragma HLS ARRAY_PARTITION variable = tmp_dist_array dim = 0 complete
    ap_uint<32> tmp_cnt_array[8];
#pragma HLS ARRAY_PARTITION variable = tmp_cnt_array dim = 0 complete

    distance_t max_val = (distance_t)(16384.0);
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);


    // distance_t cc_ini = (distance_t)(0.0);
    // ap_fixed_pod_t cc_ini_pod = *reinterpret_cast<ap_fixed_pod_t *>(&cc_ini);

merge_tmp_prop_big_krnls:
    while (true) {
#pragma HLS pipeline style = flp

        if (!process_flag[0])
                    process_flag[0] = big_kernel_1_out_stream.read_nb(tmp_prop_pkt[0]);
        if (!process_flag[1])
                    process_flag[1] = big_kernel_2_out_stream.read_nb(tmp_prop_pkt[1]);
        if (!process_flag[2])
                    process_flag[2] = big_kernel_3_out_stream.read_nb(tmp_prop_pkt[2]);
    bool merge_flag = 
        process_flag[0] & 
        process_flag[1] & 
        process_flag[2] & 
                        1;

if (merge_flag) {
            for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
                tmp_dist_array[i] = max_pod;
                tmp_cnt_array[i] = 0;
                
            }

            for (int i = 0; i < BIG_MERGER_LENGTH; i++) {
#pragma HLS UNROLL
                for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
                    ap_uint<64> update64 =
                        tmp_prop_pkt[i].data.range(63 + (j << 6), (j << 6));
                    ap_fixed_pod_t update_dist = update64.range(31, 0);
                    ap_uint<32> update_cnt = update64.range(63, 32);
                    if (update_cnt != 0) {
                        tmp_dist_array[j] =
                            (update_dist < tmp_dist_array[j]) ? update_dist
                                                              : tmp_dist_array[j];
                        tmp_cnt_array[j] += update_cnt;
                    }

                }
            }

            for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
                ap_uint<64> out64;
                out64.range(31, 0) = tmp_dist_array[i];
                out64.range(63, 32) = tmp_cnt_array[i];
                merged_write_burst.range(63 + (i << 6), (i << 6)) = out64;
            }

            one_write_burst.data = merged_write_burst;
            kernel_out_stream.write(one_write_burst);

            for (int i = 0; i < BIG_MERGER_LENGTH; i++) {
#pragma HLS unroll
                process_flag[i] = 0;
            }
        }
    }
}


extern "C" void
big_merger(
    hls::stream<write_burst_pkt_t> &big_kernel_1_out_stream,
    hls::stream<write_burst_pkt_t> &big_kernel_2_out_stream,
    hls::stream<write_burst_pkt_t> &big_kernel_3_out_stream,
    hls::stream<write_burst_pkt_t> &kernel_out_stream) {
#pragma HLS interface ap_ctrl_none port = return
#pragma HLS DATAFLOW
    merge_big_kernels(
        big_kernel_1_out_stream,
        big_kernel_2_out_stream,
        big_kernel_3_out_stream,
        kernel_out_stream);
}

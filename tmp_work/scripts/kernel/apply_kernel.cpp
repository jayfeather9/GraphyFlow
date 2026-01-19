#include "shared_kernel_params.h"


void merge_big_little_writes(
    hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
    hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
    hls::stream<in_write_burst_w_dst_pkt_t> &kernel_out_stream,
    uint32_t little_kernel_length, uint32_t big_kernel_length,
    uint32_t little_kernel_st_offset, uint32_t big_kernel_st_offset) {
    write_burst_pkt_t big_tmp_prop_pkt;
    write_burst_pkt_t little_tmp_prop_pkt;

    uint32_t little_idx = little_kernel_st_offset;
    uint32_t big_idx = big_kernel_st_offset;
    uint32_t total_length = little_kernel_length + big_kernel_length;

LOOP_MERGE_WRITES:
    while (true) {
        if (total_length == 0) {
            in_write_burst_w_dst_pkt_t end_pkt;
            end_pkt.end_flag = true;
            kernel_out_stream.write(end_pkt);
            break;
        }

        if (little_kernel_out_stream.read_nb(little_tmp_prop_pkt)) {
            in_write_burst_w_dst_pkt_t little_write_burst;
            little_write_burst.data = little_tmp_prop_pkt.data;
            little_write_burst.dest_addr = little_idx;
            little_write_burst.end_flag = false;
            kernel_out_stream.write(little_write_burst);
            little_idx++;
            total_length--;
        } else if (big_kernel_out_stream.read_nb(big_tmp_prop_pkt)) {
            in_write_burst_w_dst_pkt_t big_write_burst;
            big_write_burst.data = big_tmp_prop_pkt.data;
            big_write_burst.dest_addr = big_idx;
            big_write_burst.end_flag = false;
            kernel_out_stream.write(big_write_burst);
            big_idx++;
            total_length--;
        }
    }
}


static void
apply_func(bus_word_t* node_props,
 hls::stream<in_write_burst_w_dst_pkt_t> &write_burst_stream,
 hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream) {
    distance_t max_val = (distance_t)(16384.0);
    ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);

    static const int SLOT_CNT = 4;
    bool slot_valid[SLOT_CNT] = {false, false, false, false};
#pragma HLS ARRAY_PARTITION variable = slot_valid complete dim = 0
    uint32_t slot_word_addr[SLOT_CNT];
#pragma HLS ARRAY_PARTITION variable = slot_word_addr complete dim = 0
    bool slot_have_half0[SLOT_CNT] = {false, false, false, false};
#pragma HLS ARRAY_PARTITION variable = slot_have_half0 complete dim = 0
    bool slot_have_half1[SLOT_CNT] = {false, false, false, false};
#pragma HLS ARRAY_PARTITION variable = slot_have_half1 complete dim = 0
    bus_word_t slot_old_props[SLOT_CNT];
#pragma HLS ARRAY_PARTITION variable = slot_old_props complete dim = 0
    ap_fixed_pod_t slot_update_dist[SLOT_CNT][2][8];
#pragma HLS ARRAY_PARTITION variable = slot_update_dist complete dim = 0
    ap_uint<32> slot_update_cnt[SLOT_CNT][2][8];
#pragma HLS ARRAY_PARTITION variable = slot_update_cnt complete dim = 0

    LOOP_WHILE_44:
    while (true) {
        in_write_burst_w_dst_pkt_t in_pkt = write_burst_stream.read();
        if (in_pkt.end_flag) {
            // Expect all half-packets to be paired before termination.
            write_burst_w_dst_pkt_t end_pkt;
            end_pkt.last = true;
            kernel_out_stream.write(end_pkt);
            break;
        }

        uint32_t dest_half_addr = in_pkt.dest_addr;
        uint32_t dest_word_addr = (dest_half_addr >> 1);
        uint32_t half = (dest_half_addr & 1);

        int slot = -1;
        LOOP_FIND_SLOT:
        for (int i = 0; i < SLOT_CNT; i++) {
#pragma HLS UNROLL
            if (slot_valid[i] && slot_word_addr[i] == dest_word_addr) {
                slot = i;
            }
        }
        if (slot < 0) {
            LOOP_ALLOC_SLOT:
            for (int i = 0; i < SLOT_CNT; i++) {
#pragma HLS UNROLL
                if (!slot_valid[i] && slot < 0) {
                    slot = i;
                }
            }
            slot_valid[slot] = true;
            slot_word_addr[slot] = dest_word_addr;
            slot_have_half0[slot] = false;
            slot_have_half1[slot] = false;
            slot_old_props[slot] = node_props[dest_word_addr];
        }

        LOOP_UNPACK_64_UPDATES:
        for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
            ap_uint<64> u64 = in_pkt.data.range(63 + (i << 6), (i << 6));
            slot_update_dist[slot][half][i] = u64.range(31, 0);
            slot_update_cnt[slot][half][i] = u64.range(63, 32);
        }
        if (half == 0) {
            slot_have_half0[slot] = true;
        } else {
            slot_have_half1[slot] = true;
        }

        if (slot_have_half0[slot] && slot_have_half1[slot]) {
            bus_word_t ori_props = slot_old_props[slot];
            bus_word_t new_props;

            write_burst_w_dst_pkt_t out_pkt;
            out_pkt.dest = dest_word_addr;
            out_pkt.last = false;

            LOOP_APPLY_64_TO_32:
            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                ap_fixed_pod_t old =
                    ori_props.range(31 + (i << 5), (i << 5));

                uint32_t half_sel = (i >> 3);
                uint32_t lane = (i & 7);
                ap_fixed_pod_t update_dist =
                    slot_update_dist[slot][half_sel][lane];
                ap_uint<32> update_cnt =
                    slot_update_cnt[slot][half_sel][lane];

                ap_fixed_pod_t new_prop = old;
                if (update_cnt != 0) {
                    new_prop = (old < update_dist) ? old : update_dist;
                    if ((update_cnt & 3) == 0) {
                        new_prop = 0;
                    }
                } else {
                    // No update: treat as infinity, keep old.
                    (void)max_pod;
                }

                new_props.range(31 + (i << 5), (i << 5)) = new_prop;
            }
            out_pkt.data = new_props;
            kernel_out_stream.write(out_pkt);

            slot_valid[slot] = false;
        }
    }
}



extern "C" void
apply_kernel(bus_word_t *node_props,
             uint32_t little_kernel_length,
             uint32_t big_kernel_length,
             uint32_t little_kernel_st_offset,
             uint32_t big_kernel_st_offset,
             hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
             hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
             hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream) {
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = node_props bundle = control
#pragma HLS INTERFACE s_axilite port = little_kernel_length bundle = control
#pragma HLS INTERFACE s_axilite port = big_kernel_length bundle = control
#pragma HLS INTERFACE s_axilite port = little_kernel_st_offset bundle = control
#pragma HLS INTERFACE s_axilite port = big_kernel_st_offset bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    hls::stream<in_write_burst_w_dst_pkt_t> write_burst_stream;
#pragma HLS STREAM variable = write_burst_stream depth = 16

    merge_big_little_writes(little_kernel_out_stream, big_kernel_out_stream, write_burst_stream, little_kernel_length, big_kernel_length, little_kernel_st_offset, big_kernel_st_offset);
    apply_func(node_props, write_burst_stream, kernel_out_stream);
}

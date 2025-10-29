#include "shared_kernel_params.h"

static void
apply_kernel_inter(bus_word_t *node_props, uint32_t dst_num,
                   hls::stream<write_burst_pkt_t> &node_distance_burst_stream,
                   hls::stream<write_burst_pkt_t> &write_burst_stream) {
    uint32_t read_idx = 0;
    uint32_t addr = 0;
    bool pkt_ready = false;
    write_burst_pkt_t pkt;
LOOP_APPLY:
    while (true) {
#pragma HLS PIPELINE II = 1
        if (!pkt_ready) {
            pkt_ready = node_distance_burst_stream.read_nb(pkt);
        }
        if (pkt_ready) {
            pkt_ready = false;
            bus_word_t wide_word = pkt.data;

            bus_word_t node_prop = node_props[read_idx];
            bus_word_t new_node_prop;

            for (int i = 0; i < DBL_PE_NUM; i++) {
#pragma HLS UNROLL
                ap_fixed_pod_t update_dist =
                    wide_word.range(31 + (i << 5), (i << 5));
                ap_fixed_pod_t current_dist =
                    node_prop.range(31 + (i << 5), (i << 5));
                ap_fixed_pod_t new_dist =
                    (update_dist < current_dist) ? update_dist : current_dist;
                // printf("Node %d: current dist = %f, update dist = %f, new
                // dist = %f\n",
                //        addr + i, ap_fixed_to_float(current_dist),
                //        ap_fixed_to_float(update_dist),
                //        ap_fixed_to_float(new_dist));
                // fflush(NULL);
                new_node_prop.range(31 + (i << 5), (i << 5)) = new_dist;
            }

            write_burst_pkt_t out_pkt;
            out_pkt.data = new_node_prop;
            out_pkt.last = false;
            write_burst_stream.write(out_pkt);
            read_idx++;
            addr += (PE_NUM << 1);
            if (addr >= dst_num) {
                break;
            }
        }
    }
}

extern "C" void
apply_kernel(bus_word_t *node_props, uint32_t dst_num,
             hls::stream<write_burst_pkt_t> &kernel_out_stream,
             hls::stream<write_burst_pkt_t> &write_burst_stream) {
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = node_props bundle = control
#pragma HLS INTERFACE s_axilite port = dst_num bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    apply_kernel_inter(node_props, dst_num, kernel_out_stream,
                       write_burst_stream);
}
#include "graphyflow_big.h"

static void
apply_kernel_inter(const bus_word_t *node_props,
                   hls::stream<write_burst_pkt_t> &node_distance_burst_stream,
                   hls::stream<write_burst_pkt_t> &write_burst_stream) {
LOOP_APPLY:
    while (true) {
#pragma HLS PIPELINE II = 1
        write_burst_pkt_t pkt = node_distance_burst_stream.read();

        if (pkt.last) {
            break;
        }

        bus_word_t wide_word = pkt.data;
        uint32_t write_idx = pkt.dest;

        bus_word_t node_prop = node_props[write_idx];
        bus_word_t new_node_prop;

        for (int i = 0; i < DBL_PE_NUM; i++) {
#pragma HLS UNROLL
            ap_fixed_pod_t update_dist =
                wide_word.range(31 + (i << 5), (i << 5));
            ap_fixed_pod_t current_dist =
                node_prop.range(31 + (i << 5), (i << 5));
            ap_fixed_pod_t new_dist =
                (update_dist < current_dist) ? update_dist : current_dist;
            new_node_prop.range(31 + (i << 5), (i << 5)) = new_dist;
        }

        write_burst_pkt_t out_pkt;
        out_pkt.data = new_node_prop;
        out_pkt.dest = write_idx;
        out_pkt.last = false;
        write_burst_stream.write(out_pkt);
    }

    write_burst_pkt_t end_pkt;
    end_pkt.last = true;
    write_burst_stream.write(end_pkt);
}

extern "C" void
apply_kernel(const bus_word_t *node_props,
             hls::stream<write_burst_pkt_t> &kernel_out_stream,
             hls::stream<write_burst_pkt_t> &write_burst_stream) {
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = node_props bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW

    apply_kernel_inter(node_props, kernel_out_stream, write_burst_stream);
}
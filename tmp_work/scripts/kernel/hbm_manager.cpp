#include "graphyflow_big.h"

void write_out(bus_word_t *output,
               hls::stream<write_burst_pkt_t> &write_burst_stm) {
    uint32_t write_idx = 0;

write_out:
    while (true) {
#pragma HLS PIPELINE II = 1

        write_burst_pkt_t one_write_burst;

        if (write_burst_stm.read_nb(one_write_burst)) {

            write_idx = one_write_burst.dest;

            if (one_write_burst.last) {
                break;
            }

            bus_word_t new_prop = one_write_burst.data;
            output[write_idx] = new_prop;
        }
    }
}

static void node_distance_loader(const bus_word_t *node_distances_ddr,
                                 hls::stream<node_dist_pkt_t> &node_dist_stream,
                                 int32_t num_nodes) {
    const int num_dists_per_word =
        AXI_BUS_WIDTH / DISTANCE_BITWIDTH; // 16 dists per 512-bit word
    const int num_dists_per_pkt =
        8; // PE_NUM = 8, each packet contains 8 distances (256 bits)
    const int num_wide_reads =
        (num_nodes + num_dists_per_word - 1) / num_dists_per_word;

    int nodes_sent = 0;
LOOP_NPL_S1_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 2
        bus_word_t wide_word = node_distances_ddr[i];
        node_dist_pkt_t node_dist_pkt;
        node_dist_pkt.last = false;

        // Send first half (lower 256 bits, containing 8 distances)
        node_dist_pkt.data = wide_word.range(255, 0);
        node_dist_stream.write(node_dist_pkt);
        nodes_sent += num_dists_per_pkt;

        // Send second half (upper 256 bits, containing 8 distances)
        if (nodes_sent < num_nodes) {
            node_dist_pkt.data = wide_word.range(511, 256);
            node_dist_stream.write(node_dist_pkt);
            nodes_sent += num_dists_per_pkt;
        }
    }

    // Send end marker
    node_dist_pkt_t end_pkt;
    end_pkt.last = true;
    node_dist_stream.write(end_pkt);
}

extern "C" void hbm_writer(const bus_word_t *node_distances_ddr,
                           bus_word_t *output, int32_t num_nodes,
                           hls::stream<node_dist_pkt_t> &node_dist_stream,
                           hls::stream<write_burst_pkt_t> &write_burst_stm) {
#pragma HLS INTERFACE m_axi port = node_distances_ddr offset = slave bundle =  \
    gmem0
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = node_distances_ddr bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = num_nodes bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW
    node_distance_loader(node_distances_ddr, node_dist_stream, num_nodes);
    write_out(output, write_burst_stm);
}
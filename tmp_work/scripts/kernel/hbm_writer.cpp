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

extern "C" void hbm_writer(bus_word_t *output,
                           hls::stream<write_burst_pkt_t> &write_burst_stm) {
#pragma HLS INTERFACE m_axi port = node_distances_ddr offset = slave bundle =  \
    gmem0
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW
    write_out(output, write_burst_stm);
}
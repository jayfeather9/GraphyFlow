#include "graphyflow_little.h"

static void node_property_loader(
    const bus_word_t *node_distances_ddr,
    uint32_t dst_num,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream) {

    ppb_request_pkt_t one_ppb_request_pkg;
    ppb_response_pkt_t one_ppb_response_pkg;
    
	littleKernelReadMemory:
    while(true){
#pragma HLS PIPELINE
        if(ppb_req_stream.read_nb(one_ppb_request_pkg)){
			uint32_t request_round = one_ppb_request_pkg.data;
			bool end_flag = one_ppb_request_pkg.last;

			uint32_t  base_addr = request_round << LOG_SRC_BUFFER_SIZE >> 4;

			if(end_flag){
				one_ppb_response_pkg.last = end_flag;
                ppb_resp_stream.write(one_ppb_response_pkg);
				break;
			}
			else{
				for(int i = 0; i < (SRC_BUFFER_SIZE >> 4); i ++){
					int addr = base_addr + i;

					one_ppb_response_pkg.data = node_distances_ddr[addr];
					one_ppb_response_pkg.dest = addr;
					one_ppb_response_pkg.last = false;
                    ppb_resp_stream.write(one_ppb_response_pkg);
				}
			}
		}
    }

    LOOP_LOADER_2:
    cacheline_data_pkt_t cache_data;
    uint32_t total_cachelines =
        (dst_num + DIST_PER_WORD - 1) / DIST_PER_WORD; // Total number of cache lines
    for (uint32_t i = 0; i < total_cachelines; i++) {
#pragma HLS PIPELINE II = 1
        cache_data.data = node_distances_ddr[i];
        cache_data.last = (i == total_cachelines - 1) ? true : false;
        cacheline_data_stream.write(cache_data);
    }
}

void write_out(bus_word_t *output, uint32_t dst_num,
               hls::stream<write_burst_pkt_t> &write_burst_stream) {
    uint32_t write_idx = 0;
    uint32_t target_writes = ((dst_num + DBL_PE_NUM - 1) / DBL_PE_NUM) - 1; // Total number of write bursts
    write_out:
    while (true) {
#pragma HLS PIPELINE II = 1

        write_burst_pkt_t one_write_burst;

        if (write_burst_stream.read_nb(one_write_burst)) {
            output[write_idx] = one_write_burst.data;

            if (write_idx >= target_writes) {
                break;
            }
            write_idx = write_idx + 1;
        }
    }
}

extern "C" void
hbm_writer(bus_word_t *node_props, bus_word_t *output, uint32_t dst_num,
           hls::stream<ppb_request_pkt_t> &ppb_req_stream,
           hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream,
           hls::stream<write_burst_pkt_t> &write_burst_stream) {
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = node_props bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = dst_num bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATAFLOW
    node_property_loader(node_props, dst_num, ppb_req_stream, ppb_resp_stream, cacheline_data_stream);
    write_out(output, dst_num, write_burst_stream);
}
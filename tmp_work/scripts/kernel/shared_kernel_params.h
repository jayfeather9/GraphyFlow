#ifndef __SHARED_KERNEL_PARAMS_H__
#define __SHARED_KERNEL_PARAMS_H__

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
// #include <stdio.h>
#include <string.h>

constexpr int log2_floor(int n) {
    return (n <= 1) ? 0 : 1 + log2_floor(n >> 1);
}

constexpr int PE_NUM = 8;
constexpr int LOG_PE_NUM = log2_floor(PE_NUM);
constexpr int L = 4;
constexpr int SRC_BUFFER_SIZE = 4096;
constexpr int LOG_SRC_BUFFER_SIZE = log2_floor(SRC_BUFFER_SIZE);
constexpr int NODE_ID_BITWIDTH = 32;
constexpr int DISTANCE_BITWIDTH = 8;
constexpr int LOG_DIST_BITWIDTH = log2_floor(DISTANCE_BITWIDTH);
constexpr int AXI_BUS_WIDTH = 512;
constexpr int REDUCE_MEM_WIDTH = 64;

#ifndef INT_DISTANCE
constexpr int DISTANCE_INTEGER_PART = 16;
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
constexpr int INFINITY_DIST = (1 << DISTANCE_INTEGER_PART) - 4;
#else
typedef ap_uint<DISTANCE_BITWIDTH> distance_t;
constexpr int INFINITY_DIST = (1 << DISTANCE_BITWIDTH) - 4;
#endif

constexpr int DIST_PER_WORD = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
constexpr int LOG_DIST_PER_WORD = log2_floor(DIST_PER_WORD);

typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;
typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;
typedef ap_uint<DISTANCE_BITWIDTH> ap_fixed_pod_t;
typedef ap_axiu<512, 0, 0, 0> write_burst_pkt_t;
typedef ap_axiu<32, 0, 0, 8> cacheline_request_pkt_t;
typedef ap_axiu<512, 0, 0, 8> cacheline_response_pkt_t;
typedef ap_axiu<32, 0, 0, 0> ppb_request_pkt_t;
typedef ap_axiu<512, 0, 0, 32> ppb_response_pkt_t;
typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;

extern "C" void
apply_kernel(uint32_t dst_num,
             hls::stream<cacheline_data_pkt_t> &cacheline_data_stream,
             hls::stream<write_burst_pkt_t> &kernel_out_stream,
             hls::stream<write_burst_pkt_t> &write_burst_stream);

extern "C" void
hbm_writer(bus_word_t *node_props_1, bus_word_t *node_props_2,
           bus_word_t *node_props_3, bus_word_t *node_props_4,
           bus_word_t *node_props_5, bus_word_t *output_1, bus_word_t *output_2,
           bus_word_t *output_3, bus_word_t *output_4, bus_word_t *output_5,
           uint32_t dst_num_1, uint32_t dst_num_2, uint32_t dst_num_3,
           uint32_t dst_num_4, uint32_t dst_num_5,
           hls::stream<ppb_request_pkt_t> &ppb_req_stream_1,
           hls::stream<ppb_request_pkt_t> &ppb_req_stream_2,
           hls::stream<ppb_request_pkt_t> &ppb_req_stream_3,
           hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_4,
           hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_5,
           hls::stream<ppb_response_pkt_t> &ppb_resp_stream_1,
           hls::stream<ppb_response_pkt_t> &ppb_resp_stream_2,
           hls::stream<ppb_response_pkt_t> &ppb_resp_stream_3,
           hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_4,
           hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_5,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_1,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_2,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_3,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_4,
           hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_5,
           hls::stream<write_burst_pkt_t> &write_burst_stream_1,
           hls::stream<write_burst_pkt_t> &write_burst_stream_2,
           hls::stream<write_burst_pkt_t> &write_burst_stream_3,
           hls::stream<write_burst_pkt_t> &write_burst_stream_4,
           hls::stream<write_burst_pkt_t> &write_burst_stream_5);

#endif // __SHARED_KERNEL_PARAMS_H__
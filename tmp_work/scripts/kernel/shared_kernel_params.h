#ifndef __SHARED_KERNEL_PARAMS_H__
#define __SHARED_KERNEL_PARAMS_H__

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
// #include <stdio.h>
#include <string.h>

#define PE_NUM 8
#define DBL_PE_NUM 16
#define LOG_PE_NUM 3
#define L 4
#define SRC_BUFFER_SIZE 4096
#define LOG_SRC_BUFFER_SIZE 12

#define NODE_ID_BITWIDTH 32
#define DISTANCE_BITWIDTH 32
#define DISTANCE_INTEGER_PART 16
#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH
#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART
#define OUT_END_MARKER_BITWIDTH 4
#define DIST_PER_WORD 16 // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = 512 / 32 = 16
#define LOG_DIST_PER_WORD                                                      \
    4 // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2(512 / 32) = log2(16) = 4

// --- New Memory Word and Bus Definitions ---
#define AXI_BUS_WIDTH 512

#define REDUCE_MEM_WIDTH 64
typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;

const int INFINITY_DIST = 16384;

#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)

// --- Redefinition of Core Graph Types for HLS ---
// These typedefs override the standard integer types from common.h for
// synthesis.
typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;
typedef ap_uint<32> edge_id_t; // edge_id_t is not customized yet, keep as is.
typedef ap_uint<DISTANCE_BITWIDTH>
    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;
typedef ap_axiu<256, 0, 0, 0> node_dist_pkt_t;
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
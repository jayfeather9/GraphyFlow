#ifndef __SHARED_KERNEL_PARAMS_H__
#define __SHARED_KERNEL_PARAMS_H__

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
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

#define BIG_MERGER_LENGTH 3
#define LITTLE_MERGER_LENGTH 11

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
typedef ap_axiu<64, 0, 0, 0> little_out_pkt_t;
typedef ap_axiu<512, 0, 0, 32> write_burst_w_dst_pkt_t;
typedef ap_axiu<32, 0, 0, 8> cacheline_request_pkt_t;
typedef ap_axiu<512, 0, 0, 8> cacheline_response_pkt_t;
typedef ap_axiu<32, 0, 0, 0> ppb_request_pkt_t;
typedef ap_axiu<512, 0, 0, 32> ppb_response_pkt_t;
typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;

struct in_write_burst_w_dst_pkt_t {
    bus_word_t data;
    ap_uint<32> dest_addr;
    bool end_flag;
};

extern "C" void
apply_kernel(bus_word_t *node_props, uint32_t little_kernel_length,
             uint32_t big_kernel_length, uint32_t little_kernel_st_offset,
             uint32_t big_kernel_st_offset,
             hls::stream<write_burst_pkt_t> &little_kernel_out_stream,
             hls::stream<write_burst_pkt_t> &big_kernel_out_stream,
             hls::stream<write_burst_w_dst_pkt_t> &kernel_out_stream);

extern "C" void
big_merger(hls::stream<write_burst_pkt_t> &big_kernel_1_out_stream,
           hls::stream<write_burst_pkt_t> &big_kernel_2_out_stream,
           hls::stream<write_burst_pkt_t> &big_kernel_3_out_stream,
           hls::stream<write_burst_pkt_t> &kernel_out_stream);

extern "C" void
little_merger(hls::stream<little_out_pkt_t> &little_kernel_1_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_2_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_3_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_4_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_5_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_6_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_7_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_8_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_9_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_10_out_stream,
              hls::stream<little_out_pkt_t> &little_kernel_11_out_stream,
              hls::stream<write_burst_pkt_t> &kernel_out_stream);

extern "C" void hbm_writer(
    bus_word_t *src_prop_1, bus_word_t *src_prop_2, bus_word_t *src_prop_3,
    bus_word_t *src_prop_4, bus_word_t *src_prop_5, bus_word_t *src_prop_6,
    bus_word_t *src_prop_7, bus_word_t *src_prop_8, bus_word_t *src_prop_9,
    bus_word_t *src_prop_10, bus_word_t *src_prop_11, bus_word_t *src_prop_12,
    bus_word_t *src_prop_13, bus_word_t *src_prop_14, bus_word_t *output,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_1,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_1,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_2,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_2,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_3,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_3,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_4,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_4,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_5,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_5,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_6,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_6,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_7,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_7,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_8,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_8,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_9,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_9,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_10,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_10,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_11,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_11,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_1,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_1,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_2,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_2,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_3,
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_3,
    hls::stream<write_burst_w_dst_pkt_t> &write_burst_stream);

#endif // __SHARED_KERNEL_PARAMS_H__
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
#define LOG_PE_NUM 3 // log2_floor(8)
#define L 4
#define SRC_BUFFER_SIZE 4096
#define LOG_SRC_BUFFER_SIZE 12 // log2_floor(4096)
#define NODE_ID_BITWIDTH 32
#define DISTANCE_BITWIDTH 8
#define LOG_DIST_BITWIDTH 3 // log2_floor(8)
#define AXI_BUS_WIDTH 512
#define REDUCE_MEM_WIDTH 64
#define DISTANCES_PER_REDUCE_WORD 8 // 64 / 8
#define LOG_DISTANCES_PER_REDUCE_WORD 3 // log2_floor(8)

#ifndef INT_DISTANCE
#define DISTANCE_INTEGER_PART 16
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
#define INFINITY_DIST 32766 // (1ULL << (16 - 1)) - 2
#else
typedef ap_uint<DISTANCE_BITWIDTH> distance_t;
#define INFINITY_DIST 126 // (1ULL << (8 - 1)) - 2
#endif

#define DIST_PER_WORD 64 // 512 / 8
#define LOG_DIST_PER_WORD 6 // log2_floor(64)

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

extern "C" void hbm_writer(
    bus_word_t *node_props_1,
    //    bus_word_t *node_props_2,
    //    bus_word_t *node_props_3, bus_word_t *node_props_4,
    //    bus_word_t *node_props_5,
    bus_word_t *output_1,
    //    bus_word_t *output_2,
    //    bus_word_t *output_3, bus_word_t *output_4, bus_word_t *output_5,
    uint32_t dst_num_1,
    //    uint32_t dst_num_2, uint32_t dst_num_3,
    //    uint32_t dst_num_4, uint32_t dst_num_5,
    hls::stream<ppb_request_pkt_t> &ppb_req_stream_1,
    //    hls::stream<ppb_request_pkt_t> &ppb_req_stream_2,
    //    hls::stream<ppb_request_pkt_t> &ppb_req_stream_3,
    //    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_1,
    //    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_5,
    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_1,
    //    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_2,
    //    hls::stream<ppb_response_pkt_t> &ppb_resp_stream_3,
    //    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_1,
    //    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_5,
    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_1,
    //    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_2,
    //    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_3,
    //    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_4,
    //    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_5,
    hls::stream<write_burst_pkt_t> &write_burst_stream_1
    //    hls::stream<write_burst_pkt_t> &write_burst_stream_2,
    //    hls::stream<write_burst_pkt_t> &write_burst_stream_3,
    //    hls::stream<write_burst_pkt_t> &write_burst_stream_4,
    //    hls::stream<write_burst_pkt_t> &write_burst_stream_5
);

#endif // __SHARED_KERNEL_PARAMS_H__
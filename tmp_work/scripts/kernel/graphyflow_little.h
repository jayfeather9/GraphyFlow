#ifndef __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__
#define __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
// #include <stdio.h>
#include <string.h>

#define PE_NUM 8
#define LOG_PE_NUM 3 // log2_floor(8)
#ifdef EMULATION
#define MAX_NUM 512
#else
#define MAX_NUM 65536
#endif
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
typedef ap_axiu<32, 0, 0, 0> ppb_request_pkt_t;
typedef ap_axiu<512, 0, 0, 32> ppb_response_pkt_t;
typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;

// --- Struct Type Definitions (UNCHANGED) ---
struct __attribute__((packed)) node_id_burst_t {
    node_id_t data[PE_NUM];
};

struct __attribute__((packed)) distance_req_pack_t {
    node_id_t idx[PE_NUM];
    ap_uint<4> offset;
    bool end_flag;
};

struct __attribute__((packed)) edge_t {
    node_id_t src_id;
    node_id_t dst_id;
};

struct __attribute__((packed)) edge_descriptor_batch_t {
    edge_t edges[PE_NUM];
    uint8_t end_pos;
};

struct __attribute__((packed)) update_tuple_t {
    node_id_t node_id[PE_NUM];
    ap_fixed_pod_t prop[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) net_wrapper_kt_pair_105_t_t {
    node_id_t node_id;
    ap_fixed_pod_t prop;
    bool end_flag;
};

// --- Top-Level Function Prototype ---
extern "C" void
graphyflow_little(const bus_word_t *edge_props, int32_t num_nodes,
                  int32_t num_edges, int32_t dst_num,
                  hls::stream<ppb_request_pkt_t> &ppb_req_stream,
                  hls::stream<ppb_response_pkt_t> &ppb_resp_stream,
                  hls::stream<write_burst_pkt_t> &kernel_out_stream);

#endif // __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__

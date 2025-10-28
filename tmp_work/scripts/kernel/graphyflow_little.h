#ifndef __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__
#define __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__

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
#ifdef EMULATION
constexpr int MAX_NUM = 512;
#else
constexpr int MAX_NUM = 65536;
#endif
constexpr int L = 4;
constexpr int SRC_BUFFER_SIZE = 4096;
constexpr int LOG_SRC_BUFFER_SIZE = log2_floor(SRC_BUFFER_SIZE);
constexpr int NODE_ID_BITWIDTH = 32;
constexpr int DISTANCE_BITWIDTH = 8;
constexpr int LOG_DIST_BITWIDTH = log2_floor(DISTANCE_BITWIDTH);
constexpr int AXI_BUS_WIDTH = 512;
constexpr int REDUCE_MEM_WIDTH = 64;
constexpr int DISTANCES_PER_REDUCE_WORD = REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH;

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

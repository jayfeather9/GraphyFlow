#ifndef __COMMON_H__
#define __COMMON_H__

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <ap_fixed.h>
#include <ap_int.h>
#include <stdint.h>

#ifndef __SYNTHESIS__
#include "xcl2.h"
#endif

// --- Customizable Bitwidth Macros ---
// These macros define the bitwidths for core data types.
// They are used by the host for data packing and by the kernel for synthesis.
#define NODE_ID_BITWIDTH 24
#define DISTANCE_BITWIDTH 24
#define DISTANCE_INTEGER_PART                                                  \
    16 // Number of bits for the integer part of distance
#define WEIGHT_BITWIDTH 24
#define WEIGHT_INTEGER_PART 16 // Number of bits for the integer part of weight

// --- Host-side definition for the AXI bus word ---
#define AXI_BUS_WIDTH 512
#ifndef __SYNTHESIS__
typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<DISTANCE_BITWIDTH> ap_fixed_pod_t;
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
typedef ap_fixed<WEIGHT_BITWIDTH, WEIGHT_INTEGER_PART> weight_t;
#endif

// A constant representing infinity for distance initialization
const int INFINITY_DIST = 16384;

// --- Graph Type Definitions ---
typedef uint32_t edge_id_t;
typedef uint32_t node_id_t;
// typedef uint32_t ap_fixed_pod_t;

// Structure to hold the graph in Compressed Sparse Row (CSR) format
struct GraphCSR {
    int num_vertices;
    int num_edges;
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<int> weights;
    // Map from original global vertex ID to compressed local ID
    std::unordered_map<int, int> vtx_map;
    // Map from compressed local ID to original global vertex ID
    std::unordered_map<int, int> vtx_map_rev;
};

#define KERNEL_OUTPUT_BATCH_TYPE KernelOutputBatch
#define BATCH_TYPE edge_des_burst_t
#define EDGE_TYPE edge_t
#define NODE_TYPE node_t

#define PE_NUM 8

// --- Struct Type Definitions ---
struct __attribute__((packed)) struct_ana_3_t {
    ap_fixed_pod_t ele_0;
    node_id_t ele_1;
    ap_fixed_pod_t ele_2;
};

struct __attribute__((packed)) struct_abu_9_t {
    ap_fixed_pod_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_nbu_11_t {
    node_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ibu_14_t {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_an_15_t {
    ap_fixed_pod_t ele_0;
    node_id_t ele_1;
};

struct __attribute__((packed)) struct_ebu_20_t {
    edge_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) node_with_prop_t {
    ap_fixed_pod_t prop;
    node_id_t node_id;
};

struct __attribute__((packed)) node_distance_burst_t {
    ap_fixed_pod_t data[PE_NUM];
};

struct __attribute__((packed)) edge_batch_t {
    ap_fixed_pod_t weights[PE_NUM];
    ap_fixed_pod_t src_distances[PE_NUM];
    node_id_t dsts[PE_NUM];
    int32_t end_pos;
    bool end_flag;
};

struct __attribute__((packed)) node_dist_batch_t {
    ap_fixed_pod_t data[PE_NUM];
    uint8_t end_pos;
    bool end_flag;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    node_id_t id;
};

struct __attribute__((packed)) struct_sbu_7_t {
    struct_ana_3_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_sbu_17_t {
    struct_an_15_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) internal_end_data_batch_t {
    node_with_prop_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) kt_pair_105_t {
    int32_t key;
    node_with_prop_t transform;
};

struct __attribute__((packed)) struct_nb_58_t {
    node_with_prop_t ele_0;
    bool ele_1;
};

struct __attribute__((packed)) edge_des_burst_t {
    node_with_prop_t edges[PE_NUM];
};

struct __attribute__((packed)) edge_descriptor_batch_t {
    node_with_prop_t edges[PE_NUM];
    int32_t end_pos;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_kbu_50_t {
    kt_pair_105_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) net_wrapper_kt_pair_105_t_t {
    kt_pair_105_t data;
    bool end_flag;
};

#endif // __COMMON_H__
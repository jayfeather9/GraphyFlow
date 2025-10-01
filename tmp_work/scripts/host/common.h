#ifndef __COMMON_H__
#define __COMMON_H__

#include <limits>
#include <string>
#include <vector>

#ifndef __SYNTHESIS__
#include "xcl2.h"
#endif

// A constant representing infinity for distance initialization
const int INFINITY_DIST = 16384;
const int PE_NUM = 8; // Number of Processing Elements

typedef uint16_t node_id_t;
typedef uint16_t edge_id_t;

// Describes a single edge in CSR format for the host and kernel
struct __attribute__((packed)) edge_descriptor_t {
    node_id_t dst_id;
    int32_t weight;
};


struct __attribute__((packed)) edge_des_burst_t {
    edge_descriptor_t edges[PE_NUM];
};

// Structure to hold the graph in Compressed Sparse Row (CSR) format
struct GraphCSR {
    int num_vertices;
    int num_edges;
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<int> weights;
};

#include <ap_fixed.h>
#include <stdint.h>

#define PE_NUM 8

// --- Struct Type Definitions ---
struct __attribute__((packed)) struct_ebu_4_t {
    edge_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ini_7_t {
    int32_t ele_0;
    node_id_t ele_1;
    int32_t ele_2;
};

struct __attribute__((packed)) struct_ibu_14_t {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_nbu_16_t {
    node_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_in_17_t {
    int32_t ele_0;
    node_id_t ele_1;
};

struct __attribute__((packed)) struct_bbu_21_t {
    bool data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    int32_t id;
};

struct __attribute__((packed)) opt_struct_ini_7_t_t {
    struct_ini_7_t data;
    bool valid;
};

struct __attribute__((packed)) struct_sbu_12_t {
    struct_ini_7_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_sbu_19_t {
    struct_in_17_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) kt_pair_141_t {
    int32_t key;
    struct_in_17_t transform;
};

struct __attribute__((packed)) struct_sb_38_t {
    struct_in_17_t ele_0;
    bool ele_1;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_obu_10_t {
    opt_struct_ini_7_t_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) net_wrapper_kt_pair_141_t_t {
    kt_pair_141_t data;
    bool end_flag;
};

struct __attribute__((packed)) struct_kbu_30_t {
    kt_pair_141_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

#endif // __COMMON_H__

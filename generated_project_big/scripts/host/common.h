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
struct __attribute__((packed)) node_t {
    int32_t distance;
    int32_t id;
};

struct __attribute__((packed)) struct_ibu_10_t {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_bbu_15_t {
    bool data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    int32_t id;
};

struct __attribute__((packed)) edge_t {
    int32_t weight;
    node_t src;
    node_t dst;
};

struct __attribute__((packed)) struct_nbu_8_t {
    node_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ini_11_t {
    int32_t ele_0;
    node_t ele_1;
    int32_t ele_2;
};

struct __attribute__((packed)) struct_in_19_t {
    int32_t ele_0;
    node_t ele_1;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ebu_6_t {
    edge_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_sbu_13_t {
    struct_ini_11_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) opt_struct_ini_11_t_t {
    struct_ini_11_t data;
    bool valid;
};

struct __attribute__((packed)) struct_sbu_21_t {
    struct_in_19_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) kt_pair_141_t {
    int32_t key;
    struct_in_19_t transform;
};

struct __attribute__((packed)) struct_sb_39_t {
    struct_in_19_t ele_0;
    bool ele_1;
};

struct __attribute__((packed)) struct_obu_18_t {
    opt_struct_ini_11_t_t data[PE_NUM];
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

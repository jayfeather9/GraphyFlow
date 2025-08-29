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
struct __attribute__((packed)) GraphCSR {
    int num_vertices;
    int num_edges;
    std::vector<int> offsets; // Row pointers (size = num_vertices + 1)
    std::vector<int> columns; // Column indices (size = num_edges)
    std::vector<int> weights; // Edge weights (size = num_edges)
};

#include <ap_fixed.h>
#include <stdint.h>

#define PE_NUM 8

// --- struct Type Definitions (Moved from graphyflow.h) ---
struct __attribute__((packed)) node_t {
    ap_fixed<32, 16> distance;
    int32_t id;
};

struct __attribute__((packed)) edge_t {
    ap_fixed<32, 16> weight;
    node_t src;
    node_t dst;
};

// Input batch structure
struct __attribute__((packed)) struct_ebu_7_t {
    edge_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

// Intermediate structure (ele_0: dist, ele_1: node)
struct __attribute__((packed)) struct_an_20_t {
    ap_fixed<32, 16> ele_0;
    node_t ele_1;
};

// Output batch structure
struct __attribute__((packed)) struct_sbu_22_t {
    struct_an_20_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    int32_t id;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

#endif // __COMMON_H__
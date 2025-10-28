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
constexpr int PE_NUM = 8;
constexpr int DBL_PE_NUM = 16;
constexpr int LOG_PE_NUM = 3;
#ifdef EMULATION
constexpr int MAX_NUM = 512;
#else
constexpr int MAX_NUM = 65536;
#endif
constexpr int L = 4;
constexpr int SRC_BUFFER_SIZE = 4096;
constexpr int LOG_SRC_BUFFER_SIZE = 12;
constexpr int NODE_ID_BITWIDTH = 32;
constexpr int DISTANCE_BITWIDTH = 8;
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

constexpr int log2_floor(int n) {
    return (n <= 1) ? 0 : 1 + log2_floor(n >> 1);
}
constexpr int DIST_PER_WORD = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
constexpr int LOG_DIST_PER_WORD = log2_floor(DIST_PER_WORD);

typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;
typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;
typedef ap_uint<DISTANCE_BITWIDTH> ap_fixed_pod_t;

// Structure to hold the graph in Compressed Sparse Row (CSR) format
struct GraphCSR {
    int num_vertices;
    int num_edges;
    int num_dsts;
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<int> weights;
    // Map from original global vertex ID to compressed local ID
    std::unordered_map<int, int> vtx_map;
    // Map from compressed local ID to original global vertex ID
    std::unordered_map<int, int> vtx_map_rev;
};

#endif // __COMMON_H__
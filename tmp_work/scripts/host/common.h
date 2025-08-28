#ifndef __COMMON_H__
#define __COMMON_H__

#include <vector>
#include <string>
#include <limits>
#include "xcl2.h"

// A constant representing infinity for distance initialization
const int INFINITY_DIST = std::numeric_limits<int>::max();

// Structure to hold the graph in Compressed Sparse Row (CSR) format
struct GraphCSR {
    int num_vertices;
    int num_edges;
    std::vector<int> offsets;  // Row pointers (size = num_vertices + 1)
    std::vector<int> columns;  // Column indices (size = num_edges)
    std::vector<int> weights;  // Edge weights (size = num_edges)
};

#endif // __COMMON_H__
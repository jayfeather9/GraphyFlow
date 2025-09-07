#ifndef __FPGA_BELLMAN_FORD_H__
#define __FPGA_BELLMAN_FORD_H__

#include "common.h"

// This function represents a single iteration of the Bellman-Ford algorithm on the FPGA.
// It sets kernel arguments and enqueues the kernel for execution.
void fpga_bellman_ford_iteration(
    cl::CommandQueue& q,
    cl::Kernel& kernel,
    int num_vertices,
    cl::Buffer& d_offsets,
    cl::Buffer& d_columns,
    cl::Buffer& d_weights,
    cl::Buffer& d_distances,
    cl::Buffer& d_stop_flag,
    cl::Event& event
);

#endif // __FPGA_BELLMAN_FORD_H__
#ifndef __FPGA_HANDLER_H__
#define __FPGA_HANDLER_H__

#include "common.h"

// Main function to run the Bellman-Ford algorithm on the FPGA.
// It orchestrates data transfer and iterative kernel execution.
std::vector<int> run_fpga_graph(
    const std::string& xclbin_path, 
    const GraphCSR& graph, 
    int start_node,
    double& total_kernel_time_sec
);

#endif // __FPGA_HANDLER_H__
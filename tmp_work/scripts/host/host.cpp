#include <iostream>
#include <vector>
#include <string>
#include "common.h"
#include "graph_loader.h"
#include "fpga_handler.h"
#include "host_verifier.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <xclbin_file> <graph_data_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbin_file = argv[1];
    std::string graph_file = argv[2];
    int start_node = 0; // Assume starting from node 0

    // 1. Load graph data
    std::cout << "--- Step 1: Loading Graph Data ---" << std::endl;
    GraphCSR graph = load_graph_from_file(graph_file);
    if (graph.num_vertices == 0) {
        return EXIT_FAILURE;
    }

    // 2. Run on FPGA
    std::cout << "\n--- Step 2: Running on FPGA ---" << std::endl;
    double total_kernel_time_sec = 0;
    std::vector<int> fpga_distances = run_fpga_graph(xclbin_file, graph, start_node, total_kernel_time_sec);

    // 3. Verify on Host
    std::cout << "\n--- Step 3: Verifying on Host CPU ---" << std::endl;
    std::vector<int> host_distances = verify_on_host(graph, start_node);

    // 4. Compare results
    std::cout << "\n--- Step 4: Comparing Results ---" << std::endl;
    int error_count = 0;
    for (int i = 0; i < graph.num_vertices; ++i) {
        if (fpga_distances[i] != host_distances[i]) {
            if (error_count < 10) { // Print first 10 errors
                 std::cout << "Mismatch at vertex " << i << ": "
                          << "FPGA_Result = " << fpga_distances[i] << ", "
                          << "Host_Result = " << host_distances[i] << std::endl;
            }
            error_count++;
        }
    }

    // 5. Final Report
    std::cout << "\n--- Final Report ---" << std::endl;
    if (error_count == 0) {
        std::cout << "SUCCESS: Results match!" << std::endl;
    } else {
        std::cout << "FAILURE: Found " << error_count << " mismatches." << std::endl;
    }
    
    std::cout << "Total FPGA Kernel Execution Time: " << total_kernel_time_sec * 1000.0 << " ms" << std::endl;
    
    double average_mteps = (double)graph.num_edges / (total_kernel_time_sec / (double)graph.num_vertices) / 1.0e6;
    
    std::cout << "Total MTEPS (Edges / Total Time): " 
              << ((double)graph.num_edges * (double)graph.num_vertices) / total_kernel_time_sec / 1.0e6 << " MTEPS"
              << std::endl;

    return (error_count == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
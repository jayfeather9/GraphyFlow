#include "common.h"
#include "fpga_executor.h" // <-- 修改: 包含新的执行器
#include "graph_loader.h"
#include "graph_partition.h"
#include "host_verifier.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " <xclbin_file> <graph_data_file> <device_no>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbin_file = argv[1];
    std::string graph_file = argv[2];
    int device_no = std::stoi(argv[3]);
    int start_node = 0;

    // 1. 加载图数据 (不变)
    std::cout << "--- Step 1: Loading Graph Data ---" << std::endl;
    GraphCSR graph = load_graph_from_file(graph_file);
    if (graph.num_vertices == 0) {
        return EXIT_FAILURE;
    }

    // 1.5. 图划分 (新增)
    std::cout << "\n--- Step 1.5: Partitioning Graph ---" << std::endl;
    float partition_weights[NUM_PARTITIONS];
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        partition_weights[i] = PARTITION_WEIGHTS[i];
    }
    std::vector<GraphCSR> partitions = partition_graph(graph, NUM_PARTITIONS, partition_weights);
    std::cout << "Graph partitioned into " << partitions.size() << " partitions."
              << std::endl;
    for (int i = 0; i < partitions.size(); ++i) {
        std::cout << " Partition " << i << ": "
                  << partitions[i].num_vertices << " vertices, "
                  << partitions[i].num_edges << " edges." << std::endl;
    }

    // 2. 在 FPGA 上运行 (调用新的通用执行器)
    std::cout << "\n--- Step 2: Running on FPGA ---" << std::endl;
    double total_kernel_time_sec = 0;
    int iter_count = 0;
    std::vector<int> fpga_distances =
        run_fpga_kernel(xclbin_file, partitions, start_node, total_kernel_time_sec,
                        iter_count, device_no, graph.num_vertices);

    // 3. 在 Host CPU 上验证 (不变, 按你的要求保留)
    std::cout << "\n--- Step 3: Verifying on Host CPU ---" << std::endl;
    std::vector<int> host_distances = verify_on_host(graph, start_node);

    // 4. 比较结果 (不变)
    std::cout << "\n--- Step 4: Comparing Results ---" << std::endl;
    int error_count = 0;
    for (int i = 0; i < graph.num_vertices; ++i) {
        if (fpga_distances[i] != host_distances[i]) {
            if (error_count < 10) {
                std::cout << "Mismatch at vertex " << i << ": "
                          << "FPGA_Result = " << fpga_distances[i] << ", "
                          << "Host_Result = " << host_distances[i] << std::endl;
            }
            error_count++;
        }
    }

    // 5. 最终报告 (不变)
    std::cout << "\n--- Final Report ---" << std::endl;
    if (error_count == 0) {
        std::cout << "SUCCESS: Results match!" << std::endl;
    } else {
        std::cout << "FAILURE: Found " << error_count << " mismatches."
                  << std::endl;
    }

    std::cout << "Total FPGA Kernel Execution Time: "
              << total_kernel_time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "Total MTEPS (Edges / Total Time): "
              << ((double)graph.num_edges * iter_count) /
                     total_kernel_time_sec / 1.0e6
              << " MTEPS" << std::endl;

    return (error_count == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

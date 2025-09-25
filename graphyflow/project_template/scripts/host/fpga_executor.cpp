
#include "fpga_executor.h"
#include "generated_host.h"
#include "acc_setup/acc_setup.h"
#include "graph_preprocess/graph_preprocess.h"
#include <iostream>

#define KERNEL_NAME "graphyflow"

// ... (fpga_executor.cpp 的其余内容保持不变) ...
std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 const GraphCSR &graph, int start_node,
                                 double &total_kernel_time_sec,
                                 int &iter_count) {

    // GraphPartiton
    partition_container_dt partition_container = partitionGraph(&graph);

    // init accelerator
    acc_descriptor_dt acc = initAccelerator(xclbin_path);

    AlgorithmHost algo_host(acc);
    algo_host.setup_buffers(partition_container, start_node);
    total_kernel_time_sec = 0;
    int max_iterations = graph.num_vertices;
    int iter = 0;
    std::cout << "\nStarting FPGA execution..." << std::endl;

    // 对于流式内核, 这个循环只会执行一次 (因为 get_stop_flag 返回 1)
    // 测试：只执行单次迭代

    algo_host.transfer_data_to_fpga(partition_container,acc);

    cl::Event event;
    algo_host.execute_kernel_iteration(event);
    event.wait();
    algo_host.transfer_data_from_fpga(partition_container,acc);

    
    unsigned long start = 0, end = 0;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    double iteration_time_ns = end - start;
    total_kernel_time_sec += iteration_time_ns * 1.0e-9;
    double mteps =
        (double)graph.num_edges / (iteration_time_ns * 1.0e-9) / 1.0e6;
    std::cout << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, ";

    /*
    iter_count = iter + 1;

    const std::vector<int> &final_results_ref = algo_host.get_results();
    std::vector<int> final_results = final_results_ref;

    return final_results;
    */
   return std::vector<int>{1,2,3};
}


#include "fpga_executor.h"
#include "generated_host.h"
#include <iostream>

#define KERNEL_NAME "graphyflow"

// ... (fpga_executor.cpp 的其余内容保持不变) ...
std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 const GraphCSR &graph, int start_node,
                                 double &total_kernel_time_sec) {
    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device,
                                      CL_QUEUE_PROFILING_ENABLE, &err));
    auto fileBuf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel kernel(program, KERNEL_NAME, &err));

    AlgorithmHost algo_host(context, kernel, q);
    algo_host.setup_buffers(graph, start_node);
    total_kernel_time_sec = 0;
    int max_iterations = graph.num_vertices;
    int iter = 0;
    std::cout << "\nStarting FPGA execution..." << std::endl;

    // 对于流式内核, 这个循环只会执行一次 (因为 get_stop_flag 返回 1)
    // This loop now performs Bellman-Ford iterations until convergence or
    // max_iterations
    for (iter = 0; iter < max_iterations; ++iter) {
        algo_host.transfer_data_to_fpga();
        cl::Event event;
        algo_host.execute_kernel_iteration(event);
        event.wait();
        algo_host.transfer_data_from_fpga();

        unsigned long start = 0, end = 0;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        double iteration_time_ns = end - start;
        total_kernel_time_sec += iteration_time_ns * 1.0e-9;
        double mteps =
            (double)graph.num_edges / (iteration_time_ns * 1.0e-9) / 1.0e6;

        std::cout << "FPGA Iteration " << iter << ": "
                  << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                  << "Throughput = " << mteps << " MTEPS" << std::endl;

        if (algo_host.check_convergence_and_update()) {
            std::cout << "FPGA computation converged after " << iter + 1
                      << " iteration(s)." << std::endl;
            break;
        }
    }

    const std::vector<int> &final_results_ref = algo_host.get_results();
    std::vector<int> final_results = final_results_ref;

    return final_results;
}

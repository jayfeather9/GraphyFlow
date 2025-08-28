#include "fpga_executor.h"
#include "generated_host.h" // 包含由 Python 生成的 Host 逻辑
#include <iostream>

// 内核名称现在应该由 Python 生成并放在一个配置文件中
// 为了当前步骤，我们暂时硬编码在这里，最终会移到 generated_host.h
#define KERNEL_NAME "bellman_ford_kernel"

std::vector<int> run_fpga_kernel(
    const std::string& xclbin_path,
    const GraphCSR& graph,
    int start_node,
    double& total_kernel_time_sec)
{
    // 1. 设置 OpenCL 环境 (固定逻辑)
    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    auto fileBuf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel kernel(program, KERNEL_NAME, &err));

    // 2. 实例化算法相关的 Host 逻辑
    AlgorithmHost algo_host(context, kernel, q);

    // 3. 创建并初始化 Host/Device 内存 (调用可变逻辑)
    algo_host.setup_buffers(graph, start_node);

    total_kernel_time_sec = 0;
    int max_iterations = graph.num_vertices;
    int iter = 0;

    std::cout << "\nStarting FPGA execution..." << std::endl;

    // 4. 主执行循环 (固定逻辑)
    while (iter < max_iterations) {
        // 将数据写入 FPGA (调用可变逻辑)
        algo_host.transfer_data_to_fpga();

        // 执行 Kernel (调用可变逻辑)
        cl::Event event;
        algo_host.execute_kernel_iteration(event);
        event.wait();

        // 读回终止标志 (调用可变逻辑)
        algo_host.transfer_data_from_fpga();

        // 性能分析 (固定逻辑)
        unsigned long start = 0, end = 0;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        double iteration_time_ns = end - start;
        total_kernel_time_sec += iteration_time_ns * 1.0e-9;
        double mteps = (double)graph.num_edges / (iteration_time_ns * 1.0e-9) / 1.0e6;

        std::cout << "FPGA Iteration " << iter << ": "
                  << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                  << "Throughput = " << mteps << " MTEPS" << std::endl;

        iter++;

        // 检查终止条件 (调用可变逻辑)
        if (algo_host.get_stop_flag() == 1) {
            std::cout << "FPGA computation converged after " << iter << " iterations." << std::endl;
            break;
        }
    }

    // 5. 获取最终结果 (调用可变逻辑)
    // 注意：get_results() 返回一个 const 引用，需要复制一份来返回
    const std::vector<int>& final_results_ref = algo_host.get_results();
    std::vector<int> final_results = final_results_ref;
    
    return final_results;
}

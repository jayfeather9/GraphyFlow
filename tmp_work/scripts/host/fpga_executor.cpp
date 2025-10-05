
#include "fpga_executor.h"
#include "generated_host.h"
#include <iostream>
#include <chrono>

#define KERNEL_NAME "graphyflow_kernel"

// ... (fpga_executor.cpp 的其余内容保持不变) ...
std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 std::vector<GraphCSR> &graphs,
                                 int start_node,
                                 double &total_kernel_time_sec, int &iter_count,
                                 int device_no, int max_iterations) {
    cl_int err;
    auto devices = xcl::get_xil_devices();
    if (device_no >= devices.size()) {
        std::cerr << "Error: Invalid device number " << device_no << ". Only "
                  << devices.size() << " device(s) available." << std::endl;
        exit(EXIT_FAILURE);
    }
    auto device = devices[device_no];
    cl::Context context;
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    cl::CommandQueue q;
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                      CL_QUEUE_PROFILING_ENABLE, &err));
    // create global controller queue
    cl::CommandQueue q_glb;
    OCL_CHECK(err, q_glb = cl::CommandQueue(context, device,
                                      CL_QUEUE_PROFILING_ENABLE, &err));
    // create each graphyflow kernel queue
    std::vector<cl::CommandQueue> q_graphyflow;
    q_graphyflow.resize(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        OCL_CHECK(err, q_graphyflow[i] = cl::CommandQueue(context, device,
                                          CL_QUEUE_PROFILING_ENABLE, &err));
    }
    // read binary file
    auto fileBuf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    cl::Program program;
    OCL_CHECK(err, program = cl::Program(context, {device}, bins, NULL, &err));
    
    // create global controller kernel
    cl::Kernel kernel_glb;
    OCL_CHECK(err, kernel_glb = cl::Kernel(program, "global_controller:{global_controller_1}", &err));
    // create each graphyflow kernel
    std::vector<cl::Kernel> kernels_graphyflow;
    kernels_graphyflow.resize(NUM_PARTITIONS);
    for (int i = 0; i < NUM_PARTITIONS; ++i) {
        std::string k_id = std::to_string(i + 1);
        // std::string krnl_name_full = acc.big_gs_kernel_name + ":{" + "bigKernelScatterGather_" + cu_id + "}";
        std::string kernel_name = KERNEL_NAME;
        std::string krnl_name_full = kernel_name + ":{" + kernel_name + "_" + k_id + "}";
        printf("Creating kernel %s for partition %d\n", krnl_name_full.c_str(), i);
        OCL_CHECK(err, kernels_graphyflow[i] = cl::Kernel(program, krnl_name_full.c_str(), &err));
    }

    // TODO: modify AlgorithmHost to support multiple partitions

    AlgorithmHost algo_host(context, kernel_glb, kernels_graphyflow,
                            q, q_glb, q_graphyflow, graphs);
    algo_host.setup_buffers(start_node);
    total_kernel_time_sec = 0;
    int iter = 0;
    std::cout << "\nStarting FPGA execution..." << std::endl;

    // 对于流式内核, 这个循环只会执行一次 (因为 get_stop_flag 返回 1)
    // This loop now performs Bellman-Ford iterations until convergence or
    // max_iterations
    for (iter = 0; iter < max_iterations; ++iter) {
        algo_host.transfer_data_to_fpga();
        cl::Event event_glb;
        std::vector<cl::Event> events_graphyflow(NUM_PARTITIONS);
        algo_host.execute_kernel_iteration(event_glb, events_graphyflow);
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_PARTITIONS; i++) {
            algo_host.m_q_graphyflow[i].finish();
        }
        algo_host.m_q_glb.finish();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        double iteration_time_sec = elapsed.count();
        algo_host.transfer_data_from_fpga();

        // profile each kernel
        double max_time_ns = 0;
        long total_edges = 0;
        std::vector<double> kernel_times_ns(NUM_PARTITIONS);

        for (int i = 0; i < NUM_PARTITIONS; i++) {
            unsigned long start = 0, end = 0;
            events_graphyflow[i].getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            events_graphyflow[i].getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            double iteration_time_ns = end - start;
            kernel_times_ns[i] = iteration_time_ns;
            if (iteration_time_ns > max_time_ns) {
                max_time_ns = iteration_time_ns;
            }
            total_edges += graphs[i].num_edges;
        }
        total_kernel_time_sec += max_time_ns * 1.0e-9;

        std::cout << "FPGA Iteration " << iter << ":" << std::endl;
        for (int i = 0; i < NUM_PARTITIONS; i++) {
            double mteps = (double)graphs[i].num_edges / (kernel_times_ns[i] * 1.0e-9) / 1.0e6;
            double utilization = kernel_times_ns[i] / max_time_ns;
            std::cout << "  Kernel " << i << ": Time = " << (kernel_times_ns[i] * 1.0e-6) << " ms, "
                      << "MTEPS = " << mteps << ", "
                      << "Utilization = " << utilization << std::endl;
        }
        // profile end
    
        double mteps =
            (double)total_edges / iteration_time_sec / 1.0e6;

        std::cout << "Overall throughput = " << mteps << " MTEPS" << std::endl;

        // overall throughput with max_time_ns
        double overall_mteps = (double)total_edges / (max_time_ns * 1.0e-9) / 1.0e6;
        std::cout << "Overall throughput (max kernel time) = " << overall_mteps << " MTEPS" << std::endl;

        if (algo_host.check_convergence_and_update()) {
            std::cout << "FPGA computation converged after " << iter + 1
                      << " iteration(s)." << std::endl;
            iter_count = iter + 1;
            break;
        }
    }

    iter_count = iter + 1;

    const std::vector<int> &final_results_ref = algo_host.get_results();
    std::vector<int> final_results = final_results_ref;

    return final_results;
}


#include "fpga_executor.h"
#include "acc_setup/acc_setup.h"
#include "generated_host.h"
#include "graph_preprocess/graph_preprocess.h"
#include <chrono>
#include <iostream>

#define KERNEL_NAME "graphyflow"

std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 const GraphCSR &graph, int start_node,
                                 double &total_kernel_time_sec,
                                 int &iter_count) {

    // GraphPartiton
    PartitionContainer partition_container = partitionGraph(&graph);

    // init accelerator
    AccDescriptor acc = initAccelerator(xclbin_path);

    AlgorithmHost algo_host(acc);
    algo_host.prepare_data(partition_container, start_node);
    algo_host.setup_buffers(partition_container);
    total_kernel_time_sec = 0;
    double current_kernel_time_sec = 0;
    int max_iterations = graph.num_vertices;
    int iter = 0;
    std::cout << "\nStarting FPGA execution..." << std::endl;

    for (iter = 0; iter < max_iterations; ++iter) {
        auto iteration_start = std::chrono::high_resolution_clock::now();

        algo_host.update_data(partition_container);
        algo_host.transfer_data_to_fpga(partition_container);
        std::vector<cl::Event> big_kernel_events(acc.num_big_krnl),
            little_kernel_events(acc.num_little_krnl),
            little_writer_events(acc.num_little_krnl),
            big_writer_events(acc.num_big_krnl);

        std::cout << "--- [Host] Phase 3: Enqueuing kernel tasks ---"
                  << std::endl;

        algo_host.execute_kernel_iteration(
            partition_container, big_kernel_events, little_kernel_events,
            little_writer_events, big_writer_events);
        auto kernel_enqueue_start = std::chrono::high_resolution_clock::now();

        // Wait for all kernels to finish
        for (auto &q : acc.big_gs_queue)
            q.finish();
        // auto big_finish = std::chrono::high_resolution_clock::now();
        for (auto &q : acc.little_gs_queue)
            q.finish();
        // auto apply_finish = std::chrono::high_resolution_clock::now();
        for (auto &q : acc.hbm_writer_little_queue)
            q.finish();
        for (auto &q : acc.hbm_writer_big_queue)
            q.finish();

        auto kernel_finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> end_to_end_time =
            kernel_finish - kernel_enqueue_start;

        algo_host.transfer_data_from_fpga();

        // iv. 性能统计
        int cnt = 0;
        for (auto &event : big_kernel_events) {
            unsigned long start = 0, end = 0;
            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            double iteration_time_ns = end - start;
            current_kernel_time_sec =
                std::max(current_kernel_time_sec, iteration_time_ns * 1.0e-9);
            double mteps = (double)partition_container.SPs[cnt].num_edges /
                           (iteration_time_ns * 1.0e-9) / 1.0e6;

            std::cout << "FPGA Iteration " << iter << ": "
                      << "Big Kernel " << cnt++ << ", "
                      << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                      << "Throughput = " << mteps << " MTEPS" << std::endl;
        }
        cnt = 0;
        for (auto &event : little_kernel_events) {
            unsigned long start = 0, end = 0;
            event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
            event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
            double iteration_time_ns = end - start;
            current_kernel_time_sec =
                std::max(current_kernel_time_sec, iteration_time_ns * 1.0e-9);
            double mteps = (double)partition_container.DPs[cnt].num_edges /
                           (iteration_time_ns * 1.0e-9) / 1.0e6;

            std::cout << "FPGA Iteration " << iter << ": "
                      << "Little Kernel " << cnt++ << ", "
                      << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                      << "Throughput = " << mteps << " MTEPS" << std::endl;
        }

        for (size_t idx = 0; idx < little_writer_events.size(); ++idx) {
            unsigned long start = 0, end = 0;
            little_writer_events[idx].getProfilingInfo(
                CL_PROFILING_COMMAND_START, &start);
            little_writer_events[idx].getProfilingInfo(CL_PROFILING_COMMAND_END,
                                                       &end);
            double iteration_time_ns = end - start;
            current_kernel_time_sec =
                std::max(current_kernel_time_sec, iteration_time_ns * 1.0e-9);

            std::cout << "FPGA Iteration " << iter << ": "
                      << "HBM Writer Little Kernel " << idx << ", "
                      << "Time = " << (iteration_time_ns * 1.0e-6) << " ms"
                      << std::endl;
        }

        for (size_t idx = 0; idx < big_writer_events.size(); ++idx) {
            unsigned long start = 0, end = 0;
            big_writer_events[idx].getProfilingInfo(CL_PROFILING_COMMAND_START,
                                                    &start);
            big_writer_events[idx].getProfilingInfo(CL_PROFILING_COMMAND_END,
                                                    &end);
            double iteration_time_ns = end - start;
            current_kernel_time_sec =
                std::max(current_kernel_time_sec, iteration_time_ns * 1.0e-9);

            std::cout << "FPGA Iteration " << iter << ": "
                      << "HBM Writer Big Kernel " << idx << ", "
                      << "Time = " << (iteration_time_ns * 1.0e-6) << " ms"
                      << std::endl;
        }

        double iteration_time_ns = current_kernel_time_sec * 1.0e9;
        total_kernel_time_sec += end_to_end_time.count();
        current_kernel_time_sec = 0;
        double mteps =
            (double)graph.num_edges / (iteration_time_ns * 1.0e-9) / 1.0e6;

        std::cout << "FPGA Iteration " << iter << ": "
                  << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                  << "Throughput = " << mteps << " MTEPS" << std::endl;

        // Print end-to-end timing for the iteration
        std::cout << "FPGA Iteration " << iter
                  << " End-to-End Time: " << (end_to_end_time.count() * 1000.0)
                  << " ms" << " Throughput = "
                  << (double)graph.num_edges / end_to_end_time.count() / 1.0e6
                  << " MTEPS" << std::endl;

        // v. 检查是否收敛。如果未收敛，此函数会更新 partition_container
        // 为下次迭代做准备
        if (algo_host.check_convergence_and_update(partition_container)) {
            std::cout << "FPGA computation converged after " << iter + 1
                      << " iteration(s)." << std::endl;
            break;
        }
    }

    iter_count = iter + 1;

    const std::vector<int> &final_results_ref = algo_host.get_results();
    std::vector<int> final_results = final_results_ref;

    return final_results;
}

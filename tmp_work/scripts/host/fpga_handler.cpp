#include "fpga_handler.h"
#include "fpga_bellman_ford.h"
#include <iostream>
#include <chrono>

std::vector<int> run_fpga_graph(
    const std::string& xclbin_path, 
    const GraphCSR& graph, 
    int start_node,
    double& total_kernel_time_sec) 
{
    // Boilerplate OpenCL setup
    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    auto fileBuf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));
    
    // Kernel name must match the name in your Vitis HLS/RTL code
    OCL_CHECK(err, cl::Kernel kernel(program, "bellman_ford_kernel", &err));

    // Allocate host memory
    std::vector<int> distances(graph.num_vertices, INFINITY_DIST);
    distances[start_node] = 0;
    
    // This flag will be updated by the kernel to signal completion
    std::vector<int> stop_flag(1, 0);

    // Allocate device buffers
    OCL_CHECK(err, cl::Buffer d_offsets(context, CL_MEM_READ_ONLY, graph.offsets.size() * sizeof(int), NULL, &err));
    OCL_CHECK(err, cl::Buffer d_columns(context, CL_MEM_READ_ONLY, graph.columns.size() * sizeof(int), NULL, &err));
    OCL_CHECK(err, cl::Buffer d_weights(context, CL_MEM_READ_ONLY, graph.weights.size() * sizeof(int), NULL, &err));
    OCL_CHECK(err, cl::Buffer d_distances(context, CL_MEM_READ_WRITE, distances.size() * sizeof(int), NULL, &err));
    OCL_CHECK(err, cl::Buffer d_stop_flag(context, CL_MEM_READ_WRITE, stop_flag.size() * sizeof(int), NULL, &err));

    // Transfer graph data to FPGA (once)
    OCL_CHECK(err, q.enqueueWriteBuffer(d_offsets, CL_TRUE, 0, graph.offsets.size() * sizeof(int), graph.offsets.data(), nullptr, nullptr));
    OCL_CHECK(err, q.enqueueWriteBuffer(d_columns, CL_TRUE, 0, graph.columns.size() * sizeof(int), graph.columns.data(), nullptr, nullptr));
    OCL_CHECK(err, q.enqueueWriteBuffer(d_weights, CL_TRUE, 0, graph.weights.size() * sizeof(int), graph.weights.data(), nullptr, nullptr));
    
    total_kernel_time_sec = 0;
    int max_iterations = graph.num_vertices; // Max iterations to prevent infinite loops
    int iter = 0;

    std::cout << "\nStarting FPGA execution..." << std::endl;

    // Main execution loop
    while (iter < max_iterations) {
        stop_flag[0] = 1; // Assume no changes will be made

        // Transfer current distances and stop_flag to FPGA
        OCL_CHECK(err, q.enqueueWriteBuffer(d_distances, CL_TRUE, 0, distances.size() * sizeof(int), distances.data(), nullptr, nullptr));
        OCL_CHECK(err, q.enqueueWriteBuffer(d_stop_flag, CL_TRUE, 0, stop_flag.size() * sizeof(int), stop_flag.data(), nullptr, nullptr));

        // Execute one iteration on FPGA
        cl::Event event;
        fpga_bellman_ford_iteration(q, kernel, graph.num_vertices, d_offsets, d_columns, d_weights, d_distances, d_stop_flag, event);
        
        // Wait for kernel to finish and read back the stop_flag
        OCL_CHECK(err, q.enqueueReadBuffer(d_stop_flag, CL_TRUE, 0, stop_flag.size() * sizeof(int), stop_flag.data(), nullptr, nullptr));
        
        // Profile execution time
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

        // Termination condition
        if (stop_flag[0] == 1) {
            std::cout << "FPGA computation converged after " << iter << " iterations." << std::endl;
            break;
        }
    }
    
    // Read final distances from FPGA
    OCL_CHECK(err, q.enqueueReadBuffer(d_distances, CL_TRUE, 0, distances.size() * sizeof(int), distances.data(), nullptr, nullptr));
    
    return distances;
}
#include "fpga_bellman_ford.h"

void fpga_bellman_ford_iteration(
    cl::CommandQueue& q,
    cl::Kernel& kernel,
    int num_vertices,
    cl::Buffer& d_offsets,
    cl::Buffer& d_columns,
    cl::Buffer& d_weights,
    cl::Buffer& d_distances,
    cl::Buffer& d_stop_flag,
    cl::Event& event)
{
    cl_int err;
    int arg_idx = 0;

    // --- USER MODIFIABLE SECTION: Set Kernel Arguments ---
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, num_vertices));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, d_offsets));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, d_columns));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, d_weights));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, d_distances));
    OCL_CHECK(err, err = kernel.setArg(arg_idx++, d_stop_flag));
    // --- END USER MODIFIABLE SECTION ---

    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(kernel, nullptr, &event));
}
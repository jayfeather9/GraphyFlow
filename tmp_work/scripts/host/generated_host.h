#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__

#include "common.h"
#include "xcl2.h"
#include <vector>

class AlgorithmHost {
  public:
    AlgorithmHost(cl::Context &context, cl::Kernel &kernel,
                  cl::CommandQueue &q);
    void setup_buffers(const GraphCSR &graph, int start_node);
    void transfer_data_to_fpga();
    void execute_kernel_iteration(cl::Event &event);
    void transfer_data_from_fpga();
    bool check_convergence_and_update();
    const std::vector<int> &get_results() const;

  private:
    // OpenCL-related objects
    cl::Context &m_context;
    cl::Kernel &m_kernel;
    cl::CommandQueue &m_q;

    // Algorithm state
    int m_num_vertices;

    // Host-side memory buffers for CSR graph representation (aligned for DMA)
    std::vector<int, aligned_allocator<int>> h_src_offsets;
    std::vector<edge_descriptor_t, aligned_allocator<edge_descriptor_t>>
        h_edge_descriptors;
    std::vector<int, aligned_allocator<int>> h_node_distances;

    // Host-side buffer for kernel output
    std::vector<KernelOutputBatch, aligned_allocator<KernelOutputBatch>>
        h_o_0_342;

    // Device-side OpenCL buffer handles
    cl::Buffer d_src_offsets;
    cl::Buffer d_edge_descriptors;
    cl::Buffer d_node_distances;
    cl::Buffer d_o_0_342;
};

#endif // __GENERATED_HOST_H__
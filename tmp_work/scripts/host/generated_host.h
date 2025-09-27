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
    size_t m_num_batches;

    // Host-side memory buffers (aligned for efficient DMA)
    // These now use the POD versions of the structs defined in common.h
    std::vector<struct_ebu_4_t, aligned_allocator<struct_ebu_4_t>>
        h_i_0_edge_id_320;
    std::vector<KernelOutputBatch, aligned_allocator<KernelOutputBatch>>
        h_o_0_342;
    std::vector<int, aligned_allocator<int>> h_stop_flag;

    // Host-side state for iterative algorithm (Bellman-Ford)
    std::vector<ap_fixed<32, 16>> h_distances;

    // Device-side OpenCL buffer handles
    cl::Buffer d_i_0_edge_id_320;
    cl::Buffer d_o_0_342;
    cl::Buffer d_stop_flag;
};

#endif // __GENERATED_HOST_H__
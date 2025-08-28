
#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__

#include "common.h"
#include "graphyflow.h" // 包含内核数据类型定义
#include <vector>

class AlgorithmHost {
  public:
    AlgorithmHost(cl::Context &context, cl::Kernel &kernel,
                  cl::CommandQueue &q);
    void setup_buffers(const GraphCSR &graph, int start_node);
    void transfer_data_to_fpga();
    void execute_kernel_iteration(cl::Event &event);
    void transfer_data_from_fpga();

    // New methods for iteration control
    bool check_convergence_and_update();
    const std::vector<int> &get_results() const;

  private:
    cl::Context &m_context;
    cl::Kernel &m_kernel;
    cl::CommandQueue &m_q;
    int m_num_vertices;

    // Host-side memory
    std::vector<struct_ebu_7_t, aligned_allocator<struct_ebu_7_t>> h_i_0_20;
    std::vector<struct_sbu_22_t, aligned_allocator<struct_sbu_22_t>> h_o_0_176;

    // Host memory for stop flag
    std::vector<int, aligned_allocator<int>> h_stop_flag;

    // Device-side OpenCL buffers
    cl::Buffer d_i_0_20;
    cl::Buffer d_o_0_176;
    cl::Buffer d_stop_flag; // New buffer for stop flag

    // Host-side state for Bellman-Ford
    std::vector<ap_fixed<32, 16>> h_distances;
    size_t m_num_batches;
};

#endif // __GENERATED_HOST_H__

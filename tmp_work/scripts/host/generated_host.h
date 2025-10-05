#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__

#include "common.h"
#include "xcl2.h"
#include <ap_fixed.h>
#include <vector>

class MemoryBufferHost {
  public:
      std::vector<int, aligned_allocator<int>> h_src_offsets;
      std::vector<edge_des_burst_t, aligned_allocator<edge_des_burst_t>>
          h_edge_desc_bursts;
      std::vector<int, aligned_allocator<int>> h_node_distances;
      std::vector<KernelOutputBatch, aligned_allocator<KernelOutputBatch>> h_outputs;
};

class MemoryBufferDevice {
  public:
      cl::Buffer d_src_offsets;
      cl::Buffer d_edge_desc_bursts;
      cl::Buffer d_node_distances;
      cl::Buffer d_outputs;
};

class AlgorithmHost {
  public:
    AlgorithmHost(cl::Context &context,
                  cl::Kernel &kernel_glb,
                  std::vector<cl::Kernel> &kernels_graphyflow,
                  cl::CommandQueue &q,
                  cl::CommandQueue &q_glb,
                  std::vector<cl::CommandQueue> &q_graphyflow,
                  std::vector<GraphCSR> &graphs);
    void setup_buffers(int start_node);
    void transfer_data_to_fpga();
    void execute_kernel_iteration(cl::Event &event_glb, std::vector<cl::Event> &events_graphyflow);
    void transfer_data_from_fpga();
    bool check_convergence_and_update();
    const std::vector<int> &get_results() const;

    // OpenCL-related objects
    cl::Context &m_context;
    cl::Kernel &m_kernel_glb;
    std::vector<cl::Kernel> &m_kernels_graphyflow;
    cl::CommandQueue &m_q;
    cl::CommandQueue &m_q_glb;
    std::vector<cl::CommandQueue> &m_q_graphyflow;
    std::vector<GraphCSR> &m_graphs;

    // Host-side memory buffers for CSR graph representation & outputs
    std::vector<MemoryBufferHost> h_memory_buffers;

    // Device-side OpenCL buffer handles
    std::vector<MemoryBufferDevice> d_memory_buffers;
};

#endif // __GENERATED_HOST_H__
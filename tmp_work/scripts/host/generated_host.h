
#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__

#include "common.h"
#include "graphyflow.h" // 包含内核数据类型定义
#include <vector>

class AlgorithmHost {
public:
    AlgorithmHost(cl::Context& context, cl::Kernel& kernel, cl::CommandQueue& q);
    void setup_buffers(const GraphCSR& graph, int start_node);
    void transfer_data_to_fpga();
    void execute_kernel_iteration(cl::Event& event);
    void transfer_data_from_fpga();
    const std::vector<int>& get_results() const; // 保持接口兼容，即使内容可能为空
    std::vector<int>& get_host_memory_for_verification();
    int get_stop_flag() const;

private:
    cl::Context& m_context;
    cl::Kernel& m_kernel;
    cl::CommandQueue& m_q;
    int m_num_vertices;
    
    // Host-side memory
    std::vector<struct_ebu_7_t, aligned_allocator<struct_ebu_7_t>> h_i_0_20;
    std::vector<struct_sbu_22_t, aligned_allocator<struct_sbu_22_t>> h_o_0_176;

    // Device-side OpenCL buffers
    cl::Buffer d_i_0_20;
    cl::Buffer d_o_0_176;

}};

#endif // __GENERATED_HOST_H__

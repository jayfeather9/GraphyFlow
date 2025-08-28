
#include "generated_host.h"
#include <iostream>

// 构造函数
AlgorithmHost::AlgorithmHost(cl::Context& context, cl::Kernel& kernel, cl::CommandQueue& q)
    : m_context(context), m_kernel(kernel), m_q(q), m_num_vertices(0) {
}

void AlgorithmHost::setup_buffers(const GraphCSR& graph, int start_node) {
    m_num_vertices = graph.num_vertices;
    cl_int err;

    // --- 模拟数据填充 ---
    // 这是需要根据具体应用修改的部分。
    // 这里我们仅为buffer分配空间并填充一些默认值。
    size_t num_elements = 1024; // 示例: 假设我们的流中有1024个元素

    // Buffer for i_0_20
    h_i_0_20.resize(num_elements);
    // TODO: 使用有意义的数据填充 h_i_0_20
    for(size_t i = 0; i < num_elements; ++i) {
        // 示例填充
        h_i_0_20[i] = {}; // 使用默认构造函数
    }
    h_i_0_20[num_elements - 1].end_flag = true; // 设置最后一个元素的结束标志

    cl_mem_flags flag = CL_MEM_READ_ONLY;
    OCL_CHECK(err, d_i_0_20 = cl::Buffer(m_context, flag, h_i_0_20.size() * sizeof(struct_ebu_7_t), NULL, &err));

    // Buffer for o_0_176
    h_o_0_176.resize(num_elements);
    // TODO: 使用有意义的数据填充 h_o_0_176
    for(size_t i = 0; i < num_elements; ++i) {
        // 示例填充
        h_o_0_176[i] = {}; // 使用默认构造函数
    }
    h_o_0_176[num_elements - 1].end_flag = true; // 设置最后一个元素的结束标志

    cl_mem_flags flag = CL_MEM_WRITE_ONLY;
    OCL_CHECK(err, d_o_0_176 = cl::Buffer(m_context, flag, h_o_0_176.size() * sizeof(struct_sbu_22_t), NULL, &err));
}

void AlgorithmHost::transfer_data_to_fpga() {
    cl_int err;
    OCL_CHECK(err, err = m_q.enqueueWriteBuffer(d_i_0_20, CL_TRUE, 0, h_i_0_20.size() * sizeof(struct_ebu_7_t), h_i_0_20.data(), nullptr, nullptr));
}

void AlgorithmHost::execute_kernel_iteration(cl::Event& event) {
    cl_int err;
    int arg_idx = 0;
    uint16_t input_length = 1024; // 示例长度
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, input_length));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_i_0_20));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_o_0_176));

    OCL_CHECK(err, err = m_q.enqueueTask(m_kernel, nullptr, &event));
}

void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    OCL_CHECK(err, err = m_q.enqueueReadBuffer(d_o_0_176, CL_TRUE, 0, h_o_0_176.size() * sizeof(struct_sbu_22_t), h_o_0_176.data(), nullptr, nullptr));
}


const std::vector<int>& AlgorithmHost::get_results() const {
    // 这个函数是为了与旧的验证流程兼容。
    // 对于流式内核，可能没有一个单一的“距离”向量。
    // 我们返回一个空的静态向量来满足接口要求。
    static const std::vector<int> empty_results;
    std::cout << "Warning: get_results() called on a streaming kernel. Returning empty vector." << std::endl;
    // TODO: 实现有意义的结果提取逻辑
    return empty_results;
}

std::vector<int>& AlgorithmHost::get_host_memory_for_verification() {
    static std::vector<int> empty_vector;
    return empty_vector;
}

int AlgorithmHost::get_stop_flag() const {
    // 流式内核通常执行一次就结束，不像迭代算法。
    // 返回1表示“停止”，这样主循环在fpga_executor中执行一次后就会退出。
    return 1;
}

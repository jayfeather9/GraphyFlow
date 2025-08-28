#include "generated_host.h"

// 构造函数
AlgorithmHost::AlgorithmHost(cl::Context& context, cl::Kernel& kernel, cl::CommandQueue& q)
    : m_context(context), m_kernel(kernel), m_q(q), m_num_vertices(0) {
    h_stop_flag.resize(1);
}

// 创建和初始化 Buffers
void AlgorithmHost::setup_buffers(const GraphCSR& graph, int start_node) {
    m_num_vertices = graph.num_vertices;
    cl_int err;

    // 1. 初始化 Host 端内存
    h_distances.assign(graph.num_vertices, INFINITY_DIST);
    if (start_node < graph.num_vertices) {
        h_distances[start_node] = 0;
    }

    // 2. 分配 Device 端 OpenCL Buffers
    // 这些 Buffers 是 Bellman-Ford 算法所特有的
    OCL_CHECK(err, d_offsets = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, graph.offsets.size() * sizeof(int), (void*)graph.offsets.data(), &err));
    OCL_CHECK(err, d_columns = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, graph.columns.size() * sizeof(int), (void*)graph.columns.data(), &err));
    OCL_CHECK(err, d_weights = cl::Buffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, graph.weights.size() * sizeof(int), (void*)graph.weights.data(), &err));
    OCL_CHECK(err, d_distances = cl::Buffer(m_context, CL_MEM_READ_WRITE, h_distances.size() * sizeof(int), NULL, &err));
    OCL_CHECK(err, d_stop_flag = cl::Buffer(m_context, CL_MEM_READ_WRITE, h_stop_flag.size() * sizeof(int), NULL, &err));
}

// 将数据从 Host 写入 Device
void AlgorithmHost::transfer_data_to_fpga() {
    cl_int err;
    h_stop_flag[0] = 1; // 在每轮迭代开始前，假定算法将收敛

    // 写入本轮迭代需要更新的数据
    OCL_CHECK(err, err = m_q.enqueueWriteBuffer(d_distances, CL_TRUE, 0, h_distances.size() * sizeof(int), h_distances.data(), nullptr, nullptr));
    OCL_CHECK(err, err = m_q.enqueueWriteBuffer(d_stop_flag, CL_TRUE, 0, h_stop_flag.size() * sizeof(int), h_stop_flag.data(), nullptr, nullptr));
}

// 执行 Kernel
void AlgorithmHost::execute_kernel_iteration(cl::Event& event) {
    cl_int err;
    int arg_idx = 0;

    // 设置 Kernel 参数，这部分是算法强相关的
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, m_num_vertices));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_offsets));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_columns));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_weights));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_distances));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_stop_flag));

    // 启动 Kernel
    OCL_CHECK(err, err = m_q.enqueueTask(m_kernel, nullptr, &event));
}

// 从 Device 读回数据
void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    // 读回终止标志，判断是否需要继续迭代
    OCL_CHECK(err, err = m_q.enqueueReadBuffer(d_stop_flag, CL_TRUE, 0, h_stop_flag.size() * sizeof(int), h_stop_flag.data(), nullptr, nullptr));

    // 如果需要，也可以在这里读回每一轮的距离结果用于调试
    // OCL_CHECK(err, err = m_q.enqueueReadBuffer(d_distances, CL_TRUE, 0, h_distances.size() * sizeof(int), h_distances.data(), nullptr, nullptr));
}

// 获取最终结果
const std::vector<int>& AlgorithmHost::get_results() const {
    cl_int err;
    // 确保在返回前，Host 端的数据是最新的
    OCL_CHECK(err, err = m_q.enqueueReadBuffer(d_distances, CL_TRUE, 0, h_distances.size() * sizeof(int), (void*)h_distances.data(), nullptr, nullptr));
    return h_distances;
}

// 获取 Host 内存引用，用于验证
std::vector<int>& AlgorithmHost::get_host_memory_for_verification() {
    return h_distances;
}

// 获取终止标志
int AlgorithmHost::get_stop_flag() const {
    return h_stop_flag[0];
}
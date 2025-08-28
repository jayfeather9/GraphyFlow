#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__

#include "common.h"
#include <vector>

// 这个类封装了所有由 Python 前端生成的、与特定算法相关的 Host 端逻辑。
// 通用的 Host 执行框架 (fpga_executor) 将会使用这个类。
class AlgorithmHost {
public:
    // 构造函数，接收 OpenCL 上下文和 Kernel 对象
    AlgorithmHost(cl::Context& context, cl::Kernel& kernel, cl::CommandQueue& q);

    // 为算法创建所有必要的 Host 端内存和 OpenCL Buffers
    void setup_buffers(const GraphCSR& graph, int start_node);

    // 将 Host 端数据迁移到设备端 (FPGA)
    void transfer_data_to_fpga();

    // 执行一轮 Kernel 计算
    void execute_kernel_iteration(cl::Event& event);

    // 从设备端读取计算结果
    void transfer_data_from_fpga();

    // 获取最终的计算结果
    const std::vector<int>& get_results() const;

    // 获取用于 Host 端验证的 Host 内存指针
    std::vector<int>& get_host_memory_for_verification();

    // 获取迭代终止标志
    int get_stop_flag() const;

private:
    // OpenCL 对象引用
    cl::Context& m_context;
    cl::Kernel& m_kernel;
    cl::CommandQueue& m_q;

    // Host 端内存
    // 对于 Bellman-Ford，我们需要一个 vector 来存储距离
    std::vector<int> h_distances;
    // 终止标志
    std::vector<int> h_stop_flag;

    // Device 端 (FPGA) OpenCL Buffers
    // 这些是 Bellman-Ford 特有的
    cl::Buffer d_offsets;
    cl::Buffer d_columns;
    cl::Buffer d_weights;
    cl::Buffer d_distances;
    cl::Buffer d_stop_flag;

    // Kernel 参数
    int m_num_vertices;
};

#endif // __GENERATED_HOST_H__
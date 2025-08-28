#include <limits>

const int INFINITY_DIST = std::numeric_limits<int>::max();

// Vitis HLS 要求内核函数必须包含在一个 extern "C" 块中
extern "C" {

void bellman_ford_kernel(
    int num_vertices,
    const int* offsets,
    const int* columns,
    const int* weights,
    int* distances,
    int* stop_flag
) {
// HLS Pragmas: 告诉 Vitis HLS 如何创建硬件接口
// 将所有指针参数映射到独立的 AXI Master 内存端口
#pragma HLS INTERFACE m_axi port=offsets  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=columns  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=weights  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=distances offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=stop_flag offset=slave bundle=gmem4

// 将标量参数映射到 AXI Lite 控制总线
#pragma HLS INTERFACE s_axilite port=num_vertices
#pragma HLS INTERFACE s_axilite port=return

    // 内核逻辑：实现 Bellman-Ford 的一轮迭代
    // 这个逻辑和 host_bellman_ford.cpp 中的几乎一样
    
    // Host 端在调用前将 stop_flag[0] 设置为 1 (表示停止)
    // 如果内核中发生了任何距离更新，我们需要将其设置为 0 (表示不停止)
    bool changed_in_kernel = false;

    // 遍历所有顶点
    relax_edges: for (int u = 0; u < num_vertices; ++u) {
        #pragma HLS LOOP_TRIPCOUNT min=1000 max=4000
        if (distances[u] != INFINITY_DIST) {
            int start_offset = offsets[u];
            int end_offset = offsets[u + 1];
            // 遍历该顶点的所有出边
            for (int i = start_offset; i < end_offset; ++i) {
                #pragma HLS PIPELINE
                int v = columns[i];
                int weight = weights[i];

                // 松弛操作
                if (distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    changed_in_kernel = true;
                }
            }
        }
    }

    if (changed_in_kernel) {
        stop_flag[0] = 0; // 距离已更新，通知 Host 继续迭代
    }
}

} // extern "C"
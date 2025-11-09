#include "host_verifier.h"
#include "host_bellman_ford.h"
#include <iostream>

/*
std::vector<int> verify_on_host(const GraphCSR &graph, int start_node) {
    std::vector<int> distances(graph.num_vertices, INFINITY_DIST);
    distances[start_node] = 0;

    int max_iterations = graph.num_vertices;
    int iter = 0;
    bool changed = true;

    std::cout << "\nStarting Host verification..." << std::endl;

    while (changed && iter < max_iterations) {
        // Run one iteration of the algorithm
        changed = host_bellman_ford_iteration(graph, distances);
        iter++;
    }

    std::cout << "Host computation converged after " << iter << " iterations."
              << std::endl;

    // Check for negative weight cycles (optional but good practice)
    if (iter == max_iterations &&
        host_bellman_ford_iteration(graph, distances)) {
        std::cout << "Warning: Negative weight cycle detected by host verifier."
                  << std::endl;
    }

    return distances;
}
*/

/*
std::vector<unsigned int> verify_on_host(const GraphCSR &graph, int start_node) {
    std::vector<unsigned int> distances(graph.num_vertices, 0);
    
    for (int u = 0; u < 32; u++)
    {
        int select_index = u;//((double)std::rand())/((RAND_MAX + 1u)/csr->vertexNum);
        distances[select_index] = 1 << u;
    }


    int max_iterations = graph.num_vertices;
    int iter = 0;
    bool changed = true;

    std::cout << "\nStarting Host verification..." << std::endl;

    while (changed && iter < max_iterations) {
        // Run one iteration of the algorithm
        changed = host_cc_iteration(graph, distances);
        iter++;
    }

    std::cout << "Host computation converged after " << iter << " iterations."
              << std::endl;

    // Check for negative weight cycles (optional but good practice)
    if (iter == max_iterations &&
        host_cc_iteration(graph, distances)) {
        std::cout << "Warning: Negative weight cycle detected by host verifier."
                  << std::endl;
    }

    return distances;
}
*/

std::vector<unsigned int> verify_on_host(const GraphCSR &graph, int start_node) {
    
    // --- 核心修改：创建两个向量 ---
    // distances_k     : 存储第 k 次迭代的状态 (用于读取)
    // distances_kplus1: 存储第 k+1 次迭代的状态 (用于写入)
    std::vector<unsigned int> distances_k(graph.num_vertices, 0);
    std::vector<unsigned int> distances_kplus1(graph.num_vertices, 0);
    
    // 初始化 k 状态 (distances_k)
    for (int u = 0; u < 32; u++)
    {
        int select_index = u;
        distances_k[select_index] = 1 << u;
    }

    int max_iterations = graph.num_vertices;
    int iter = 0;
    bool changed = true;

    std::cout << "\nStarting Host verification..." << std::endl;

    while (changed && iter < max_iterations) {
        
        // --- 核心修改：调用新的 Jacobi 迭代函数 ---
        // 它会从 distances_k 读取，并写入 distances_kplus1
        changed = host_cc_iteration(graph, distances_k, distances_kplus1);
        
        // --- 核心修改：为下一次迭代做准备 ---
        // 将 k+1 状态 "提交" 为新的 k 状态
        distances_k = distances_kplus1;
        
        iter++;
    }

    std::cout << "Host computation converged after " << iter << " iterations."
              << std::endl;

    // 检查负环 (现在也使用 Jacobi 方式)
    if (iter == max_iterations &&
        host_cc_iteration(graph, distances_k, distances_kplus1)) {
        std::cout << "Warning: Negative weight cycle detected by host verifier."
                  << std::endl;
    }

    // 返回最终收敛的状态 (存储在 distances_k 中)
    return distances_k;
}

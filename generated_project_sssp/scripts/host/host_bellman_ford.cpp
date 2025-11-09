#include "host_bellman_ford.h"

bool host_bellman_ford_iteration(const GraphCSR &graph,
                                 std::vector<int> &distances) {
    bool changed = false;

    // --- USER MODIFIABLE SECTION: Host Computation Logic ---
    // For each vertex, relax all outgoing edges
    for (int u = 0; u < graph.num_vertices; ++u) {
        if (distances[u] != INFINITY_DIST) {
            for (int i = graph.offsets[u]; i < graph.offsets[u + 1]; ++i) {
                int v = graph.columns[i];
                if ((v & 0x40000000) != 0) {
                    continue; // Skip dummy edges
                }
                int weight = graph.weights[i];

                // Relaxation step
                if (distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    changed = true;
                }
            }
        }
    }
    // --- END USER MODIFIABLE SECTION ---

    return changed;
}

bool host_cc_iteration(const GraphCSR &graph,
                           std::vector<unsigned int> &bitmasks) {
    

    bool changed = false;

    for (int u = 0; u < graph.num_vertices; ++u) {
        
        if (bitmasks[u] != 0) {
            
            for (int i = graph.offsets[u]; i < graph.offsets[u + 1]; ++i) {
                int v = graph.columns[i];

                if ((v & 0x40000000) != 0) {
                    continue; 
                }

                // (bitmasks[u] & ~bitmasks[v]) != 0 
                // 等价于 (bitmasks[v] | bitmasks[u]) != bitmasks[v]
                if ((bitmasks[u] & ~bitmasks[v]) != 0) {
                    
                    // 将 'u' 的掩码位合并到 'v' 的掩码中
                    bitmasks[v] = bitmasks[v] | bitmasks[u];
                    
                    // 标记发生了变化
                    changed = true;
                }
            }
        }
    }


    return changed;
}
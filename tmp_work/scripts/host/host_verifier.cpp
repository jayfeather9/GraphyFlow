#include "host_verifier.h"
#include "host_bellman_ford.h"
#include <iostream>

std::vector<unsigned int> verify_on_host(const GraphCSR &graph, int start_node) {
    // Classic Bellman-Ford style verification to match kernel min-plus logic
    std::vector<int> distances(graph.num_vertices, INFINITY_DIST);
    if (start_node >= 0 && start_node < graph.num_vertices) {
        distances[start_node] = 0;
    }

    // Compute indegree (used to emulate cnt accumulation in the 64-bit
    // {min_dist, cnt} experiment).
    std::vector<uint32_t> indegree(graph.num_vertices, 0);
    for (int u = 0; u < graph.num_vertices; ++u) {
        for (int i = graph.offsets[u]; i < graph.offsets[u + 1]; ++i) {
            int v = graph.columns[i];
            if ((v & 0x40000000) != 0) {
                continue;
            }
            if (v >= 0 && v < graph.num_vertices) {
                indegree[v]++;
            }
        }
    }

    int max_iterations = graph.num_vertices;
    int iter = 0;
    bool changed = true;

    std::cout << "\nStarting Host verification..." << std::endl;

    while (changed && iter < max_iterations) {
        changed = host_bellman_ford_iteration(graph, distances);

        // Apply-kernel experiment rule: if cnt % 4 == 0 and cnt != 0, force
        // distance to 0 so the host can observe 64-bit transfers.
        for (int v = 0; v < graph.num_vertices; ++v) {
            uint32_t cnt = indegree[v];
            if (cnt != 0 && (cnt & 3) == 0) {
                if (distances[v] != 0) {
                    distances[v] = 0;
                    changed = true;
                }
            }
        }

        iter++;
    }

    std::cout << "Host computation converged after " << iter << " iterations."
              << std::endl;

    if (iter == max_iterations &&
        host_bellman_ford_iteration(graph, distances)) {
        std::cout << "Warning: Negative weight cycle detected by host verifier."
                  << std::endl;
    }

    // Cast to unsigned for the return type expected by the host harness
    std::vector<unsigned int> result(distances.begin(), distances.end());
    return result;
}

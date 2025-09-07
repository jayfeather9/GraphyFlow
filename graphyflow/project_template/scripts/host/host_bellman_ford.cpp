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
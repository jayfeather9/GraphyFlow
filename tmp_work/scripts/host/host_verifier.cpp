#include "host_verifier.h"
#include "host_bellman_ford.h"
#include <iostream>

std::vector<unsigned int> verify_on_host(const GraphCSR &graph, int start_node) {
    // Classic Bellman-Ford style verification to match kernel min-plus logic
    std::vector<int> distances(graph.num_vertices, INFINITY_DIST);
    if (start_node >= 0 && start_node < graph.num_vertices) {
        distances[start_node] = 0;
    }

    int max_iterations = graph.num_vertices;
    int iter = 0;
    bool changed = true;

    std::cout << "\nStarting Host verification..." << std::endl;

    while (changed && iter < max_iterations) {
        changed = host_bellman_ford_iteration(graph, distances);
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

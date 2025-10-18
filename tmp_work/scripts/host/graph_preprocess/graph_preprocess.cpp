#include "graph_preprocess.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

// A local helper struct to temporarily hold edge information with global vertex
// IDs.
struct Edge {
    int src, dest, weight;
};

/**
 * @brief Partitions a global graph and preprocesses each partition into a local
 * CSR format.
 *
 * This function implements a partitioning strategy based on destination
 * vertices.
 * 1.  It identifies all unique destination vertices in the graph.
 * 2.  It distributes these destination vertices disjointly and as evenly as
 * possible among all available partitions (for both big and little kernels).
 * 3.  It assigns each edge from the global graph to the partition that is
 * responsible for its destination vertex.
 * 4.  For each partition, it collects all unique vertices involved (both
 * sources and destinations).
 * 5.  It performs vertex ID compression for each partition, creating a local ID
 * space. Destination vertices are mapped first to ensure they occupy the lower
 * ID range.
 * 6.  It rewrites the partition's edges using these new local IDs.
 * 7.  Finally, it converts the rewritten edges into a local CSR format.
 *
 * @param graph The input global graph in CSR format.
 * @return A PartitionContainer object containing all processed partitions.
 */
PartitionContainer partitionGraph(const GraphCSR *graph) {
    std::cout << "--- Starting Graph Partitioning and Preprocessing ---"
              << std::endl;
    PartitionContainer container;
    container.num_graph_vertices = graph->num_vertices;
    container.num_graph_edges = graph->num_edges;
    printf("Global graph has %d vertices and %d edges.\n", graph->num_vertices,
           graph->num_edges);
    const int num_partitions = BIG_KERNEL_NUM + LITTLE_KERNEL_NUM;

    std::cout<<"DEBUG big num:"<<BIG_KERNEL_NUM<<" little num :"<<LITTLE_KERNEL_NUM;

    if (num_partitions == 0) {
        std::cerr << "Error: No kernels defined (BIG_KERNEL_NUM and "
                     "LITTLE_KERNEL_NUM are both 0)."
                  << std::endl;
        return container;
    }
    std::cout << "[INFO] Total partitions to create: " << num_partitions
              << std::endl;

    // --- PHASE 1: Identify and Collect All Unique Destination Vertices ---
    std::set<int> unique_dst_vertices_set;
    for (int i = 0; i < graph->num_edges; ++i) {
        unique_dst_vertices_set.insert(graph->columns[i]);
    }
    std::vector<int> unique_dst_vertices(unique_dst_vertices_set.begin(),
                                         unique_dst_vertices_set.end());
    std::cout << "[PHASE 1] Found " << unique_dst_vertices.size()
              << " unique destination vertices." << std::endl;

    // --- PHASE 2: Distribute Destination Vertices to Partitions ---
    std::vector<std::set<int>> dst_vertices_per_partition(num_partitions);
    std::unordered_map<int, int> dst_vertex_to_partition_map;

    size_t base_dst_per_part = unique_dst_vertices.size() / num_partitions;
    size_t remainder_dst = unique_dst_vertices.size() % num_partitions;
    size_t current_dst_idx = 0;

    for (int i = 0; i < num_partitions; ++i) {
        size_t num_dst_in_part =
            base_dst_per_part + (i < remainder_dst ? 1 : 0);
        for (size_t j = 0; j < num_dst_in_part; ++j) {
            if (current_dst_idx < unique_dst_vertices.size()) {
                int vertex_id = unique_dst_vertices[current_dst_idx];
                dst_vertices_per_partition[i].insert(vertex_id);
                dst_vertex_to_partition_map[vertex_id] = i;
                current_dst_idx++;
            }
        }
        std::cout << "[PHASE 2] Partition " << i << " assigned "
                  << dst_vertices_per_partition[i].size()
                  << " destination vertices." << std::endl;
    }

    // --- PHASE 3: Assign Edges to Partitions Based on Destination Vertex ---
    std::vector<std::vector<Edge>> edges_per_partition(num_partitions);
    for (int u = 0; u < graph->num_vertices; ++u) {
        for (int i = graph->offsets[u]; i < graph->offsets[u + 1]; ++i) {
            int v = graph->columns[i];
            int w = graph->weights[i];

            // Find which partition this edge belongs to
            auto it = dst_vertex_to_partition_map.find(v);
            if (it != dst_vertex_to_partition_map.end()) {
                int partition_id = it->second;
                edges_per_partition[partition_id].push_back({u, v, w});
            } else {
                // This case should not happen if all dst vertices are mapped.
                // It might occur for sink nodes with no incoming edges, which
                // is fine.
            }
        }
    }
    std::cout << "[PHASE 3] All edges have been assigned to their respective "
                 "partitions."
              << std::endl;

    // --- PHASE 4: Process Each Partition (Compress IDs and Convert to CSR) ---
    std::cout << "[PHASE 4] Processing each partition..." << std::endl;
    for (int i = 0; i < num_partitions; ++i) {
        PartitionDescriptor pd;
        GraphCSR &p_graph = pd.partitioned_graph;
        const auto &partition_edges = edges_per_partition[i];
        const auto &partition_dst_nodes = dst_vertices_per_partition[i];

        if (partition_edges.empty()) {
            std::cout << "  - Partition " << i << " has no edges. Skipping."
                      << std::endl;
            // Still create a valid (but empty) partition descriptor
            pd.num_edges = 0;
            pd.num_vertices = 0;
            p_graph.num_edges = 0;
            p_graph.num_vertices = 0;
            p_graph.offsets.push_back(0);

        } else {
            // --- 4.1: Collect unique vertices and build ID mappings ---
            std::set<int> local_vertices_set;
            for (const auto &edge : partition_edges) {
                local_vertices_set.insert(edge.src);
                local_vertices_set.insert(edge.dest);
            }

            int local_id_counter = 0;
            // First, map destination vertices to guarantee they have
            // lower-range IDs
            for (int global_id : partition_dst_nodes) {
                p_graph.vtx_map[global_id] = local_id_counter;
                p_graph.vtx_map_rev[local_id_counter] = global_id;
                local_id_counter++;
            }
            p_graph.num_dsts = partition_dst_nodes.size();
            // Then, map the remaining source vertices
            for (int global_id : local_vertices_set) {
                if (p_graph.vtx_map.find(global_id) == p_graph.vtx_map.end()) {
                    p_graph.vtx_map[global_id] = local_id_counter;
                    p_graph.vtx_map_rev[local_id_counter] = global_id;
                    local_id_counter++;
                }
            }
            p_graph.num_vertices = local_vertices_set.size();
            p_graph.num_edges = partition_edges.size();

            // --- 4.2: Rewrite edges with local, compressed IDs ---
            std::vector<Edge> local_edges;
            local_edges.reserve(p_graph.num_edges);
            for (const auto &global_edge : partition_edges) {
                local_edges.push_back({p_graph.vtx_map[global_edge.src],
                                       p_graph.vtx_map[global_edge.dest],
                                       global_edge.weight});
            }

            // --- 4.3: Convert local edges to CSR format ---
            std::sort(
                local_edges.begin(), local_edges.end(),
                [](const Edge &a, const Edge &b) { return a.src < b.src; });

            p_graph.offsets.resize(p_graph.num_vertices + 1, 0);
            p_graph.columns.resize(p_graph.num_edges);
            p_graph.weights.resize(p_graph.num_edges);

            std::vector<int> out_degree(p_graph.num_vertices, 0);
            for (int j = 0; j < p_graph.num_edges; ++j) {
                p_graph.columns[j] = local_edges[j].dest;
                p_graph.weights[j] = local_edges[j].weight;
                out_degree[local_edges[j].src]++;
            }

            p_graph.offsets[0] = 0;
            for (int j = 0; j < p_graph.num_vertices; ++j) {
                p_graph.offsets[j + 1] = p_graph.offsets[j] + out_degree[j];
            }

            // --- 4.4: Finalize partition descriptor metadata ---
            pd.num_edges = p_graph.num_edges;
            pd.num_vertices = p_graph.num_vertices;
        }

        pd.kernel_id = i;
        if (i < LITTLE_KERNEL_NUM) {
            pd.is_dense =
                true; // This is a Dense Partition (DP) for a little kernel
            container.DPs.push_back(pd);
        } else {
            pd.is_dense =
                false; // This is a Sparse Partition (SP) for a big kernel
            container.SPs.push_back(pd);
        }
        std::cout << "  - Processed Partition " << i << ": " << pd.num_vertices
                  << " local vertices, " << pd.num_edges << " edges."
                  << std::endl;
    }

    container.num_dense_partitions = container.DPs.size();
    container.num_sparse_partitions = container.SPs.size();

    std::cout<<"DP: "<<container.num_dense_partitions<<"SP: "<<container.num_sparse_partitions<<std::endl;
    std::cout << "[SUCCESS] Graph partitioning and preprocessing complete."
              << std::endl;
    return container;
}

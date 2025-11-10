#include "graph_preprocess.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
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
    container.num_dense_groups = NUM_LITTLE_MERGERS;
    container.num_sparse_groups = NUM_BIG_MERGERS;
    container.num_dense_partitions = 0;
    container.num_sparse_partitions = 0;

    container.dense_groups.resize(container.num_dense_groups);
    container.sparse_groups.resize(container.num_sparse_groups);
    container.dense_partition_indices.resize(container.num_dense_groups);
    container.sparse_partition_indices.resize(container.num_sparse_groups);

    constexpr size_t little_pipeline_len =
        sizeof(LITTLE_MERGER_PIPELINE_LENGTHS) / sizeof(uint32_t);
    constexpr size_t little_offset_len =
        sizeof(LITTLE_MERGER_KERNEL_OFFSETS) / sizeof(uint32_t);
    constexpr size_t big_pipeline_len =
        sizeof(BIG_MERGER_PIPELINE_LENGTHS) / sizeof(uint32_t);
    constexpr size_t big_offset_len =
        sizeof(BIG_MERGER_KERNEL_OFFSETS) / sizeof(uint32_t);

    for (size_t g = 0; g < container.num_dense_groups; ++g) {
        auto &group = container.dense_groups[g];
        group.group_id = static_cast<unsigned int>(g);
        group.pipeline_offset =
            (g < little_offset_len) ? LITTLE_MERGER_KERNEL_OFFSETS[g] : 0;
        group.num_pipelines =
            (g < little_pipeline_len) ? LITTLE_MERGER_PIPELINE_LENGTHS[g] : 0;
    }

    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        auto &group = container.sparse_groups[g];
        group.group_id = static_cast<unsigned int>(g);
        group.pipeline_offset =
            (g < big_offset_len) ? BIG_MERGER_KERNEL_OFFSETS[g] : 0;
        group.num_pipelines =
            (g < big_pipeline_len) ? BIG_MERGER_PIPELINE_LENGTHS[g] : 0;
    }

    printf("Global graph has %d vertices and %d edges.\n", graph->num_vertices,
           graph->num_edges);
    std::cout << "[INFO] Dense groups: " << container.num_dense_groups
              << ", Sparse groups: " << container.num_sparse_groups
              << std::endl;

    // --- PHASE 1: Identify and collect unique destination vertices ---
    std::set<int> unique_dst_vertices_set;
    std::unordered_map<int, int> node_indegrees;
    for (int i = 0; i < graph->num_edges; ++i) {
        int dst = graph->columns[i];
        unique_dst_vertices_set.insert(dst);
        node_indegrees[dst]++;
    }
    std::vector<int> unique_dst_vertices(unique_dst_vertices_set.begin(),
                                         unique_dst_vertices_set.end());

    std::sort(unique_dst_vertices.begin(), unique_dst_vertices.end(),
              [&node_indegrees](int a, int b) {
                  return node_indegrees[a] > node_indegrees[b];
              });

    std::cout << "[PHASE 1] Found " << unique_dst_vertices.size()
              << " unique destination vertices (sorted by indegree)."
              << std::endl;

    // --- PHASE 2: distribute destination vertices among groups ---
    struct DestinationAssignment {
        bool is_dense;
        size_t group_idx;
    };

    std::unordered_map<int, DestinationAssignment> dst_assignment;
    std::vector<std::set<int>> little_dst_sets(container.num_dense_groups);
    std::vector<std::set<int>> big_dst_sets(container.num_sparse_groups);

    auto compute_distribution = [](size_t num_groups,
                                   const std::vector<uint32_t> &pipeline_counts,
                                   size_t available_vertices) {
        std::vector<size_t> counts(num_groups, 0);
        if (num_groups == 0 || available_vertices == 0) {
            return counts;
        }

        double total_weight = 0.0;
        for (size_t g = 0; g < num_groups; ++g) {
            total_weight += std::max<uint32_t>(1, pipeline_counts[g]);
        }

        size_t remaining_vertices = available_vertices;
        double remaining_weight = total_weight;
        for (size_t g = 0; g < num_groups; ++g) {
            if (remaining_vertices == 0) {
                counts[g] = 0;
                continue;
            }

            double weight = std::max<uint32_t>(1, pipeline_counts[g]);
            double fraction = (remaining_weight <= 0.0)
                                  ? (1.0 / std::max<size_t>(1, num_groups - g))
                                  : (weight / remaining_weight);
            size_t assign =
                static_cast<size_t>(fraction * static_cast<double>(remaining_vertices));
            if (assign == 0 && remaining_vertices > 0 && weight > 0.0) {
                assign = 1;
            }
            if (assign > remaining_vertices) {
                assign = remaining_vertices;
            }
            if (g == num_groups - 1) {
                assign = remaining_vertices;
            }

            counts[g] = assign;
            remaining_vertices -= assign;
            remaining_weight -= weight;
        }

        return counts;
    };

    std::vector<uint32_t> little_pipeline_counts(container.num_dense_groups, 0);
    for (size_t g = 0; g < container.num_dense_groups; ++g) {
        little_pipeline_counts[g] = container.dense_groups[g].num_pipelines;
    }

    std::vector<uint32_t> big_pipeline_counts(container.num_sparse_groups, 0);
    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        big_pipeline_counts[g] = container.sparse_groups[g].num_pipelines;
    }

    size_t total_vertices = unique_dst_vertices.size();
    size_t vertex_cursor = 0;

    auto little_counts =
        compute_distribution(container.num_dense_groups, little_pipeline_counts,
                             total_vertices);

    for (size_t g = 0; g < container.num_dense_groups; ++g) {
        size_t assign = little_counts[g];
        for (size_t i = 0; i < assign && vertex_cursor < total_vertices; ++i) {
            int vertex_id = unique_dst_vertices[vertex_cursor++];
            little_dst_sets[g].insert(vertex_id);
            dst_assignment[vertex_id] = {true, g};
        }
    }

    size_t remaining_vertices = (vertex_cursor <= total_vertices)
                                    ? (total_vertices - vertex_cursor)
                                    : 0;

    auto big_counts = compute_distribution(container.num_sparse_groups,
                                           big_pipeline_counts,
                                           remaining_vertices);

    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        size_t assign = big_counts[g];
        for (size_t i = 0; i < assign && vertex_cursor < total_vertices; ++i) {
            int vertex_id = unique_dst_vertices[vertex_cursor++];
            big_dst_sets[g].insert(vertex_id);
            dst_assignment[vertex_id] = {false, g};
        }
    }

    std::cout << "[PHASE 2] Dense groups assigned "
              << std::accumulate(little_counts.begin(), little_counts.end(),
                                 static_cast<size_t>(0))
              << " dst vertices." << std::endl;
    std::cout << "[PHASE 2] Sparse groups assigned "
              << std::accumulate(big_counts.begin(), big_counts.end(),
                                 static_cast<size_t>(0))
              << " dst vertices." << std::endl;

    // --- PHASE 3: assign edges based on destination ownership ---
    std::vector<std::vector<Edge>> edges_lists(
        container.num_dense_groups + container.num_sparse_groups);
    size_t little_edge_num = 0, big_edge_num = 0;

    for (int u = 0; u < graph->num_vertices; ++u) {
        for (int i = graph->offsets[u]; i < graph->offsets[u + 1]; ++i) {
            int v = graph->columns[i];
            int w = graph->weights[i];

            auto it = dst_assignment.find(v);
            if (it == dst_assignment.end()) {
                std::cerr << "[ERROR] Destination vertex " << v
                          << " not found in any partition!" << std::endl;
                exit(1);
            }

            const auto &assignment = it->second;
            size_t list_idx = assignment.is_dense
                                  ? assignment.group_idx
                                  : (container.num_dense_groups + assignment.group_idx);
            edges_lists[list_idx].push_back({u, v, w});
            if (assignment.is_dense) {
                little_edge_num++;
            } else {
                big_edge_num++;
            }
        }
    }

    std::cout << "[PHASE 3] Assigned " << little_edge_num
              << " edges to dense groups." << std::endl;
    std::cout << "[PHASE 3] Assigned " << big_edge_num
              << " edges to sparse groups." << std::endl;

    // --- PHASE 4: process each group partition ---
    std::cout << "[PHASE 4] Processing partitions..." << std::endl;

    auto process_partition = [&](const std::vector<Edge> &partition_edges,
                                 const std::set<int> &partition_dst_nodes,
                                 bool is_dense,
                                 unsigned int num_pipelines) {
        PartitionDescriptor pd;
        pd.is_dense = is_dense;
        pd.num_pipelines = num_pipelines;

        if (partition_edges.empty()) {
            pd.num_edges = 0;
            pd.num_vertices = 0;
            pd.num_dsts = partition_dst_nodes.size();
            pd.pipeline_edges.resize(num_pipelines);
            for (unsigned int pip = 0; pip < num_pipelines; ++pip) {
                pd.pipeline_edges[pip].pipeline_id = pip;
                pd.pipeline_edges[pip].num_edges = 0;
                pd.pipeline_edges[pip].offsets.assign(1, 0);
                pd.pipeline_edges[pip].columns.clear();
                pd.pipeline_edges[pip].weights.clear();
            }
            return pd;
        }

        std::set<int> local_vertices_set;
        for (const auto &edge : partition_edges) {
            local_vertices_set.insert(edge.src);
            local_vertices_set.insert(edge.dest);
        }

        std::vector<int> ordered_dst_vertices(partition_dst_nodes.begin(),
                                              partition_dst_nodes.end());
        std::srand(42);
        std::random_shuffle(ordered_dst_vertices.begin(),
                            ordered_dst_vertices.end());

        int local_id_counter = 0;
        for (int global_id : ordered_dst_vertices) {
            pd.vtx_map[global_id] = local_id_counter;
            pd.vtx_map_rev[local_id_counter] = global_id;
            local_id_counter++;
        }
        pd.num_dsts = partition_dst_nodes.size();

        for (int global_id : local_vertices_set) {
            if (pd.vtx_map.find(global_id) == pd.vtx_map.end()) {
                pd.vtx_map[global_id] = local_id_counter;
                pd.vtx_map_rev[local_id_counter] = global_id;
                local_id_counter++;
            }
        }
        pd.num_vertices = local_vertices_set.size();

        std::vector<Edge> local_edges;
        local_edges.reserve(partition_edges.size());
        for (const auto &global_edge : partition_edges) {
            uint32_t src_id = pd.vtx_map[global_edge.src];
            uint32_t dest_id = pd.vtx_map[global_edge.dest];
            uint32_t weight = global_edge.weight;
            local_edges.push_back({static_cast<int>(src_id),
                                  static_cast<int>(dest_id),
                                  static_cast<int>(weight)});
        }

        std::sort(local_edges.begin(), local_edges.end(),
                  [](const Edge &a, const Edge &b) { return a.src < b.src; });

        pd.num_edges = local_edges.size();
        pd.pipeline_edges.resize(num_pipelines);
        unsigned int edges_per_pipeline =
            (num_pipelines == 0)
                ? 0
                : static_cast<unsigned int>((pd.num_edges + num_pipelines - 1) /
                                             num_pipelines);

        for (unsigned int pip = 0; pip < num_pipelines; ++pip) {
            std::vector<Edge> cur_pip_edges;
            pd.pipeline_edges[pip].pipeline_id = pip;
            int start_idx =
                std::min(static_cast<int>(pip * edges_per_pipeline),
                         static_cast<int>(pd.num_edges));
            int end_idx =
                std::min(start_idx + static_cast<int>(edges_per_pipeline),
                         static_cast<int>(pd.num_edges));

            uint32_t last_src_buffer = 0;
            uint32_t last_src_id = 0;
            for (int edge_idx = start_idx; edge_idx < end_idx; ++edge_idx) {
                auto local_edge = local_edges[edge_idx];
                uint32_t src_id = local_edge.src;
                uint32_t dest_id = local_edge.dest;
                uint32_t weight = local_edge.weight;
                uint32_t cur_src_buffer =
                    static_cast<uint32_t>(std::floor(static_cast<double>(src_id) /
                                                      SRC_BUFFER_SIZE));
                if (is_dense && cur_src_buffer != last_src_buffer) {
                    uint32_t mod8 = cur_pip_edges.size() % 8;
                    if (mod8 != 0) {
                        for (uint32_t pad = 0; pad < (8 - mod8); ++pad) {
                            cur_pip_edges.push_back(
                                {static_cast<int>(last_src_id), 0x7FFFFFFF, 1});
                        }
                    }
                    last_src_buffer = cur_src_buffer;
                }
                last_src_id = src_id;
                cur_pip_edges.push_back({static_cast<int>(src_id),
                                         static_cast<int>(dest_id),
                                         static_cast<int>(weight)});
            }

            pd.pipeline_edges[pip].num_edges = cur_pip_edges.size();

            int padding_size =
                (8 - (pd.pipeline_edges[pip].num_edges % 8)) % 8;
            pd.pipeline_edges[pip].num_edges += padding_size;

            pd.pipeline_edges[pip].offsets.resize(pd.num_vertices + 1, 0);
            pd.pipeline_edges[pip].columns.reserve(
                pd.pipeline_edges[pip].num_edges);
            pd.pipeline_edges[pip].weights.reserve(
                pd.pipeline_edges[pip].num_edges);

            std::vector<int> out_degree(pd.num_vertices, 0);
            for (size_t j = 0; j < cur_pip_edges.size(); ++j) {
                out_degree[cur_pip_edges[j].src]++;
                if (j == cur_pip_edges.size() - 1) {
                    for (int p = 0; p < padding_size; ++p) {
                        out_degree[cur_pip_edges[j].src]++;
                    }
                }
            }

            for (int v = 0; v < pd.num_vertices; ++v) {
                pd.pipeline_edges[pip].offsets[v + 1] =
                    pd.pipeline_edges[pip].offsets[v] + out_degree[v];
            }

            std::vector<int> current_offset = pd.pipeline_edges[pip].offsets;
            for (size_t j = 0; j < cur_pip_edges.size(); ++j) {
                int src = cur_pip_edges[j].src;
                int idx = current_offset[src]++;
                pd.pipeline_edges[pip].columns.push_back(cur_pip_edges[j].dest);
                pd.pipeline_edges[pip].weights.push_back(cur_pip_edges[j].weight);
                if (j == cur_pip_edges.size() - 1) {
                    for (int p = 0; p < padding_size; ++p) {
                        idx = current_offset[src]++;
                        pd.pipeline_edges[pip].columns.push_back(0x7FFFFFFF);
                        pd.pipeline_edges[pip].weights.push_back(1);
                    }
                }
            }
        }

        return pd;
    };

    for (size_t g = 0; g < container.num_dense_groups; ++g) {
        const auto &dst_nodes = little_dst_sets[g];
        PartitionDescriptor pd = process_partition(
            edges_lists[g], dst_nodes, true, container.dense_groups[g].num_pipelines);

        if (pd.num_vertices == 0 || pd.num_edges == 0) {
            std::cout << "  - Dense group " << g
                      << ": no vertices/edges, skipping partition." << std::endl;
            continue;
        }
        container.dense_groups[g].partitions.push_back(pd);
        size_t part_idx = container.dense_groups[g].partitions.size() - 1;
        size_t flat_idx = container.dense_partition_order.size();
        container.dense_partition_order.emplace_back(g, part_idx);
        container.dense_partition_indices[g].push_back(flat_idx);
        container.num_dense_partitions = container.dense_partition_order.size();

        std::cout << "  - Dense group " << g << ": partition " << part_idx
                  << " | vertices: " << pd.num_vertices << ", dsts: "
                  << pd.num_dsts << ", edges: " << pd.num_edges
                  << ", pipelines: " << pd.num_pipelines << std::endl;

        for (unsigned int pip = 0; pip < pd.num_pipelines; ++pip) {
            std::cout << "      pipeline " << pip << ": "
                      << pd.pipeline_edges[pip].num_edges << " edges"
                      << std::endl;
        }
    }

    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        const auto &dst_nodes = big_dst_sets[g];
        size_t list_idx = container.num_dense_groups + g;
        PartitionDescriptor pd = process_partition(
            edges_lists[list_idx], dst_nodes, false,
            container.sparse_groups[g].num_pipelines);
        if (pd.num_vertices == 0 || pd.num_edges == 0) {
            std::cout << "  - Sparse group " << g
                      << ": no vertices/edges, skipping partition." << std::endl;
            continue;
        }
        container.sparse_groups[g].partitions.push_back(pd);
        size_t part_idx = container.sparse_groups[g].partitions.size() - 1;
        size_t flat_idx = container.sparse_partition_order.size();
        container.sparse_partition_order.emplace_back(g, part_idx);
        container.sparse_partition_indices[g].push_back(flat_idx);
        container.num_sparse_partitions = container.sparse_partition_order.size();

        std::cout << "  - Sparse group " << g << ": partition " << part_idx
                  << " | vertices: " << pd.num_vertices << ", dsts: "
                  << pd.num_dsts << ", edges: " << pd.num_edges
                  << ", pipelines: " << pd.num_pipelines << std::endl;

        for (unsigned int pip = 0; pip < pd.num_pipelines; ++pip) {
            std::cout << "      pipeline " << pip << ": "
                      << pd.pipeline_edges[pip].num_edges << " edges"
                      << std::endl;
        }
    }

    std::cout << "[SUCCESS] Graph partitioning and preprocessing complete."
              << std::endl;
    std::cout << "  Dense partitions: " << container.num_dense_partitions
              << " across " << container.num_dense_groups << " groups." << std::endl;
    std::cout << "  Sparse partitions: " << container.num_sparse_partitions
              << " across " << container.num_sparse_groups << " groups." << std::endl;

    return container;
}

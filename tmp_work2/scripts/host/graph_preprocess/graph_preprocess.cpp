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
        if (g >= little_offset_len || g >= little_pipeline_len) {
            std::cerr
                << "[ERROR] Mismatch in little merger group configuration!"
                << std::endl;
            exit(1);
        }
        group.pipeline_offset = LITTLE_MERGER_KERNEL_OFFSETS[g];
        group.num_pipelines = LITTLE_MERGER_PIPELINE_LENGTHS[g];
    }

    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        auto &group = container.sparse_groups[g];
        group.group_id = static_cast<unsigned int>(g);
        if (g >= big_offset_len || g >= big_pipeline_len) {
            std::cerr << "[ERROR] Mismatch in big merger group configuration!"
                      << std::endl;
            exit(1);
        }
        group.pipeline_offset = BIG_MERGER_KERNEL_OFFSETS[g];
        group.num_pipelines = BIG_MERGER_PIPELINE_LENGTHS[g];
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

    // --- PHASE 2: distribute destination vertices among groups and partitions
    // ---
    struct DestinationAssignment {
        bool is_dense;
        size_t group_idx;
        size_t partition_idx;
    };

    std::unordered_map<int, DestinationAssignment> dst_assignment;
    // For dense: [group][partition] -> set of dst nodes
    std::vector<std::vector<std::set<int>>> little_dst_sets(
        container.num_dense_groups, std::vector<std::set<int>>());
    // For sparse: [group][partition] -> set of dst nodes
    std::vector<std::vector<std::set<int>>> big_dst_sets(
        container.num_sparse_groups, std::vector<std::set<int>>());

    size_t total_dst_vertices = unique_dst_vertices.size();
    size_t vertex_cursor = 0;

    std::cout << "[PHASE 2] Total dst vertices: " << total_dst_vertices
              << std::endl;

    size_t one_partition_capacity =
        static_cast<size_t>(container.num_dense_groups) * LITTLE_MAX_DST +
        static_cast<size_t>(container.num_sparse_groups) * BIG_MAX_DST;

    size_t partition_number =
        (total_dst_vertices + one_partition_capacity - 1) /
        one_partition_capacity;

    std::cout << "[PHASE 2] Calculated partition number needed: "
              << partition_number << std::endl;

    // evenly distribute dst vertices into partitions
    for (size_t part = 0; part < partition_number; ++part) {
        size_t vertices_to_assign = std::min(
            one_partition_capacity, total_dst_vertices - vertex_cursor);

        std::cout << "[PHASE 2]   Partition " << part << ": assigning "
                  << vertices_to_assign << " vertices" << std::endl;

        // Create partition for each group
        for (size_t g = 0; g < container.num_dense_groups; ++g) {
            little_dst_sets[g].resize(part + 1);
        }
        for (size_t g = 0; g < container.num_sparse_groups; ++g) {
            big_dst_sets[g].resize(part + 1);
        }

        // Distribute vertices evenly across all groups for this partition
        size_t total_groups =
            container.num_dense_groups + container.num_sparse_groups;
        for (size_t i = 0; i < vertices_to_assign; ++i) {
            size_t group_idx = i % total_groups;
            int vertex_id = unique_dst_vertices[vertex_cursor++];
            if (group_idx >= container.num_dense_groups) {
                // Sparse group
                group_idx -= container.num_dense_groups;
                big_dst_sets[group_idx][part].insert(vertex_id);
                dst_assignment[vertex_id] = {false, group_idx, part};
                continue;
            }
            little_dst_sets[group_idx][part].insert(vertex_id);
            dst_assignment[vertex_id] = {true, group_idx, part};
        }
    }

    // Case 1: If dst_num > NUM_LITTLE_MERGERS * LITTLE_MAX_DST
    // Spread all dsts evenly for each dense group (1 partition for each group),
    // sparse empty if (total_dst_vertices <= dense_capacity_per_partition &&
    // DENSE_PARTITION_NUM > 0) {
    //     std::cout << "[PHASE 2] Case 1: dst_num (" << total_dst_vertices
    //               << ") <= dense_capacity_per_partition (" <<
    //               dense_capacity_per_partition
    //               << "), using single partition per dense group" <<
    //               std::endl;

    //     // Create one partition for each dense group
    //     for (size_t g = 0; g < container.num_dense_groups; ++g) {
    //         little_dst_sets[g].resize(1);
    //     }

    //     for (size_t g = 0; g < container.num_sparse_groups; ++g) {
    //         big_dst_sets[g].resize(1);
    //     }

    //     // Distribute vertices evenly across dense & sparse groups
    //     int all_group_num = container.num_dense_groups +
    //     container.num_sparse_groups; for (size_t i = 0; i <
    //     total_dst_vertices; ++i) {
    //         size_t group_idx = i % all_group_num;
    //         int vertex_id = unique_dst_vertices[i];
    //         if (group_idx >= container.num_dense_groups) {
    //             // Sparse group
    //             group_idx -= container.num_dense_groups;
    //             big_dst_sets[group_idx][0].insert(vertex_id);
    //             dst_assignment[vertex_id] = {false, group_idx, 0};
    //             continue;
    //         }
    //         little_dst_sets[group_idx][0].insert(vertex_id);
    //         dst_assignment[vertex_id] = {true, group_idx, 0};
    //     }
    //     vertex_cursor = total_dst_vertices;

    // } else {
    //     // Case 2 & 3: Can fit in dense partitions (possibly with overflow to
    //     sparse) std::cout << "[PHASE 2] Case 2/3: dst_num (" <<
    //     total_dst_vertices
    //               << ") <= total_dense_capacity (" << total_dense_capacity
    //               << "), using multiple partitions" << std::endl;

    //     // first, fill the first dense partition and first sparse partition
    //     {
    //         size_t vertices_to_assign = dense_capacity_per_partition;

    //         std::cout << "[PHASE 2]   Dense partition 0: assigning "
    //                   << vertices_to_assign << " vertices" << std::endl;

    //         // Create partition for each group
    //         for (size_t g = 0; g < container.num_dense_groups; ++g) {
    //             little_dst_sets[g].resize(1);
    //         }

    //         for (size_t g = 0; g < container.num_sparse_groups; ++g) {
    //             big_dst_sets[g].resize(1);
    //         }

    //         // Distribute vertices evenly across groups for this partition
    //         int all_group_num = container.num_dense_groups +
    //         container.num_sparse_groups; for (size_t i = 0; i <
    //         vertices_to_assign; ++i) {
    //             size_t group_idx = i % all_group_num;
    //             int vertex_id = unique_dst_vertices[i];
    //             if (group_idx >= container.num_dense_groups) {
    //                 // Sparse group
    //                 group_idx -= container.num_dense_groups;
    //                 big_dst_sets[group_idx][0].insert(vertex_id);
    //                 dst_assignment[vertex_id] = {false, group_idx, 0};
    //                 continue;
    //             }
    //             little_dst_sets[group_idx][0].insert(vertex_id);
    //             dst_assignment[vertex_id] = {true, group_idx, 0};
    //         }
    //         vertex_cursor += vertices_to_assign;
    //     }

    //     // Fill dense partitions for the rest
    //     for (size_t part = 1; part < DENSE_PARTITION_NUM && vertex_cursor <
    //     total_dst_vertices; ++part) {
    //         size_t vertices_to_assign = std::min(
    //             dense_capacity_per_partition,
    //             total_dst_vertices - vertex_cursor
    //         );

    //         std::cout << "[PHASE 2]   Dense partition " << part
    //                   << ": assigning " << vertices_to_assign << " vertices"
    //                   << std::endl;

    //         // Create partition for each group
    //         for (size_t g = 0; g < container.num_dense_groups; ++g) {
    //             little_dst_sets[g].resize(part + 1);
    //         }

    //         // Distribute vertices evenly across groups for this partition
    //         size_t vertices_per_group = vertices_to_assign /
    //         container.num_dense_groups; size_t extra_vertices =
    //         vertices_to_assign % container.num_dense_groups;

    //         for (size_t g = 0; g < container.num_dense_groups; ++g) {
    //             size_t count = vertices_per_group + (g < extra_vertices ? 1 :
    //             0); for (size_t i = 0; i < count && vertex_cursor <
    //             total_dst_vertices; ++i) {
    //                 int vertex_id = unique_dst_vertices[vertex_cursor++];
    //                 little_dst_sets[g][part].insert(vertex_id);
    //                 dst_assignment[vertex_id] = {true, g, part};
    //             }
    //         }
    //     }

    //     // If there are remaining vertices after filling all dense
    //     partitions, use sparse if (vertex_cursor < total_dst_vertices) {
    //         std::cout << "[PHASE 2] Case 3: Remaining " <<
    //         (total_dst_vertices - vertex_cursor)
    //                   << " vertices overflow to sparse groups" << std::endl;

    //         // Similar logic for sparse partitions
    //         size_t remaining_vertices = total_dst_vertices - vertex_cursor;
    //         size_t sparse_partition_idx = 0;

    //         while (vertex_cursor < total_dst_vertices) {
    //             size_t vertices_to_assign = std::min(
    //                 static_cast<size_t>(container.num_sparse_groups) *
    //                 BIG_MAX_DST, total_dst_vertices - vertex_cursor
    //             );

    //             std::cout << "[PHASE 2]   Sparse partition " <<
    //             sparse_partition_idx
    //                       << ": assigning " << vertices_to_assign << "
    //                       vertices" << std::endl;

    //             // Create partition for each group
    //             for (size_t g = 0; g < container.num_sparse_groups; ++g) {
    //                 if (big_dst_sets[g].size() <= sparse_partition_idx) {
    //                     big_dst_sets[g].resize(sparse_partition_idx + 1);
    //                 }
    //             }

    //             // Distribute vertices evenly across groups for this
    //             partition size_t vertices_per_group = vertices_to_assign /
    //             container.num_sparse_groups; size_t extra_vertices =
    //             vertices_to_assign % container.num_sparse_groups;

    //             for (size_t g = 0; g < container.num_sparse_groups; ++g) {
    //                 size_t count = vertices_per_group + (g < extra_vertices ?
    //                 1 : 0); for (size_t i = 0; i < count && vertex_cursor <
    //                 total_dst_vertices; ++i) {
    //                     int vertex_id = unique_dst_vertices[vertex_cursor++];
    //                     big_dst_sets[g][sparse_partition_idx].insert(vertex_id);
    //                     dst_assignment[vertex_id] = {false, g,
    //                     sparse_partition_idx};
    //                 }
    //             }

    //             sparse_partition_idx++;
    //         }
    //     }
    // }

    // Count total assignments
    size_t total_dense_assigned = 0;
    size_t total_sparse_assigned = 0;

    for (size_t g = 0; g < container.num_dense_groups; ++g) {
        for (size_t p = 0; p < little_dst_sets[g].size(); ++p) {
            total_dense_assigned += little_dst_sets[g][p].size();
        }
    }

    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        for (size_t p = 0; p < big_dst_sets[g].size(); ++p) {
            total_sparse_assigned += big_dst_sets[g][p].size();
        }
    }

    std::cout << "[PHASE 2] Dense groups assigned " << total_dense_assigned
              << " dst vertices across " << container.num_dense_groups
              << " groups." << std::endl;
    std::cout << "[PHASE 2] Sparse groups assigned " << total_sparse_assigned
              << " dst vertices across " << container.num_sparse_groups
              << " groups." << std::endl;

    // --- PHASE 3: assign edges based on destination ownership ---
    // For dense: [group][partition] -> list of edges
    std::vector<std::vector<std::vector<Edge>>> dense_edges_lists(
        container.num_dense_groups);
    // For sparse: [group][partition] -> list of edges
    std::vector<std::vector<std::vector<Edge>>> sparse_edges_lists(
        container.num_sparse_groups);

    // Initialize the sizes based on partition counts (after ensuring minimum 1)
    for (size_t g = 0; g < container.num_dense_groups; ++g) {
        dense_edges_lists[g].resize(little_dst_sets[g].size());
    }
    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        sparse_edges_lists[g].resize(big_dst_sets[g].size());
    }

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
            if (assignment.is_dense) {
                dense_edges_lists[assignment.group_idx]
                                 [assignment.partition_idx]
                                     .push_back({u, v, w});
                little_edge_num++;
            } else {
                sparse_edges_lists[assignment.group_idx]
                                  [assignment.partition_idx]
                                      .push_back({u, v, w});
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
                                 bool is_dense, unsigned int num_pipelines) {
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
            int start_idx = std::min(static_cast<int>(pip * edges_per_pipeline),
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
                uint32_t cur_src_buffer = static_cast<uint32_t>(
                    std::floor(static_cast<double>(src_id) / SRC_BUFFER_SIZE));
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

            int padding_size = (8 - (pd.pipeline_edges[pip].num_edges % 8)) % 8;
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
                pd.pipeline_edges[pip].weights.push_back(
                    cur_pip_edges[j].weight);
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
        for (size_t p = 0; p < little_dst_sets[g].size(); ++p) {
            const auto &dst_nodes = little_dst_sets[g][p];
            const auto &partition_edges = dense_edges_lists[g][p];

            PartitionDescriptor pd =
                process_partition(partition_edges, dst_nodes, true,
                                  container.dense_groups[g].num_pipelines);

            // Keep empty partitions so kernels are enqueued even when dst_num
            // == 0
            container.dense_groups[g].partitions.push_back(pd);
            size_t part_idx = container.dense_groups[g].partitions.size() - 1;
            size_t flat_idx = container.dense_partition_order.size();
            container.dense_partition_order.emplace_back(g, part_idx);
            container.dense_partition_indices[g].push_back(flat_idx);

            std::cout << "  - Dense group " << g << ": partition " << part_idx
                      << " | vertices: " << pd.num_vertices
                      << ", dsts: " << pd.num_dsts
                      << ", edges: " << pd.num_edges
                      << ", pipelines: " << pd.num_pipelines << std::endl;

            for (unsigned int pip = 0; pip < pd.num_pipelines; ++pip) {
                std::cout << "      pipeline " << pip << ": "
                          << pd.pipeline_edges[pip].num_edges << " edges"
                          << std::endl;
            }
        }
    }

    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        for (size_t p = 0; p < big_dst_sets[g].size(); ++p) {
            const auto &dst_nodes = big_dst_sets[g][p];
            const auto &partition_edges = sparse_edges_lists[g][p];

            PartitionDescriptor pd =
                process_partition(partition_edges, dst_nodes, false,
                                  container.sparse_groups[g].num_pipelines);

            // Keep empty partitions so kernels are enqueued even when dst_num
            // == 0
            container.sparse_groups[g].partitions.push_back(pd);
            size_t part_idx = container.sparse_groups[g].partitions.size() - 1;
            size_t flat_idx = container.sparse_partition_order.size();
            container.sparse_partition_order.emplace_back(g, part_idx);
            container.sparse_partition_indices[g].push_back(flat_idx);

            std::cout << "  - Sparse group " << g << ": partition " << part_idx
                      << " | vertices: " << pd.num_vertices
                      << ", dsts: " << pd.num_dsts
                      << ", edges: " << pd.num_edges
                      << ", pipelines: " << pd.num_pipelines << std::endl;

            for (unsigned int pip = 0; pip < pd.num_pipelines; ++pip) {
                std::cout << "      pipeline " << pip << ": "
                          << pd.pipeline_edges[pip].num_edges << " edges"
                          << std::endl;
            }
        }
    }

    container.num_dense_partitions =
        static_cast<unsigned int>(container.dense_partition_order.size());
    container.num_sparse_partitions =
        static_cast<unsigned int>(container.sparse_partition_order.size());

    std::cout << "[SUCCESS] Graph partitioning and preprocessing complete."
              << std::endl;
    std::cout << "  Dense partitions: " << container.num_dense_partitions
              << " across " << container.num_dense_groups << " groups."
              << std::endl;
    std::cout << "  Sparse partitions: " << container.num_sparse_partitions
              << " across " << container.num_sparse_groups << " groups."
              << std::endl;

    // --- DETAILED DEBUG PRINTS: Print each partition's detailed information
    // ---
    std::cout << "\n========== DETAILED PARTITION INFORMATION =========="
              << std::endl;

    // Print dense partitions
    for (size_t g = 0; g < container.num_dense_groups; ++g) {
        const auto &group = container.dense_groups[g];
        std::cout << "\n--- DENSE GROUP " << g
                  << " (pipeline_offset=" << group.pipeline_offset
                  << ", num_pipelines=" << group.num_pipelines << ") ---"
                  << std::endl;

        for (size_t p = 0; p < group.partitions.size(); ++p) {
            const auto &partition = group.partitions[p];
            std::cout << "\n  [DENSE] Group " << g << ", Partition " << p << ":"
                      << std::endl;
            std::cout << "    Type: DENSE (LITTLE)" << std::endl;
            std::cout << "    num_vertices: " << partition.num_vertices
                      << std::endl;
            std::cout << "    num_dsts: " << partition.num_dsts << std::endl;
            std::cout << "    num_edges: " << partition.num_edges << std::endl;
            std::cout << "    num_pipelines: " << partition.num_pipelines
                      << std::endl;

            // Print vertex mappings
            std::cout << "    Vertex Mappings (local_id -> global_id):"
                      << std::endl;
            for (const auto &[local_id, global_id] : partition.vtx_map_rev) {
                std::cout << "      local_id " << local_id << " -> global_id "
                          << global_id << std::endl;
            }

            // Print graph edges in "a -> b" format for each pipeline
            for (unsigned int pip = 0; pip < partition.num_pipelines; ++pip) {
                const auto &pipeline_edges = partition.pipeline_edges[pip];
                std::cout << "\n    Pipeline " << pip
                          << " (num_edges=" << pipeline_edges.num_edges
                          << "):" << std::endl;

                // Print offsets array
                std::cout << "      Offsets array (size="
                          << pipeline_edges.offsets.size() << "):" << std::endl;
                std::cout << "        ";
                for (size_t i = 0; i < pipeline_edges.offsets.size(); ++i) {
                    std::cout << "[" << i << "]=" << pipeline_edges.offsets[i]
                              << " ";
                    if ((i + 1) % 16 == 0 &&
                        i + 1 < pipeline_edges.offsets.size()) {
                        std::cout << std::endl << "        ";
                    }
                }
                std::cout << std::endl;

                // Print columns array
                std::cout << "      Columns array (size="
                          << pipeline_edges.columns.size() << "):" << std::endl;
                std::cout << "        ";
                for (size_t i = 0; i < pipeline_edges.columns.size(); ++i) {
                    std::cout << "[" << i << "]=" << pipeline_edges.columns[i]
                              << " ";
                    if ((i + 1) % 16 == 0 &&
                        i + 1 < pipeline_edges.columns.size()) {
                        std::cout << std::endl << "        ";
                    }
                }
                std::cout << std::endl;

                // Print weights array
                std::cout << "      Weights array (size="
                          << pipeline_edges.weights.size() << "):" << std::endl;
                std::cout << "        ";
                for (size_t i = 0; i < pipeline_edges.weights.size(); ++i) {
                    std::cout << "[" << i << "]=" << pipeline_edges.weights[i]
                              << " ";
                    if ((i + 1) % 16 == 0 &&
                        i + 1 < pipeline_edges.weights.size()) {
                        std::cout << std::endl << "        ";
                    }
                }
                std::cout << std::endl;

                // Print edges in "a -> b" format
                std::cout << "      Edges (local_id format):" << std::endl;
                for (int v = 0; v < static_cast<int>(partition.num_vertices);
                     ++v) {
                    int start = pipeline_edges.offsets[v];
                    int end = pipeline_edges.offsets[v + 1];
                    for (int e = start; e < end; ++e) {
                        int src_local = v;
                        int dst_local = pipeline_edges.columns[e];
                        int weight = pipeline_edges.weights[e];
                        int src_global =
                            partition.vtx_map_rev.count(src_local)
                                ? partition.vtx_map_rev.at(src_local)
                                : -1;
                        int dst_global = -1;
                        if (dst_local == 0x7FFFFFFF) {
                            dst_global = -1;
                        } else if (partition.vtx_map_rev.count(dst_local)) {
                            dst_global = partition.vtx_map_rev.at(dst_local);
                        }
                        if (dst_global == -1) {
                            std::cout << "        local[" << src_local
                                      << "] -> DUMMY (weight=" << weight << ")"
                                      << std::endl;
                        } else {
                            std::cout
                                << "        local[" << src_local
                                << "] -> local[" << dst_local << "] (global["
                                << src_global << "] -> global[" << dst_global
                                << "], weight=" << weight << ")" << std::endl;
                        }
                    }
                }
            }
        }
    }

    // Print sparse partitions
    for (size_t g = 0; g < container.num_sparse_groups; ++g) {
        const auto &group = container.sparse_groups[g];
        std::cout << "\n--- SPARSE GROUP " << g
                  << " (pipeline_offset=" << group.pipeline_offset
                  << ", num_pipelines=" << group.num_pipelines << ") ---"
                  << std::endl;

        for (size_t p = 0; p < group.partitions.size(); ++p) {
            const auto &partition = group.partitions[p];
            std::cout << "\n  [SPARSE] Group " << g << ", Partition " << p
                      << ":" << std::endl;
            std::cout << "    Type: SPARSE (BIG)" << std::endl;
            std::cout << "    num_vertices: " << partition.num_vertices
                      << std::endl;
            std::cout << "    num_dsts: " << partition.num_dsts << std::endl;
            std::cout << "    num_edges: " << partition.num_edges << std::endl;
            std::cout << "    num_pipelines: " << partition.num_pipelines
                      << std::endl;

            // Print vertex mappings
            std::cout << "    Vertex Mappings (local_id -> global_id):"
                      << std::endl;
            for (const auto &[local_id, global_id] : partition.vtx_map_rev) {
                std::cout << "      local_id " << local_id << " -> global_id "
                          << global_id << std::endl;
            }

            // Print graph edges in "a -> b" format for each pipeline
            for (unsigned int pip = 0; pip < partition.num_pipelines; ++pip) {
                const auto &pipeline_edges = partition.pipeline_edges[pip];
                std::cout << "\n    Pipeline " << pip
                          << " (num_edges=" << pipeline_edges.num_edges
                          << "):" << std::endl;

                // Print offsets array
                std::cout << "      Offsets array (size="
                          << pipeline_edges.offsets.size() << "):" << std::endl;
                std::cout << "        ";
                for (size_t i = 0; i < pipeline_edges.offsets.size(); ++i) {
                    std::cout << "[" << i << "]=" << pipeline_edges.offsets[i]
                              << " ";
                    if ((i + 1) % 16 == 0 &&
                        i + 1 < pipeline_edges.offsets.size()) {
                        std::cout << std::endl << "        ";
                    }
                }
                std::cout << std::endl;

                // Print columns array
                std::cout << "      Columns array (size="
                          << pipeline_edges.columns.size() << "):" << std::endl;
                std::cout << "        ";
                for (size_t i = 0; i < pipeline_edges.columns.size(); ++i) {
                    std::cout << "[" << i << "]=" << pipeline_edges.columns[i]
                              << " ";
                    if ((i + 1) % 16 == 0 &&
                        i + 1 < pipeline_edges.columns.size()) {
                        std::cout << std::endl << "        ";
                    }
                }
                std::cout << std::endl;

                // Print weights array
                std::cout << "      Weights array (size="
                          << pipeline_edges.weights.size() << "):" << std::endl;
                std::cout << "        ";
                for (size_t i = 0; i < pipeline_edges.weights.size(); ++i) {
                    std::cout << "[" << i << "]=" << pipeline_edges.weights[i]
                              << " ";
                    if ((i + 1) % 16 == 0 &&
                        i + 1 < pipeline_edges.weights.size()) {
                        std::cout << std::endl << "        ";
                    }
                }
                std::cout << std::endl;

                // Print edges in "a -> b" format
                std::cout << "      Edges (local_id format):" << std::endl;
                for (int v = 0; v < static_cast<int>(partition.num_vertices);
                     ++v) {
                    int start = pipeline_edges.offsets[v];
                    int end = pipeline_edges.offsets[v + 1];
                    for (int e = start; e < end; ++e) {
                        int src_local = v;
                        int dst_local = pipeline_edges.columns[e];
                        int weight = pipeline_edges.weights[e];
                        int src_global =
                            partition.vtx_map_rev.count(src_local)
                                ? partition.vtx_map_rev.at(src_local)
                                : -1;
                        int dst_global = -1;
                        if (dst_local == 0x7FFFFFFF) {
                            dst_global = -1;
                        } else if (partition.vtx_map_rev.count(dst_local)) {
                            dst_global = partition.vtx_map_rev.at(dst_local);
                        }
                        if (dst_global == -1) {
                            std::cout << "        local[" << src_local
                                      << "] -> DUMMY (weight=" << weight << ")"
                                      << std::endl;
                        } else {
                            std::cout
                                << "        local[" << src_local
                                << "] -> local[" << dst_local << "] (global["
                                << src_global << "] -> global[" << dst_global
                                << "], weight=" << weight << ")" << std::endl;
                        }
                    }
                }
            }
        }
    }

    std::cout
        << "\n========== END OF DETAILED PARTITION INFORMATION ==========\n"
        << std::endl;

    return container;
}

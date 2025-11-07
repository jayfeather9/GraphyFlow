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
 * @brief 处理单个分区：压缩顶点 ID 并将边分配到多个 pipeline 中
 *
 * @param partition_edges 该分区的所有边（使用全局 ID）
 * @param partition_dst_nodes 该分区负责的目标顶点集合（全局 ID）
 * @param is_dense 是否为 dense 分区（little 为 true，big 为 false）
 * @param num_pipelines 该分区的 pipeline 数量
 * @return PartitionDescriptor 处理后的分区描述符
 */
static PartitionDescriptor
processPartition(const std::vector<Edge> &partition_edges,
                 const std::set<int> &partition_dst_nodes, bool is_dense,
                 int num_pipelines) {
    PartitionDescriptor pd;
    pd.is_dense = is_dense;
    pd.num_pipelines = num_pipelines;

    if (partition_edges.empty()) {
        std::cout << "  - Partition has no edges. Creating empty partition."
                  << std::endl;
        pd.num_edges = 0;
        pd.num_vertices = 0;
        pd.num_dsts = 0;
        return pd;
    }

    // --- 4.1: Collect unique vertices and build ID mappings ---
    // 根据边集维护所有相关的顶点：本partition需要维护mirror顶点
    std::set<int> local_vertices_set;
    for (const auto &edge : partition_edges) {
        local_vertices_set.insert(edge.src);
        local_vertices_set.insert(edge.dest);
    }

    // --- 4.1.1: Create ordered list of destination vertices (already
    // sorted by indegree from PHASE 1) ---
    std::vector<int> ordered_dst_vertices(partition_dst_nodes.begin(),
                                          partition_dst_nodes.end());
    // 原先按照indegree排序，现在进行shuffle
    // random shuffle ordered_dst_vertices
    std::srand(42);
    std::random_shuffle(ordered_dst_vertices.begin(),
                        ordered_dst_vertices.end());

    // 给partition中的每个目标顶点分配局部id：目标顶点是master顶点，需要首先关注
    int local_id_counter = 0;
    // First, map destination vertices to guarantee they have
    // lower-range IDs
    for (int global_id : ordered_dst_vertices) {
        pd.vtx_map[global_id] = local_id_counter;
        pd.vtx_map_rev[local_id_counter] = global_id;
        local_id_counter++;
    }
    pd.num_dsts = partition_dst_nodes.size();

    // 为剩余的源顶点分配
    // Then, map the remaining source vertices
    for (int global_id : local_vertices_set) {
        if (pd.vtx_map.find(global_id) == pd.vtx_map.end()) {
            pd.vtx_map[global_id] = local_id_counter;
            pd.vtx_map_rev[local_id_counter] = global_id;
            local_id_counter++;
        }
    }
    pd.num_vertices = local_vertices_set.size();

    // --- 4.2: Rewrite edges with local, compressed IDs and sort by src
    // ---
    // 将顶点使用local id进行重新组织
    std::vector<Edge> local_edges;
    local_edges.reserve(partition_edges.size());

    for (const auto &global_edge : partition_edges) {
        uint32_t src_id = pd.vtx_map[global_edge.src];
        uint32_t dest_id = pd.vtx_map[global_edge.dest];
        uint32_t weight = global_edge.weight;
        local_edges.push_back({(int)src_id, (int)dest_id, (int)weight});
    }

    // CSR 格式要求：边按照源顶点分组，便于后续构造offsets数组
    // Sort edges by source node ID (ascending)
    std::sort(local_edges.begin(), local_edges.end(),
              [](const Edge &a, const Edge &b) { return a.src < b.src; });

    pd.num_edges = local_edges.size();

    // --- 4.3: Distribute edges evenly among pipelines ---
    // 将边分配到多个pipeline
    pd.pipeline_edges.resize(num_pipelines);
    // 将边均匀的分配到每个pipeline
    int edges_per_pipeline = (pd.num_edges + num_pipelines - 1) / num_pipelines;

    for (int pip = 0; pip < num_pipelines; ++pip) {
        // 当前pipeline分配到的边
        std::vector<Edge> cur_pip_edges;
        pd.pipeline_edges[pip].pipeline_id = pip;
        int start_idx = std::min(pip * edges_per_pipeline, (int)pd.num_edges);
        int end_idx =
            std::min(start_idx + edges_per_pipeline, (int)pd.num_edges);

        // 记录buffer信息
        uint32_t last_src_buffer = 0;
        uint32_t last_src_id = 0;
        // 处理边并添加对齐填充（Dense 分区）
        for (int edge_idx = start_idx; edge_idx < end_idx; ++edge_idx) {
            auto local_edge = local_edges[edge_idx];
            uint32_t src_id = local_edge.src;
            uint32_t dest_id = local_edge.dest;
            uint32_t weight = local_edge.weight;
            // 将源顶点按照 SRC_BUFFER_SIZE 分组
            uint32_t cur_src_buffer = floor(src_id / SRC_BUFFER_SIZE);
            // 当切换到新的 buffer 时，将当前 buffer 填充到 8 的倍数
            // 硬件 SIMD 优化：8 条边一起处理
            if (is_dense && cur_src_buffer != last_src_buffer) {
                uint32_t mod8 = cur_pip_edges.size() % 8;
                if (mod8 != 0) {
                    // Pad with dummy edges to align to 8-edge boundary
                    for (uint32_t pad = 0; pad < (8 - mod8); pad++) {
                        cur_pip_edges.push_back({last_src_id, 0x7FFFFFFF, 1});
                    }
                }
                last_src_buffer = cur_src_buffer;
            }
            last_src_id = src_id;
            cur_pip_edges.push_back({src_id, dest_id, weight});
        }

        // 进行尾部填充
        pd.pipeline_edges[pip].num_edges = cur_pip_edges.size();

        int padding_size = (8 - (pd.pipeline_edges[pip].num_edges % 8)) % 8;
        pd.pipeline_edges[pip].num_edges += padding_size;

        // Build CSR for this pipeline
        pd.pipeline_edges[pip].offsets.resize(pd.num_vertices + 1, 0);
        pd.pipeline_edges[pip].columns.reserve(
            pd.pipeline_edges[pip].num_edges);
        pd.pipeline_edges[pip].weights.reserve(
            pd.pipeline_edges[pip].num_edges);

        // Count out-degrees for this pipeline's edges
        std::vector<int> out_degree(pd.num_vertices, 0);
        for (int j = 0; j < cur_pip_edges.size(); ++j) {
            out_degree[cur_pip_edges[j].src]++;
            if (j == cur_pip_edges.size() - 1) {
                // If this is the last edge, add padding edges if needed
                for (int p = 0; p < padding_size; ++p) {
                    out_degree[cur_pip_edges[j].src]++;
                }
            }
        }

        // Build offsets
        pd.pipeline_edges[pip].offsets[0] = 0;
        for (int v = 0; v < pd.num_vertices; ++v) {
            pd.pipeline_edges[pip].offsets[v + 1] =
                pd.pipeline_edges[pip].offsets[v] + out_degree[v];
        }

        // Fill columns and weights
        std::vector<int> current_offset = pd.pipeline_edges[pip].offsets;
        for (int j = 0; j < cur_pip_edges.size(); ++j) {
            int src = cur_pip_edges[j].src;
            int idx = current_offset[src]++;
            pd.pipeline_edges[pip].columns.push_back(cur_pip_edges[j].dest);
            pd.pipeline_edges[pip].weights.push_back(cur_pip_edges[j].weight);
            if (j == cur_pip_edges.size() - 1) {
                // If this is the last edge, add padding edges if needed
                for (int p = 0; p < padding_size; ++p) {
                    idx = current_offset[src]++;
                    pd.pipeline_edges[pip].columns.push_back(0x7FFFFFFF);
                    pd.pipeline_edges[pip].weights.push_back(1);
                }
            }
        }
    }

    return pd;
}

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
    std::cout << "--- Starting Graph Partitioning and Preprocessing "
                 "(2-Partition Mode) ---"
              << std::endl;
    PartitionContainer container;
    container.num_graph_vertices = graph->num_vertices;
    container.num_graph_edges = graph->num_edges;
    printf("Global graph has %d vertices and %d edges.\n", graph->num_vertices,
           graph->num_edges);

    std::cout << "[INFO] Creating 2 partitions: 1 little (max "
              << LITTLE_MAX_DST << " dsts) and 1 big (max " << BIG_MAX_DST
              << " dsts)" << std::endl;

    // --- PHASE 1: Identify and Collect All Unique Destination Vertices ---
    // 顶点的集合
    std::set<int> unique_dst_vertices_set;
    // 记录了顶点的in degree
    std::unordered_map<int, int> node_indegrees;
    for (int i = 0; i < graph->num_edges; ++i) {
        int dst = graph->columns[i];
        unique_dst_vertices_set.insert(dst);
        // 找不到顶点，初始化
        if (node_indegrees.find(dst) == node_indegrees.end()) {
            node_indegrees[dst] = 0;
        }
        node_indegrees[dst]++;
    }
    // 按照in degree排序的顶点集合
    std::vector<int> unique_dst_vertices(unique_dst_vertices_set.begin(),
                                         unique_dst_vertices_set.end());

    // Sort unique_dst_vertices by indegree (descending order) to prioritize
    // high-degree nodes for little partition
    std::sort(unique_dst_vertices.begin(), unique_dst_vertices.end(),
              [&node_indegrees](int a, int b) {
                  return node_indegrees[a] > node_indegrees[b];
              });

    std::cout << "[PHASE 1] Found " << unique_dst_vertices.size()
              << " unique destination vertices (sorted by indegree)."
              << std::endl;

    // --- PHASE 2: Distribute Destination Vertices to 2 Partitions ---
    std::vector<std::set<int>> little_dst_sets, big_dst_sets;
    std::unordered_map<int, int>
        dst_vertex_to_partition_map; // 0 ~ little_partition_sizes.size()-1 =>
                                     // little

    // 每个partition的size
    std::vector<size_t> little_partition_sizes;
    std::vector<size_t> big_partition_sizes;

    size_t remaining_dsts = unique_dst_vertices.size();
    // 每次分配时，先分配LITTLE_MAX_DST个给little，再分配BIG_MAX_DST个给big
    while (remaining_dsts > 0) {
        // 每次批量分配一定数量给little
        size_t assign_to_little =
            std::min((size_t)LITTLE_MAX_DST, remaining_dsts);
        // 如果剩余的顶点数不足 LITTLE_MAX_DST，那么只分配 80% 给 little
        // 分区，留 20% 给后续的 big 分区 维持little和big数量上的相同
        if (assign_to_little == remaining_dsts) {
            assign_to_little = (size_t)(remaining_dsts * 0.8);
        }
        // 记录little partition的元素数量
        little_partition_sizes.push_back(assign_to_little);
        // 创建一个元素：目前用于占位
        little_dst_sets.emplace_back();
        remaining_dsts -= assign_to_little;

        // 分配BIG
        size_t assign_to_big = std::min((size_t)BIG_MAX_DST, remaining_dsts);
        big_partition_sizes.push_back(assign_to_big);
        big_dst_sets.emplace_back();
        remaining_dsts -= assign_to_big;
    }

    size_t vertex_idx = 0;
    // 遍历所有的little：讲顶点按照in degree大小分配给little
    for (size_t p = 0; p < little_partition_sizes.size(); ++p) {
        size_t part_size = little_partition_sizes[p];
        for (size_t i = 0; i < part_size; ++i) {
            int vertex_id = unique_dst_vertices[vertex_idx++];
            little_dst_sets[p].insert(vertex_id);
            // 记录顶点在那个partition中
            dst_vertex_to_partition_map[vertex_id] = p; // Little partition
        }
    }

    for (size_t p = 0; p < big_partition_sizes.size(); ++p) {
        size_t part_size = big_partition_sizes[p];
        for (size_t i = 0; i < part_size; ++i) {
            int vertex_id = unique_dst_vertices[vertex_idx++];
            big_dst_sets[p].insert(vertex_id);
            dst_vertex_to_partition_map[vertex_id] =
                p + little_partition_sizes.size(); // Big partition
        }
    }

    std::cout << "[PHASE 2] Little partition assigned "
              << std::accumulate(little_partition_sizes.begin(),
                                 little_partition_sizes.end(), 0)
              << " dst vertices across " << little_partition_sizes.size()
              << " partitions." << std::endl;
    std::cout << "[PHASE 2] Big partition assigned "
              << std::accumulate(big_partition_sizes.begin(),
                                 big_partition_sizes.end(), 0)
              << " dst vertices across " << big_partition_sizes.size()
              << " partitions." << std::endl;

    // --- PHASE 3: Assign Edges to 2 Partitions Based on Destination Vertex ---
    // 在得到顶点分配后，将相应的边加入partition中
    // 记录每个partition的edge的列表
    std::vector<std::vector<Edge>> edges_lists;
    // 确定大小为little和big的partition数目之和
    edges_lists.resize(little_partition_sizes.size() +
                       big_partition_sizes.size());
    size_t little_edge_num = 0, big_edge_num = 0;
    // 遍历顶点，按照src分配对应的边
    for (int u = 0; u < graph->num_vertices; ++u) {
        // offsets[u] ～ offsets[u+1]-1：顶点 u 的所有出边在 columns
        // 数组中的范围
        for (int i = graph->offsets[u]; i < graph->offsets[u + 1]; ++i) {
            // 出边的顶点id
            int v = graph->columns[i];
            int w = graph->weights[i];

            // Find which partition this edge belongs to
            auto it = dst_vertex_to_partition_map.find(v);
            if (it != dst_vertex_to_partition_map.end()) {
                int partition_idx = it->second;
                // 按照src分配边倒partition
                edges_lists[partition_idx].push_back({u, v, w});
                if (partition_idx < little_partition_sizes.size()) {
                    little_edge_num++;
                } else {
                    big_edge_num++;
                }
            } else {
                std::cerr << "[ERROR] Destination vertex " << v
                          << " not found in any partition!" << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "[PHASE 3] Assigned " << little_edge_num
              << " edges to little partitions." << std::endl;
    std::cout << "[PHASE 3] Assigned " << big_edge_num
              << " edges to big partitions." << std::endl;

    // --- PHASE 4: Process Each Partition (Compress IDs and Distribute Edges)
    // ---
    std::cout << "[PHASE 4] Processing partitions..." << std::endl;

    // 处理 little 分区
    for (size_t p = 0; p < little_partition_sizes.size(); ++p) {
        PartitionDescriptor little_pd = processPartition(
            edges_lists[p], little_dst_sets[p], true, LITTLE_KERNEL_NUM);
        container.DPs.push_back(little_pd);
        std::cout << "  - Little partition " << p << ": "
                  << little_pd.num_vertices << " vertices, "
                  << little_pd.num_dsts << " dsts, " << little_pd.num_edges
                  << " edges distributed to " << little_pd.num_pipelines
                  << " pipelines." << std::endl;
        // print how much edge for each partition each pipeline
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            std::cout << "    - Pipeline " << pip << ": "
                      << little_pd.pipeline_edges[pip].num_edges << " edges."
                      << std::endl;
        }
    }

    // 处理 big 分区
    for (size_t p = 0; p < big_partition_sizes.size(); ++p) {
        PartitionDescriptor big_pd =
            processPartition(edges_lists[p + little_partition_sizes.size()],
                             big_dst_sets[p], false, BIG_KERNEL_NUM);
        container.SPs.push_back(big_pd);
        std::cout << "  - Big partition " << p << ": " << big_pd.num_vertices
                  << " vertices, " << big_pd.num_dsts << " dsts, "
                  << big_pd.num_edges << " edges distributed to "
                  << big_pd.num_pipelines << " pipelines." << std::endl;
        // print how much edge for each partition each pipeline
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            std::cout << "    - Pipeline " << pip << ": "
                      << big_pd.pipeline_edges[pip].num_edges << " edges."
                      << std::endl;
        }
    }

    container.num_dense_partitions = container.DPs.size();
    container.num_sparse_partitions = container.SPs.size();

    std::cout << "[SUCCESS] Graph partitioning and preprocessing complete "
                 "(2-partition mode)."
              << std::endl;
    std::cout << "  Total: " << container.num_dense_partitions << " dense + "
              << container.num_sparse_partitions << " sparse partitions."
              << std::endl;
    return container;
}

#include "graph_partition.h"

std::vector<GraphCSR> partition_graph(const GraphCSR &graph, int num_partitions, std::vector<float> partition_weights) {
    std::vector<GraphCSR> partitions(num_partitions);
    // partition the graph based on dst id
    // partition different dst id to different partition
    // first, change the format to vector-vector
    // first vector is dst, second vector is different edges (src, weight)
    std::vector<std::vector<std::pair<int, int>>> adj_list(graph.num_nodes);
    for (int src = 0; src < graph.num_vertices; ++src) {
        for (int i = graph.offsets[src]; i < graph.offsets[src + 1]; ++i) {
            int dst = graph.columns[i];
            int weight = graph.weights[i];
            adj_list[dst].emplace_back(src, weight);
        }
    }
    // then, count the number of edges for each dst and number of edges
    std::vector<std::pair<int, int>> dst_sizes(graph.num_vertices);
    int total_edge_cnt = 0;
    for (int dst = 0; dst < graph.num_vertices; ++dst) {
        dst_sizes[dst] = {dst, static_cast<int>(adj_list[dst].size())};
        total_edge_cnt += adj_list[dst].size();
    }
    // sort the dst by number of edges from big to small
    std::sort(dst_sizes.begin(), dst_sizes.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });
    // normalize the partition weights to sum = total_edge_cnt
    float total_weight = 0.0f;
    for (float w : partition_weights) {
        total_weight += w;
    }
    for (float &w : partition_weights) {
        w = w / total_weight * total_edge_cnt;
    }
    // assign dst to partitions
    // iterate dst from big to small
    // for each dst, find the first partition that can hold the dst
    std::vector<std::vector<int>> partition_dst_ids(num_partitions);
    std::vector<int> partition_edge_counts(num_partitions, 0);
    for (const auto &[dst, size] : dst_sizes) {
        bool assigned = false;
        for (int p = 0; p < num_partitions; ++p) {
            if (partition_edge_counts[p] + size <= partition_weights[p]) {
                partition_dst_ids[p].push_back(dst);
                partition_edge_counts[p] += size;
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            // if no partition can hold the dst, assign it to the partition with the most space left
            // find the partition that partition_weights[p] - partition_edge_counts[p] is the largest
            int best_p = 0;
            float best_space = partition_weights[0] - partition_edge_counts[0];
            for (int p = 1; p < num_partitions; ++p) {
                float space = partition_weights[p] - partition_edge_counts[p];
                if (space > best_space) {
                    best_space = space;
                    best_p = p;
                }
            }
            partition_dst_ids[best_p].push_back(dst);
            partition_edge_counts[best_p] += size;
        }
    }
    // now we have the dst ids for each partition
    // construct the partitions
    for (int p = 0; p < num_partitions; ++p) {
        const auto &dst_ids = partition_dst_ids[p];
        // count the number of vertices and edges
        std::unordered_set<int> vertex_set;
        int edge_count = 0;
        for (int dst : dst_ids) {
            for (const auto &[src, weight] : adj_list[dst]) {
                vertex_set.insert(src);
                vertex_set.insert(dst);
                edge_count++;
            }
        }
        partitions[p].num_vertices = vertex_set.size();
        partitions[p].num_edges = edge_count;
        partitions[p].offsets.resize(partitions[p].num_vertices + 1, 0);
        partitions[p].columns.reserve(edge_count);
        partitions[p].weights.reserve(edge_count);
        // use GraphCSR's std::unordered_map<int, int> vtx_map to map old vertex id to new vertex id
        // new vertex id from 0 to num_vertices - 1
        partitions[p].vtx_map.clear();
        int new_id = 0;
        for (int v : vertex_set) {
            partitions[p].vtx_map[v] = new_id;
            partitions[p].vtx_map_rev[new_id] = v;
            new_id++;
        }
        // fill in offsets, columns, weights
        // a map from new src to list of (new dst, weight)
        std::unordered_map<int, std::vector<std::pair<int, int>>> new_adj_list;
        for (int dst : dst_ids) {
            int new_dst = partitions[p].vtx_map[dst];
            for (const auto &[src, weight] : adj_list[dst]) {
                int new_src = partitions[p].vtx_map[src];
                // the edge is src -> dst
                partitions[p].offsets[new_src + 1]++;
                new_adj_list[new_src].emplace_back(new_dst, weight);
            }
        }
        // compute offsets
        for (int i = 0; i < partitions[p].offsets.size() - 1; ++i) {
            partitions[p].offsets[i + 1] += partitions[p].offsets[i];
        }
        // fill in columns and weights
        for (int src = 0; src < partitions[p].num_vertices; ++src) {
            for (const auto &[dst, weight] : new_adj_list[src]) {
                partitions[p].columns.push_back(dst);
                partitions[p].weights.push_back(weight);
            }
        }
    }
    return partitions;
}

GraphCSR merge_partitions(const std::vector<GraphCSR> &partitions) {
    GraphCSR merged_graph;
    merged_graph.num_vertices = 0;
    merged_graph.num_edges = 0;
    // vertices may overlap, just get the max id
    int max_id = -1;
    for (const auto &part : partitions) {
        merged_graph.num_edges += part.num_edges;
        for (const auto &[old_id, new_id] : part.vtx_map) {
            if (old_id > max_id) {
                max_id = old_id;
            }
        }
    }
    merged_graph.num_vertices = max_id + 1;
    merged_graph.offsets.resize(merged_graph.num_vertices + 1, 0);
    merged_graph.columns.reserve(merged_graph.num_edges);
    merged_graph.weights.reserve(merged_graph.num_edges);
    // a map from ori src to list of (ori dst, weight)
    std::unordered_map<int, std::vector<std::pair<int, int>>> new_adj_list;
    // fill in offsets, columns, weights
    for (const auto &part : partitions) {
        for (int src = 0; src < part.num_vertices; ++src) {
            int ori_src = part.vtx_map_rev.at(src);
            for (int i = part.offsets[src]; i < part.offsets[src + 1]; ++i) {
                int ori_dst = part.vtx_map_rev.at(part.columns[i]);
                int weight = part.weights[i];
                merged_graph.offsets[ori_src + 1]++;
                new_adj_list[ori_src].emplace_back(ori_dst, weight);
            }
        }
    }
    // compute offsets
    for (int i = 0; i < merged_graph.offsets.size() - 1; ++i) {
        merged_graph.offsets[i + 1] += merged_graph.offsets[i];
    }
    // fill in columns and weights
    for (int src = 0; src < merged_graph.num_vertices; ++src) {
        for (const auto &[dst, weight] : new_adj_list[src]) {
            merged_graph.columns.push_back(dst);
            merged_graph.weights.push_back(weight);
        }
    }
    return merged_graph;
}

// REPLACE THE ENTIRE FILE CONTENT WITH THIS:
#include "generated_host.h"
#include <iostream>
#include <map>

// Constructor
AlgorithmHost::AlgorithmHost(cl::Context &context, cl::Kernel &kernel,
                             cl::CommandQueue &q)
    : m_context(context), m_kernel(kernel), m_q(q), m_num_vertices(0),
      m_num_batches(0) {}

void AlgorithmHost::setup_buffers(const GraphCSR &graph, int start_node) {
    m_num_vertices = graph.num_vertices;
    cl_int err;

    // Initialize host-side distances for Bellman-Ford
    h_distances.assign(m_num_vertices, std::numeric_limits<int>::max());
    if (start_node < m_num_vertices) {
        h_distances[start_node] = 0;
    }

    // --- Pack graph data into batches for the kernel ---
    // This is a simplified packing logic. A real implementation would be more
    // complex.
    h_i_0_20.clear();
    edge_t edge_pack[PE_NUM];
    int pack_idx = 0;

    for (int i = 0; i < graph.num_edges; ++i) {
        // Find source vertex 'u' for edge 'i'
        int u = 0;
        for (int j = 0; j < graph.num_vertices; ++j) {
            if (graph.offsets[j] <= i && i < graph.offsets[j + 1]) {
                u = j;
                break;
            }
        }
        int v = graph.columns[i];

        edge_pack[pack_idx].src.id = u;
        edge_pack[pack_idx].src.distance = h_distances[u];
        edge_pack[pack_idx].dst.id = v;
        edge_pack[pack_idx].dst.distance = h_distances[v];
        edge_pack[pack_idx].weight = graph.weights[i];

        pack_idx++;
        if (pack_idx == PE_NUM) {
            struct_ebu_7_t batch;
            memcpy(batch.data, edge_pack, sizeof(edge_pack));
            batch.end_flag = false;
            batch.end_pos = PE_NUM;
            h_i_0_20.push_back(batch);
            pack_idx = 0;
        }
    }

    // Handle the last partial batch
    if (pack_idx > 0) {
        struct_ebu_7_t batch;
        memcpy(batch.data, edge_pack, pack_idx * sizeof(edge_t));
        batch.end_flag = false; // Will be set to true later
        batch.end_pos = pack_idx;
        h_i_0_20.push_back(batch);
    }

    // Mark the very last batch with the end_flag
    if (!h_i_0_20.empty()) {
        h_i_0_20.back().end_flag = true;
    }

    m_num_batches = h_i_0_20.size();

    // Allocate space for output data
    h_o_0_176.resize(m_num_batches);

    // --- Setup OpenCL Buffers ---
    OCL_CHECK(err, d_i_0_20 = cl::Buffer(m_context,
                                         CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                         m_num_batches * sizeof(struct_ebu_7_t),
                                         h_i_0_20.data(), &err));
    OCL_CHECK(err, d_o_0_176 = cl::Buffer(
                       m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                       m_num_batches * sizeof(struct_sbu_22_t),
                       h_o_0_176.data(), &err));

    // Setup stop_flag buffer
    h_stop_flag.resize(1);
    OCL_CHECK(err, d_stop_flag = cl::Buffer(
                       m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(int), h_stop_flag.data(), &err));
}

void AlgorithmHost::transfer_data_to_fpga() {
    cl_int err;
    h_stop_flag[0] = 0; // Reset stop flag before each iteration
    OCL_CHECK(
        err, err = m_q.enqueueMigrateMemObjects(
                 {d_i_0_20, d_stop_flag}, 0 /* 0 means from host to device */));
}

void AlgorithmHost::execute_kernel_iteration(cl::Event &event) {
    cl_int err;
    int arg_idx = 0;
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_i_0_20));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_o_0_176));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_stop_flag));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, (uint16_t)m_num_batches));

    OCL_CHECK(err, err = m_q.enqueueTask(m_kernel, nullptr, &event));
}

void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects(
                       {d_o_0_176, d_stop_flag}, CL_MIGRATE_MEM_OBJECT_HOST));
}

bool AlgorithmHost::check_convergence_and_update() {
    bool changed = false;

    // Hardcoded Bellman-Ford update and convergence check logic
    std::map<int, ap_fixed<32, 16>> min_distances;

    // Collect minimum distances from kernel output
    for (const auto &batch : h_o_0_176) {
        for (int i = 0; i < batch.end_pos; ++i) {
            int node_id = batch.data[i].ele_1.id;
            ap_fixed<32, 16> dist = batch.data[i].ele_0;
            if (min_distances.find(node_id) == min_distances.end() ||
                dist < min_distances[node_id]) {
                min_distances[node_id] = dist;
            }
        }
    }

    // Update host-side distances and check for changes
    for (auto const &[node_id, new_dist] : min_distances) {
        if (new_dist < h_distances[node_id]) {
            h_distances[node_id] = new_dist;
            changed = true;
        }
    }

    // If changed, repack the input buffer for the next iteration
    if (changed) {
        // This is a simplified repack logic. It only updates the distances.
        for (auto &batch : h_i_0_20) {
            for (int i = 0; i < batch.end_pos; ++i) {
                batch.data[i].src.distance = h_distances[batch.data[i].src.id];
                batch.data[i].dst.distance = h_distances[batch.data[i].dst.id];
            }
        }
    }

    return !changed; // Return true if converged (no changes)
}

const std::vector<int> &AlgorithmHost::get_results() const {
    // This function now converts the final ap_fixed distances to int for
    // verification.
    static std::vector<int> final_distances;
    final_distances.clear();
    final_distances.reserve(h_distances.size());
    for (const auto &dist : h_distances) {
        if (dist > std::numeric_limits<int>::max()) {
            final_distances.push_back(std::numeric_limits<int>::max());
        } else {
            final_distances.push_back(dist.to_int());
        }
    }
    return final_distances;
}

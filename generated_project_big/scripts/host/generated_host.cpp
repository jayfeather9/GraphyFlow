#include "generated_host.h"
#include <iostream>
#include <map>
#include <limits>
#include <cstring> // For memcpy

// Constructor
AlgorithmHost::AlgorithmHost(cl::Context &context, cl::Kernel &kernel,
                             cl::CommandQueue &q)
    : m_context(context), m_kernel(kernel), m_q(q), m_num_vertices(0),
      m_num_batches(0) {}

// --- DYNAMICALLY GENERATED SECTIONS ---


// Helper function to convert ap_fixed to int32_t by reinterpreting its bits
static int32_t ap_fixed_to_int32(const ap_fixed<32, 16>& val) {
    return *reinterpret_cast<const int32_t*>(&val);
}



void AlgorithmHost::setup_buffers(const GraphCSR &graph, int start_node) {
    m_num_vertices = graph.num_vertices;
    cl_int err;
    
    h_distances.assign(m_num_vertices, INFINITY_DIST);
    if (start_node < m_num_vertices) { h_distances[start_node] = 0; }

    h_i_0_20.clear();
    struct_ebu_6_t current_batch;
    int edges_in_batch = 0;

    for (int u = 0; u < graph.num_vertices; ++u) {
        for (int i = graph.offsets[u]; i < graph.offsets[u + 1]; ++i) {
            int v = graph.columns[i];
            int w = graph.weights[i];

            edge_t edge; // This is now the POD version of edge_t
            edge.src.id = u;
            edge.src.distance = ap_fixed_to_int32(h_distances[u]);
            edge.dst.id = v;
            edge.dst.distance = ap_fixed_to_int32(h_distances[v]);
            edge.weight = ap_fixed_to_int32(ap_fixed<32, 16>(w));

            current_batch.data[edges_in_batch] = edge;
            edges_in_batch++;

            if (edges_in_batch == PE_NUM) {
                current_batch.end_pos = PE_NUM;
                current_batch.end_flag = false;
                h_i_0_20.push_back(current_batch);
                edges_in_batch = 0;
            }
        }
    }

    if (edges_in_batch > 0) {
        current_batch.end_pos = edges_in_batch;
        current_batch.end_flag = false;
        h_i_0_20.push_back(current_batch);
    }
    
    if (!h_i_0_20.empty()) {
        h_i_0_20.back().end_flag = true;
    }

    m_num_batches = h_i_0_20.size();
    h_o_0_176.resize(m_num_batches);
    h_stop_flag.resize(1);

    OCL_CHECK(err, d_i_0_20 = cl::Buffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, h_i_0_20.size() * sizeof(struct_ebu_6_t), h_i_0_20.data(), &err));
    OCL_CHECK(err, d_o_0_176 = cl::Buffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, h_o_0_176.size() * sizeof(KernelOutputBatch), h_o_0_176.data(), &err));
    OCL_CHECK(err, d_stop_flag = cl::Buffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int), h_stop_flag.data(), &err));
}



void AlgorithmHost::transfer_data_to_fpga() {
    cl_int err;
    h_stop_flag[0] = 0; // Reset stop flag before each iteration
    OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects({d_i_0_20, d_stop_flag}, 0));
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
    OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects({d_o_0_176, d_stop_flag}, CL_MIGRATE_MEM_OBJECT_HOST));
    m_q.finish(); // Ensure data is synced back to host
}



// --- LOGIC FOR RESULT PROCESSING (GENERIC FOR ITERATIVE ALGORITHMS) ---


bool AlgorithmHost::check_convergence_and_update() {
    bool changed = false;
    std::map<int, ap_fixed<32, 16>> min_distances;

    for (const auto& batch : h_o_0_176) {
        for (int i = 0; i < batch.end_pos; ++i) {
            int node_id = batch.data[i].id;
            ap_fixed<32, 16> dist = batch.data[i].distance;
            if (min_distances.find(node_id) == min_distances.end() || dist < min_distances[node_id]) {
                min_distances[node_id] = dist;
            }
        }
    }

    for (auto const &[node_id, new_dist] : min_distances) {
        if (new_dist < h_distances[node_id]) {
            h_distances[node_id] = new_dist;
            changed = true;
        }
    }

    if (changed) {
        for (auto& batch : h_i_0_20) {
            for (int i = 0; i < batch.end_pos; ++i) {
                batch.data[i].src.distance = ap_fixed_to_int32(h_distances[batch.data[i].src.id]);
                batch.data[i].dst.distance = ap_fixed_to_int32(h_distances[batch.data[i].dst.id]);
            }
        }
    }

    return !changed;
}

const std::vector<int> &AlgorithmHost::get_results() const {
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
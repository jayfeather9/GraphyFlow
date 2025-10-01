// --- REPLACE THE ENTIRE CONTENT OF THE FILE ---
#include "generated_host.h"
#include <cstring> // For memcpy
#include <iostream>
#include <limits>
#include <map>

// Constructor
AlgorithmHost::AlgorithmHost(cl::Context &context, cl::Kernel &kernel,
                             cl::CommandQueue &q)
    : m_context(context), m_kernel(kernel), m_q(q), m_num_vertices(0) {}

// Helper function to convert float to int32_t by reinterpreting its bits
// This aligns with ap_fixed<32,16> representation in the kernel.
static int32_t float_to_int32_bits(ap_fixed<32, 16> val) {
    return *reinterpret_cast<int32_t *>(&val);
}

// Helper function to convert int32_t bits back to float
static ap_fixed<32, 16> int32_bits_to_float(int32_t val) {
    return *reinterpret_cast<ap_fixed<32, 16> *>(&val);
}

void AlgorithmHost::setup_buffers(const GraphCSR &graph, int start_node) {
    m_num_vertices = graph.num_vertices;
    m_num_edges = graph.num_edges;
    cl_int err;

    // --- PHASE 1: Prepare host-side CSR buffers ---

    // 1.1 Source Offsets
    h_src_offsets.assign(graph.offsets.begin(), graph.offsets.end());

    // 1.2 Edge Descriptors (Destination and Weight)
    int bursts_num = (m_num_edges + PE_NUM - 1) / PE_NUM;
    h_edge_desc_bursts.resize(bursts_num);
    for (int i = 0; i < graph.num_edges; ++i) {
        h_edge_desc_bursts[i / PE_NUM].edges[i % PE_NUM].dst_id = graph.columns[i];
        // The kernel expects ap_fixed<32,16> stored as int32_t.
        // We can treat host-side weights as floats for this conversion.
        ap_fixed<32, 16> weight_fp =
            static_cast<ap_fixed<32, 16>>(graph.weights[i]);
        h_edge_desc_bursts[i / PE_NUM].edges[i % PE_NUM].weight = float_to_int32_bits(weight_fp);
    }

    // 1.3 Node Distances (initial state for Bellman-Ford)
    h_node_distances.assign(
        m_num_vertices,
        float_to_int32_bits(static_cast<ap_fixed<32, 16>>(INFINITY_DIST)));
    if (start_node < m_num_vertices) {
        h_node_distances[start_node] = float_to_int32_bits(0.0f);
    }

    // 1.4 Kernel output buffer
    // The number of output batches depends on the kernel's internal logic.
    // We estimate a sufficient size. A safe upper bound is num_vertices.
    size_t max_output_batches = (m_num_vertices + PE_NUM - 1) / PE_NUM;
    h_o_0_342.resize(max_output_batches);

    cl_mem_ext_ptr_t h_src_offsets_ext;
    h_src_offsets_ext.obj = h_src_offsets.data();
    h_src_offsets_ext.param = 0;
    h_src_offsets_ext.flags = (0 | XCL_MEM_TOPOLOGY);

    cl_mem_ext_ptr_t h_edge_desc_bursts_ext;
    h_edge_desc_bursts_ext.obj = h_edge_desc_bursts.data();
    h_edge_desc_bursts_ext.param = 0;
    h_edge_desc_bursts_ext.flags = (0 | XCL_MEM_TOPOLOGY);

    cl_mem_ext_ptr_t h_node_distances_ext;
    h_node_distances_ext.obj = h_node_distances.data();
    h_node_distances_ext.param = 0;
    h_node_distances_ext.flags = (0 | XCL_MEM_TOPOLOGY);

    cl_mem_ext_ptr_t h_o_0_342_ext;
    h_o_0_342_ext.obj = h_o_0_342.data();
    h_o_0_342_ext.param = 0;
    h_o_0_342_ext.flags = (0 | XCL_MEM_TOPOLOGY);

    // --- PHASE 2: Create OpenCL device buffers ---
    OCL_CHECK(
        err, d_src_offsets = cl::Buffer(
                 m_context,
                 CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                 h_src_offsets.size() * sizeof(int), &h_src_offsets_ext, &err));
    OCL_CHECK(
        err, d_edge_desc_bursts = cl::Buffer(
                 m_context,
                 CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                 h_edge_desc_bursts.size() * sizeof(edge_des_burst_t),
                 &h_edge_desc_bursts_ext, &err));
    OCL_CHECK(err, d_node_distances =
                       cl::Buffer(m_context,
                                  CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                      CL_MEM_USE_HOST_PTR,
                                  h_node_distances.size() * sizeof(int),
                                  &h_node_distances_ext, &err));
    OCL_CHECK(err, d_o_0_342 =
                       cl::Buffer(m_context,
                                  CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX |
                                      CL_MEM_USE_HOST_PTR,
                                  h_o_0_342.size() * sizeof(KernelOutputBatch),
                                  &h_o_0_342_ext, &err));
}

void AlgorithmHost::transfer_data_to_fpga() {
    cl_int err;
    // For Bellman-Ford, node_distances must be updated each iteration.
    // The other buffers are read-only and can be transferred just once if
    // desired, but transferring them all simplifies the logic.
    OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects(
                       {d_src_offsets, d_edge_desc_bursts, d_node_distances},
                       0 /* 0 means from host to device */));
}

void AlgorithmHost::execute_kernel_iteration(cl::Event &event) {
    cl_int err;
    int arg_idx = 0;

    // Set the new kernel arguments in the correct order
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_src_offsets));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_edge_desc_bursts));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_node_distances));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, m_num_vertices));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, m_num_edges));
    OCL_CHECK(err,
              err = m_kernel.setArg(arg_idx++, d_o_0_342)); // Output buffer

    OCL_CHECK(err, err = m_q.enqueueTask(m_kernel, nullptr, &event));
}

void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    // Migrate the results and the (potentially updated) distances back to the
    // host
    OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects(
                       {d_o_0_342}, CL_MIGRATE_MEM_OBJECT_HOST));
    m_q.finish(); // Ensure data is synced back to host
}

bool AlgorithmHost::check_convergence_and_update() {
    bool changed = false;
    // This map stores the minimum distance found for each node in this
    // iteration's output.
    std::map<int, float> min_distances;

    // --- PHASE 1: Process kernel output to find new minimum distances ---
    for (const auto &batch : h_o_0_342) {
        for (int i = 0; i < batch.end_pos; ++i) {
            int node_id = batch.data[i].id;
            float dist = batch.data[i].distance;

            if (min_distances.find(node_id) == min_distances.end() ||
                dist < min_distances[node_id]) {
                min_distances[node_id] = dist;
            }
        }
        if (batch.end_flag)
            break;
    }

    // --- PHASE 2: Compare with current distances and update if a shorter path
    // is found ---
    for (auto const &[node_id, new_dist_float] : min_distances) {
        float current_dist_float =
            int32_bits_to_float(h_node_distances[node_id]);

        if (new_dist_float < current_dist_float) {
            h_node_distances[node_id] = float_to_int32_bits(new_dist_float);
            changed = true;
        }
    }

    // The function returns 'true' if no values were changed, indicating
    // convergence.
    return !changed;
}

const std::vector<int> &AlgorithmHost::get_results() const {
    // The h_node_distances vector now holds the final results directly
    // in the integer format that the host verifier expects (after conversion).
    // We need to convert from our float-bit-representation back to integer
    // distances.
    static std::vector<int> final_distances;
    final_distances.clear();
    final_distances.reserve(h_node_distances.size());

    for (const auto &dist_bits : h_node_distances) {
        float dist_float = int32_bits_to_float(dist_bits);
        if (dist_float >= INFINITY_DIST) {
            final_distances.push_back(std::numeric_limits<int>::max());
        } else {
            final_distances.push_back(static_cast<int>(dist_float));
        }
    }
    return final_distances;
}
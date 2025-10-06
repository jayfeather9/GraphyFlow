// --- REPLACE THE ENTIRE CONTENT OF THE FILE ---
#include "generated_host.h"
#include <cstring> // For memcpy
#include <iostream>
#include <limits>
#include <map>

// Constructor
AlgorithmHost::AlgorithmHost(cl::Context &context,
                  cl::Kernel &kernel_glb,
                  std::vector<cl::Kernel> &kernels_graphyflow,
                  cl::CommandQueue &q,
                  cl::CommandQueue &q_glb,
                  std::vector<cl::CommandQueue> &q_graphyflow,
                  std::vector<GraphCSR> &graphs)
    : m_context(context), m_kernel_glb(kernel_glb), m_kernels_graphyflow(kernels_graphyflow),
      m_q(q), m_q_glb(q_glb), m_q_graphyflow(q_graphyflow), m_graphs(graphs) {}

// Helper function to convert float to int32_t by reinterpreting its bits
// This aligns with ap_fixed<32,16> representation in the kernel.
static int32_t float_to_int32_bits(ap_fixed<32, 16> val) {
    return *reinterpret_cast<int32_t *>(&val);
}

// Helper function to convert int32_t bits back to float
static ap_fixed<32, 16> int32_bits_to_float(int32_t val) {
    return *reinterpret_cast<ap_fixed<32, 16> *>(&val);
}

void AlgorithmHost::setup_buffers(int start_node) {
    cl_int err;
    printf("Setting up buffers for %zu partitions...\n", m_graphs.size());
    int src_off_base = 0 * NUM_PARTITIONS;
    int edge_desc_base = 1 * NUM_PARTITIONS;
    int node_dist_base = 2 * NUM_PARTITIONS;
    int output_base = 3 * NUM_PARTITIONS;
    
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        printf(" Setting up partition %d...\n", i);
        h_memory_buffers.emplace_back();
        d_memory_buffers.emplace_back();
        GraphCSR graph = m_graphs[i];
        int m_num_vertices = graph.num_vertices;
        int m_num_edges = graph.num_edges;

        // --- PHASE 1: Prepare host-side CSR buffers ---

        // 1.1 Source Offsets
        h_memory_buffers[i].h_src_offsets.assign(graph.offsets.begin(), graph.offsets.end());

        // 1.2 Edge Descriptors (Destination and Weight)
        int bursts_num = (m_num_edges + PE_NUM - 1) / PE_NUM;
        h_memory_buffers[i].h_edge_desc_bursts.resize(bursts_num);
        for (int edge_idx = 0; edge_idx < graph.num_edges; ++edge_idx) {
            h_memory_buffers[i].h_edge_desc_bursts[edge_idx / PE_NUM].edges[edge_idx % PE_NUM].dst_id = graph.columns[edge_idx];
            // The kernel expects ap_fixed<32,16> stored as int32_t.
            // We can treat host-side weights as floats for this conversion.
            ap_fixed<32, 16> weight_fp =
                static_cast<ap_fixed<32, 16>>(graph.weights[edge_idx]);
            h_memory_buffers[i].h_edge_desc_bursts[edge_idx / PE_NUM].edges[edge_idx % PE_NUM].weight = float_to_int32_bits(weight_fp);
        }

        // 1.3 Node Distances (initial state for Bellman-Ford)
        h_memory_buffers[i].h_node_distances.assign(
            m_num_vertices,
            float_to_int32_bits(static_cast<ap_fixed<32, 16>>(INFINITY_DIST)));
        // Check if ori idx for start_node is in the graph.vtx_map
        if (graph.vtx_map.find(start_node) != graph.vtx_map.end()) {
            int mapped_start_node = graph.vtx_map[start_node];
            h_memory_buffers[i].h_node_distances[mapped_start_node] = float_to_int32_bits(0.0f);
        }

        // 1.4 Kernel output buffer
        // The number of output batches depends on the kernel's internal logic.
        // We estimate a sufficient size. A safe upper bound is num_vertices.
        size_t max_output_batches = (m_num_vertices + PE_NUM - 1) / PE_NUM;
        h_memory_buffers[i].h_outputs.resize(max_output_batches);

        cl_mem_ext_ptr_t h_src_offsets_ext;
        h_src_offsets_ext.obj = h_memory_buffers[i].h_src_offsets.data();
        h_src_offsets_ext.param = 0;
        h_src_offsets_ext.flags = ((src_off_base + i) | XCL_MEM_TOPOLOGY);

        cl_mem_ext_ptr_t h_edge_desc_bursts_ext;
        h_edge_desc_bursts_ext.obj = h_memory_buffers[i].h_edge_desc_bursts.data();
        h_edge_desc_bursts_ext.param = 0;
        h_edge_desc_bursts_ext.flags = ((edge_desc_base + i) | XCL_MEM_TOPOLOGY);

        cl_mem_ext_ptr_t h_node_distances_ext;
        h_node_distances_ext.obj = h_memory_buffers[i].h_node_distances.data();
        h_node_distances_ext.param = 0;
        h_node_distances_ext.flags = ((node_dist_base + i) | XCL_MEM_TOPOLOGY);

        cl_mem_ext_ptr_t h_outputs_ext;
        h_outputs_ext.obj = h_memory_buffers[i].h_outputs.data();
        h_outputs_ext.param = 0;
        h_outputs_ext.flags = ((output_base + i) | XCL_MEM_TOPOLOGY);

        // --- PHASE 2: Create OpenCL device buffers ---
        OCL_CHECK(
            err, d_memory_buffers[i].d_src_offsets = cl::Buffer(
                    m_context,
                    CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                    h_memory_buffers[i].h_src_offsets.size() * sizeof(int), &h_src_offsets_ext, &err));
        OCL_CHECK(
            err, d_memory_buffers[i].d_edge_desc_bursts = cl::Buffer(
                    m_context,
                    CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                    h_memory_buffers[i].h_edge_desc_bursts.size() * sizeof(edge_des_burst_t),
                    &h_edge_desc_bursts_ext, &err));
        OCL_CHECK(err, d_memory_buffers[i].d_node_distances =
                        cl::Buffer(m_context,
                                    CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                        CL_MEM_USE_HOST_PTR,
                                    h_memory_buffers[i].h_node_distances.size() * sizeof(int),
                                    &h_node_distances_ext, &err));
        OCL_CHECK(err, d_memory_buffers[i].d_outputs =
                        cl::Buffer(m_context,
                                    CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX |
                                        CL_MEM_USE_HOST_PTR,
                                    h_memory_buffers[i].h_outputs.size() * sizeof(KernelOutputBatch),
                                    &h_outputs_ext, &err));
    }
}

void AlgorithmHost::transfer_data_to_fpga() {
    cl_int err;
    // For Bellman-Ford, node_distances must be updated each iteration.
    // The other buffers are read-only and can be transferred just once if
    // desired, but transferring them all simplifies the logic.
    // OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects(
    //                    {d_src_offsets, d_edge_desc_bursts, d_node_distances},
    //                    0 /* 0 means from host to device */));
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects(
                           {d_memory_buffers[i].d_src_offsets,
                            d_memory_buffers[i].d_edge_desc_bursts,
                            d_memory_buffers[i].d_node_distances},
                           0 /* 0 means from host to device */));
    }
    m_q.finish();
}

void AlgorithmHost::execute_kernel_iteration(cl::Event &event_glb, std::vector<cl::Event> &events_graphyflow) {
    cl_int err;
    int arg_idx = 0;

    // Set the new kernel arguments in the correct order
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_kernel_glb.setArg(arg_idx++, d_memory_buffers[i].d_src_offsets));
    }
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_kernel_glb.setArg(arg_idx++, d_memory_buffers[i].d_edge_desc_bursts));
    }
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_kernel_glb.setArg(arg_idx++, d_memory_buffers[i].d_node_distances));
    }
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_kernel_glb.setArg(arg_idx++, d_memory_buffers[i].d_outputs));
    }
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_kernel_glb.setArg(arg_idx++, m_graphs[i].num_vertices));
    }
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_kernel_glb.setArg(arg_idx++, m_graphs[i].num_edges));
    }

    OCL_CHECK(err, err = m_q_glb.enqueueTask(m_kernel_glb, NULL, &event_glb));
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_q_graphyflow[i].enqueueTask(m_kernels_graphyflow[i], NULL, &events_graphyflow[i]));
    }
}

void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    // Migrate the results and the (potentially updated) distances back to the
    // host
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects(
                           {d_memory_buffers[i].d_outputs},
                           CL_MIGRATE_MEM_OBJECT_HOST));
    }
    m_q.finish();
}

bool AlgorithmHost::check_convergence_and_update() {
    bool changed = false;
    // This map stores the minimum distance found for each original node ID in this
    // iteration's output.
    std::map<int, float> min_distances;

    // --- PHASE 1: Process all kernel outputs to find new minimum distances for original node IDs ---
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        const auto& graph = m_graphs[i];
        const auto& vtx_map_rev = graph.vtx_map_rev;
        const auto& kernel_output = h_memory_buffers[i].h_outputs;

        for (const auto &batch : kernel_output) {
            for (int j = 0; j < batch.end_pos; ++j) {
                int local_node_id = batch.data[j].id;
                float dist = batch.data[j].distance;

                // Convert local ID back to original ID
                int original_node_id = vtx_map_rev.at(local_node_id);

                if (min_distances.find(original_node_id) == min_distances.end() ||
                    dist < min_distances[original_node_id]) {
                    min_distances[original_node_id] = dist;
                }
            }
            if (batch.end_flag)
                break;
        }
    }

    // --- PHASE 2: Compare with current distances and update if a shorter path is found ---
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        auto& graph = m_graphs[i];
        auto& node_distances = h_memory_buffers[i].h_node_distances;

        for (auto const &[original_node_id, new_dist] : min_distances) {
            // Check if this original node ID exists in the current partition's graph
            if (graph.vtx_map.count(original_node_id)) {
                int local_node_id = graph.vtx_map[original_node_id];
                
                ap_fixed<32, 16> current_dist = int32_bits_to_float(node_distances[local_node_id]);
                ap_fixed<32, 16> new_dist_apf = static_cast<ap_fixed<32, 16>>(new_dist);

                if (new_dist_apf < current_dist) {
                    node_distances[local_node_id] = float_to_int32_bits(new_dist_apf);
                    changed = true;
                }
            }
        }
    }

    // The function returns 'true' if no values were changed, indicating convergence.
    return !changed;
}

const std::vector<int> &AlgorithmHost::get_results() const {
    // This map will store the final distance for each original node ID.
    std::map<int, ap_fixed<32, 16>> original_distances;
    int max_node_id = -1;

    // --- PHASE 1: Collect final distances from all partitions ---
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        const auto& graph = m_graphs[i];
        const auto& node_distances = h_memory_buffers[i].h_node_distances;

        for (size_t local_id = 0; local_id < node_distances.size(); ++local_id) {
            if (graph.vtx_map_rev.count(local_id)) {
                int original_id = graph.vtx_map_rev.at(local_id);
                ap_fixed<32, 16> dist = int32_bits_to_float(node_distances[local_id]);
                
                // Store the distance for the original node ID.
                // Since destination nodes do not overlap between partitions,
                // we don't need to check for minimums here.
                original_distances[original_id] = dist;

                if (original_id > max_node_id) {
                    max_node_id = original_id;
                }
            }
        }
    }

    // --- PHASE 2: Prepare the final flat vector of distances ---
    static std::vector<int> final_distances;
    final_distances.clear();
    if (max_node_id >= 0) {
        final_distances.resize(max_node_id + 1, std::numeric_limits<int>::max());
    }

    for (const auto& pair : original_distances) {
        int original_id = pair.first;
        ap_fixed<32, 16> dist_fp = pair.second;

        if (dist_fp >= INFINITY_DIST) {
            final_distances[original_id] = std::numeric_limits<int>::max();
        } else {
            final_distances[original_id] = static_cast<int>(dist_fp);
        }
    }

    return final_distances;
}
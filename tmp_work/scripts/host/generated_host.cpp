#include "generated_host.h"
#include <iostream>
#include <limits>
#include <map>
#include <vector>

// Custom ap_fixed type for host-side conversion, matching the kernel's new
// format.
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;
typedef ap_fixed<WEIGHT_BITWIDTH, WEIGHT_INTEGER_PART> weight_t;

AlgorithmHost::AlgorithmHost(AccDescriptor &acc) : acc(acc) {}

// --- PHASE 1: BUFFER SETUP ---
// MODIFIED: Buffer sizes are now calculated based on the number of 512-bit
// words required.
void AlgorithmHost::setup_buffers(const PartitionContainer &container,
                                  int start_node) {
    cl_int err;
    std::cout
        << "--- [Host] Phase 1: Setting up HBM buffers for all kernels ---"
        << std::endl;

    // 1.1: Initialize global distance vector on the host
    m_num_vertices = container.num_graph_vertices;
    h_distances.assign(m_num_vertices, ap_fixed<32, 16>(INFINITY_DIST));
    if (start_node < m_num_vertices) {
        h_distances[start_node] = 0;
    }

    // 1.2: Clear old buffer handles and resize host-side result vectors
    big_kernel_buffers.clear();
    little_kernel_buffers.clear();
    big_kernel_host_outputs.resize(container.SPs.size());
    little_kernel_host_outputs.resize(container.DPs.size());

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;

    // --- 1.3: Setup buffers for BIG kernels (Sparse Partitions) ---
    for (size_t i = 0; i < container.SPs.size(); ++i) {
        const auto &p_graph = container.SPs[i].partitioned_graph;
        KernelBuffers buffers;

        // Calculate buffer sizes in terms of 512-bit words
        size_t num_offset_words =
            ((p_graph.num_vertices + 1) * sizeof(int32_t) + bytes_per_word -
             1) /
            bytes_per_word;
        size_t bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
        size_t num_edge_words =
            (p_graph.num_edges * bits_per_edge + AXI_BUS_WIDTH - 1) /
            AXI_BUS_WIDTH;
        size_t num_dist_words =
            (p_graph.num_vertices * DISTANCE_BITWIDTH + AXI_BUS_WIDTH - 1) /
            AXI_BUS_WIDTH;
        size_t bits_per_output = NODE_ID_BITWIDTH + DISTANCE_BITWIDTH;
        size_t num_output_words =
            (p_graph.num_vertices * bits_per_output + AXI_BUS_WIDTH - 1) /
                AXI_BUS_WIDTH +
            1; // +1 for safety

        cl_mem_ext_ptr_t hbm_ext_in0, hbm_ext_in1, hbm_ext_in2, hbm_ext_out;
        hbm_ext_in0.flags = XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_input_id[i];
        hbm_ext_in0.obj = nullptr;
        hbm_ext_in0.param = 0;
        hbm_ext_in1.flags = XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_input_id[i];
        hbm_ext_in1.obj = nullptr;
        hbm_ext_in1.param = 0;
        hbm_ext_in2.flags = XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_input_id[i];
        hbm_ext_in2.obj = nullptr;
        hbm_ext_in2.param = 0;
        hbm_ext_out.flags = XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_output_id[i];
        hbm_ext_out.obj = nullptr;
        hbm_ext_out.param = 0;
        printf("[BIG] HBM channel IDs - in0: %d, in1: %d, in2: %d, out: %d\n",
               acc.big_kernel_hbm_input_id[i], acc.big_kernel_hbm_input_id[i],
               acc.big_kernel_hbm_input_id[i], acc.big_kernel_hbm_output_id[i]);
        fflush(NULL);

        OCL_CHECK(err,
                  buffers.src_offsets_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_offset_words * bytes_per_word, &hbm_ext_in0, &err));
        OCL_CHECK(err,
                  buffers.edge_props_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_edge_words * bytes_per_word, &hbm_ext_in1, &err));
        OCL_CHECK(err,
                  buffers.node_props_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_dist_words * bytes_per_word, &hbm_ext_in2, &err));
        printf("[BIG] Allocated buffers - offsets: %zu bytes, edges: %zu "
               "bytes, dists: %zu bytes\n",
               num_offset_words * bytes_per_word,
               num_edge_words * bytes_per_word,
               num_dist_words * bytes_per_word);
        fflush(NULL);

        big_kernel_host_outputs[i].resize(num_output_words);
        OCL_CHECK(err,
                  buffers.output_buf = cl::Buffer(
                      acc.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_output_words * bytes_per_word, &hbm_ext_out, &err));

        big_kernel_buffers.push_back(buffers);
    }

    // --- 1.4: Setup buffers for LITTLE kernels (Dense Partitions) ---
    for (size_t i = 0; i < container.DPs.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;
        KernelBuffers buffers;

        size_t num_offset_words =
            ((p_graph.num_vertices + 1) * sizeof(int32_t) + bytes_per_word -
             1) /
            bytes_per_word;
        size_t bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
        size_t num_edge_words =
            (p_graph.num_edges * bits_per_edge + AXI_BUS_WIDTH - 1) /
            AXI_BUS_WIDTH;
        size_t num_dist_words =
            (p_graph.num_vertices * DISTANCE_BITWIDTH + AXI_BUS_WIDTH - 1) /
            AXI_BUS_WIDTH;
        size_t bits_per_output = NODE_ID_BITWIDTH + DISTANCE_BITWIDTH;
        size_t num_output_words =
            (p_graph.num_vertices * bits_per_output + AXI_BUS_WIDTH - 1) /
                AXI_BUS_WIDTH +
            1;

        cl_mem_ext_ptr_t hbm_ext_in0, hbm_ext_in1, hbm_ext_in2, hbm_ext_out;
        hbm_ext_in0.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_input_id[i];
        hbm_ext_in0.obj = nullptr;
        hbm_ext_in0.param = 0;
        hbm_ext_in1.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_input_id[i];
        hbm_ext_in1.obj = nullptr;
        hbm_ext_in1.param = 0;
        hbm_ext_in2.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_input_id[i];
        hbm_ext_in2.obj = nullptr;
        hbm_ext_in2.param = 0;
        hbm_ext_out.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_output_id[i];
        hbm_ext_out.obj = nullptr;
        hbm_ext_out.param = 0;
        printf(
            "[LITTLE] HBM channel IDs - in0: %d, in1: %d, in2: %d, out: %d\n",
            acc.little_kernel_hbm_input_id[i],
            acc.little_kernel_hbm_input_id[i],
            acc.little_kernel_hbm_input_id[i],
            acc.little_kernel_hbm_output_id[i]);
        fflush(NULL);

        OCL_CHECK(err,
                  buffers.src_offsets_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_offset_words * bytes_per_word, &hbm_ext_in0, &err));
        OCL_CHECK(err,
                  buffers.edge_props_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_edge_words * bytes_per_word, &hbm_ext_in1, &err));
        OCL_CHECK(err,
                  buffers.node_props_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_dist_words * bytes_per_word, &hbm_ext_in2, &err));
        printf("[LITTLE] Allocated buffers - offsets: %zu bytes, edges: %zu "
               "bytes, dists: %zu bytes\n",
               num_offset_words * bytes_per_word,
               num_edge_words * bytes_per_word,
               num_dist_words * bytes_per_word);
        fflush(NULL);

        little_kernel_host_outputs[i].resize(num_output_words);
        OCL_CHECK(err,
                  buffers.output_buf = cl::Buffer(
                      acc.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_output_words * bytes_per_word, &hbm_ext_out, &err));

        little_kernel_buffers.push_back(buffers);
    }

    std::cout << "[SUCCESS] HBM buffers created for " << container.SPs.size()
              << " big and " << container.DPs.size() << " little kernels."
              << std::endl;
}

// --- PHASE 2: DATA TRANSFER TO FPGA ---
// REWRITTEN: Implements packing logic to convert host data into 512-bit words
// for the kernel.
void AlgorithmHost::transfer_data_to_fpga(const PartitionContainer &container) {
    cl_int err;
    std::cout << "--- [Host] Phase 2: Packing and transferring data to HBM ---"
              << std::endl;

    // 2.1: Transfer data for BIG kernels
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        const auto &p_graph = container.SPs[i].partitioned_graph;

        // Pack node distances (24-bit each)
        const int dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
        std::vector<bus_word_t> packed_node_props(
            (p_graph.num_vertices + dists_per_word - 1) / dists_per_word, 0);
        for (int j = 0; j < p_graph.num_vertices; ++j) {
            int global_id = p_graph.vtx_map_rev.at(j);
            distance_t dist_24b = h_distances[global_id];
            ap_uint<DISTANCE_BITWIDTH> dist_pod =
                *reinterpret_cast<ap_uint<DISTANCE_BITWIDTH> *>(&dist_24b);
            int word_idx = j / dists_per_word;
            int bit_offset = (j % dists_per_word) * DISTANCE_BITWIDTH;
            packed_node_props[word_idx].range(
                bit_offset + DISTANCE_BITWIDTH - 1, bit_offset) = dist_pod;
        }

        // Pack edge properties (node_id: 24-bit, weight: 24-bit => 48-bit per
        // edge)
        const int bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
        const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
        std::vector<bus_word_t> packed_edge_props(
            (p_graph.num_edges + edges_per_word - 1) / edges_per_word, 0);
        for (size_t j = 0; j < p_graph.num_edges; ++j) {
            ap_uint<NODE_ID_BITWIDTH> dest_id = p_graph.columns[j];
            weight_t weight_val = (float)p_graph.weights[j];
            ap_uint<WEIGHT_BITWIDTH> weight_pod =
                *reinterpret_cast<ap_uint<WEIGHT_BITWIDTH> *>(&weight_val);
            ap_uint<bits_per_edge> packed_edge;
            packed_edge.range(NODE_ID_BITWIDTH - 1, 0) = dest_id;
            packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH) = weight_pod;

            int word_idx = j / edges_per_word;
            int bit_offset = (j % edges_per_word) * bits_per_edge;
            packed_edge_props[word_idx].range(bit_offset + bits_per_edge - 1,
                                              bit_offset) = packed_edge;
        }

        // Pack offsets (32-bit each)
        const int offsets_per_word = AXI_BUS_WIDTH / 32;
        std::vector<bus_word_t> packed_offsets(
            ((p_graph.num_vertices + 1) + offsets_per_word - 1) /
                offsets_per_word,
            0);
        for (int j = 0; j < p_graph.num_vertices + 1; ++j) {
            int word_idx = j / offsets_per_word;
            int bit_offset = (j % offsets_per_word) * 32;
            packed_offsets[word_idx].range(bit_offset + 31, bit_offset) =
                p_graph.offsets[j];
            printf("[BIG]Packed offset %d into word %d at bit %d.\n",
                   p_graph.offsets[j], word_idx, bit_offset);
            fflush(NULL);
        }

        // Enqueue writes
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueWriteBuffer(
                           big_kernel_buffers[i].node_props_buf, CL_TRUE, 0,
                           packed_node_props.size() * sizeof(bus_word_t),
                           packed_node_props.data()));
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueWriteBuffer(
                           big_kernel_buffers[i].edge_props_buf, CL_TRUE, 0,
                           packed_edge_props.size() * sizeof(bus_word_t),
                           packed_edge_props.data()));
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueWriteBuffer(
                           big_kernel_buffers[i].src_offsets_buf, CL_TRUE, 0,
                           packed_offsets.size() * sizeof(bus_word_t),
                           packed_offsets.data()));
    }

    // 2.2: Transfer data for LITTLE kernels (FIXED: UNIFIED PACKING LOGIC)
    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;

        // Pack node distances (24-bit each)
        const int dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
        std::vector<bus_word_t> packed_node_props(
            (p_graph.num_vertices + dists_per_word - 1) / dists_per_word, 0);
        for (int j = 0; j < p_graph.num_vertices; ++j) {
            int global_id = p_graph.vtx_map_rev.at(j);
            distance_t dist_24b = h_distances[global_id];
            ap_uint<DISTANCE_BITWIDTH> dist_pod =
                *reinterpret_cast<ap_uint<DISTANCE_BITWIDTH> *>(&dist_24b);
            int word_idx = j / dists_per_word;
            int bit_offset = (j % dists_per_word) * DISTANCE_BITWIDTH;
            packed_node_props[word_idx].range(
                bit_offset + DISTANCE_BITWIDTH - 1, bit_offset) = dist_pod;
        }

        // Pack edge properties (node_id: 24-bit, weight: 24-bit => 48-bit per
        // edge)
        const int bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
        const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
        std::vector<bus_word_t> packed_edge_props(
            (p_graph.num_edges + edges_per_word - 1) / edges_per_word, 0);
        for (size_t j = 0; j < p_graph.num_edges; ++j) {
            ap_uint<NODE_ID_BITWIDTH> dest_id = p_graph.columns[j];
            weight_t weight_val = (float)p_graph.weights[j];
            ap_uint<WEIGHT_BITWIDTH> weight_pod =
                *reinterpret_cast<ap_uint<WEIGHT_BITWIDTH> *>(&weight_val);
            ap_uint<bits_per_edge> packed_edge;
            packed_edge.range(NODE_ID_BITWIDTH - 1, 0) = dest_id;
            packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH) = weight_pod;

            int word_idx = j / edges_per_word;
            int bit_offset = (j % edges_per_word) * bits_per_edge;
            packed_edge_props[word_idx].range(bit_offset + bits_per_edge - 1,
                                              bit_offset) = packed_edge;
        }

        // Pack offsets (32-bit each)
        const int offsets_per_word = AXI_BUS_WIDTH / 32;
        std::vector<bus_word_t> packed_offsets(
            ((p_graph.num_vertices + 1) + offsets_per_word - 1) /
                offsets_per_word,
            0);
        for (int j = 0; j < p_graph.num_vertices + 1; ++j) {
            int word_idx = j / offsets_per_word;
            int bit_offset = (j % offsets_per_word) * 32;
            packed_offsets[word_idx].range(bit_offset + 31, bit_offset) =
                p_graph.offsets[j];
            printf("[LITTLE]Packed offset %d into word %d at bit %d.\n",
                   p_graph.offsets[j], word_idx, bit_offset);
            fflush(NULL);
        }

        // Enqueue writes
        OCL_CHECK(err, err = acc.little_gs_queue[i].enqueueWriteBuffer(
                           little_kernel_buffers[i].node_props_buf, CL_TRUE, 0,
                           packed_node_props.size() * sizeof(bus_word_t),
                           packed_node_props.data()));
        OCL_CHECK(err, err = acc.little_gs_queue[i].enqueueWriteBuffer(
                           little_kernel_buffers[i].edge_props_buf, CL_TRUE, 0,
                           packed_edge_props.size() * sizeof(bus_word_t),
                           packed_edge_props.data()));
        OCL_CHECK(err, err = acc.little_gs_queue[i].enqueueWriteBuffer(
                           little_kernel_buffers[i].src_offsets_buf, CL_TRUE, 0,
                           packed_offsets.size() * sizeof(bus_word_t),
                           packed_offsets.data()));
    }

    // 2.3: Wait for all transfers to complete
    for (auto &q : acc.big_gs_queue)
        q.finish();
    for (auto &q : acc.little_gs_queue)
        q.finish();
    std::cout
        << "[SUCCESS] All data packed and transferred for current iteration."
        << std::endl;
}

// --- PHASE 3: KERNEL EXECUTION ---
// MODIFIED: Kernel arguments are updated to match the new kernel signature.
void AlgorithmHost::execute_kernel_iteration(
    const PartitionContainer &container,
    std::vector<cl::Event> &big_kernel_events,
    std::vector<cl::Event> &little_kernel_events) {
    cl_int err;
    std::cout << "--- [Host] Phase 3: Enqueuing kernel tasks ---" << std::endl;

    // 3.1: Enqueue BIG kernels
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        auto &kernel = acc.big_gs_krnls[i];
        auto &buffers = big_kernel_buffers[i];
        const auto &p_graph = container.SPs[i].partitioned_graph;

        int arg_idx = 0;
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.src_offsets_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.edge_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.node_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.output_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_vertices));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_edges));

        cl::Event *event_ptr = &big_kernel_events[i];
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueTask(kernel, nullptr,
                                                             event_ptr));
    }

    // 3.2: Enqueue LITTLE kernels
    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        auto &kernel = acc.little_gs_krnls[i];
        auto &buffers = little_kernel_buffers[i];
        const auto &p_graph = container.DPs[i].partitioned_graph;

        int arg_idx = 0;
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.src_offsets_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.edge_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.node_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.output_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_vertices));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_edges));

        cl::Event *event_ptr = &little_kernel_events[i];
        OCL_CHECK(err, err = acc.little_gs_queue[i].enqueueTask(kernel, nullptr,
                                                                event_ptr));
    }
    std::cout << "[SUCCESS] All kernel tasks enqueued for one iteration."
              << std::endl;
}

// --- PHASE 4: DATA TRANSFER FROM FPGA ---
void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    std::cout << "--- [Host] Phase 4: Transferring results from HBM ---"
              << std::endl;

    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        OCL_CHECK(err,
                  err = acc.big_gs_queue[i].enqueueReadBuffer(
                      big_kernel_buffers[i].output_buf, CL_FALSE, 0,
                      big_kernel_host_outputs[i].size() * sizeof(bus_word_t),
                      big_kernel_host_outputs[i].data()));
    }

    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        OCL_CHECK(err,
                  err = acc.little_gs_queue[i].enqueueReadBuffer(
                      little_kernel_buffers[i].output_buf, CL_FALSE, 0,
                      little_kernel_host_outputs[i].size() * sizeof(bus_word_t),
                      little_kernel_host_outputs[i].data()));
    }

    // Wait for all transfers to complete
    for (auto &q : acc.big_gs_queue)
        q.finish();
    for (auto &q : acc.little_gs_queue)
        q.finish();
    std::cout << "[SUCCESS] All results transferred from HBM." << std::endl;
}

// --- PHASE 5: CONVERGENCE CHECK AND GLOBAL STATE UPDATE ---
// REWRITTEN: Implements unpacking logic to parse results from 512-bit words.
bool AlgorithmHost::check_convergence_and_update(
    const PartitionContainer &container) {
    bool changed = false;
    std::cout << "--- [Host] Phase 5: Unpacking results and checking for "
                 "convergence ---"
              << std::endl;

    std::map<int, ap_fixed<32, 16>> min_distances;
    const int bits_per_output = NODE_ID_BITWIDTH + DISTANCE_BITWIDTH;
    const int outputs_per_word = AXI_BUS_WIDTH / bits_per_output;

    // 5.1: Unpack and gather results from BIG kernels
    for (size_t i = 0; i < big_kernel_host_outputs.size(); ++i) {
        const auto &p_graph = container.SPs[i].partitioned_graph;
        for (const auto &word : big_kernel_host_outputs[i]) {
            for (int k = 0; k < outputs_per_word; ++k) {
                int bit_offset = k * bits_per_output;
                ap_uint<bits_per_output> packed_output =
                    word.range(bit_offset + bits_per_output - 1, bit_offset);

                if (packed_output == 0)
                    continue; // Skip padding

                ap_uint<NODE_ID_BITWIDTH> local_id_pod =
                    packed_output.range(NODE_ID_BITWIDTH - 1, 0);
                ap_uint<DISTANCE_BITWIDTH> dist_pod =
                    packed_output.range(bits_per_output - 1, NODE_ID_BITWIDTH);

                int local_id = local_id_pod;
                if (p_graph.vtx_map_rev.count(local_id) == 0)
                    continue; // Invalid local ID

                int global_id = p_graph.vtx_map_rev.at(local_id);
                if (global_id >= m_num_vertices)
                    continue; // Skip if ID is out of bounds (padding)

                distance_t dist_24b =
                    *reinterpret_cast<distance_t *>(&dist_pod);
                ap_fixed<32, 16> new_dist =
                    dist_24b; // Widen for host-side master copy

                if (min_distances.find(global_id) == min_distances.end() ||
                    new_dist < min_distances[global_id]) {
                    min_distances[global_id] = new_dist;
                }
            }
        }
    }

    // 5.2: Unpack and gather results from LITTLE kernels (identical logic)
    for (size_t i = 0; i < little_kernel_host_outputs.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;
        for (const auto &word : little_kernel_host_outputs[i]) {
            for (int k = 0; k < outputs_per_word; ++k) {
                int bit_offset = k * bits_per_output;
                ap_uint<bits_per_output> packed_output =
                    word.range(bit_offset + bits_per_output - 1, bit_offset);

                if (packed_output == 0)
                    continue;

                ap_uint<NODE_ID_BITWIDTH> local_id_pod =
                    packed_output.range(NODE_ID_BITWIDTH - 1, 0);
                ap_uint<DISTANCE_BITWIDTH> dist_pod =
                    packed_output.range(bits_per_output - 1, NODE_ID_BITWIDTH);

                int local_id = local_id_pod;
                if (p_graph.vtx_map_rev.count(local_id) == 0)
                    continue;

                int global_id = p_graph.vtx_map_rev.at(local_id);
                if (global_id >= m_num_vertices)
                    continue;

                distance_t dist_24b =
                    *reinterpret_cast<distance_t *>(&dist_pod);
                ap_fixed<32, 16> new_dist = dist_24b;

                if (min_distances.find(global_id) == min_distances.end() ||
                    new_dist < min_distances[global_id]) {
                    min_distances[global_id] = new_dist;
                }
            }
        }
    }

    // 5.3: Update global distance vector and check for changes
    for (auto const &[global_id, new_dist] : min_distances) {
        if (global_id < m_num_vertices && new_dist < h_distances[global_id]) {
            h_distances[global_id] = new_dist;
            changed = true;
        }
    }

    if (changed) {
        std::cout << "[INFO] Distances updated. Preparing for next iteration."
                  << std::endl;
    } else {
        std::cout << "[INFO] No distance updates. Algorithm has converged."
                  << std::endl;
    }

    return !changed;
}

// --- FINALIZATION ---
const std::vector<int> &AlgorithmHost::get_results() const {
    static std::vector<int> final_distances;
    final_distances.clear();
    final_distances.reserve(h_distances.size());

    for (const auto &dist : h_distances) {
        if (dist >= INFINITY_DIST) {
            final_distances.push_back(std::numeric_limits<int>::max());
        } else {
            final_distances.push_back(dist.to_int());
        }
    }
    return final_distances;
}
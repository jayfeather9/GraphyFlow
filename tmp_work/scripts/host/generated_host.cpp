#include "generated_host.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

AlgorithmHost::AlgorithmHost(AccDescriptor &acc) : acc(acc) {}

void AlgorithmHost::prepare_data(const PartitionContainer &container,
                                 int start_node) {
    std::cout << "--- [Host] Phase 0: Preparing data structures ---"
              << std::endl;

    // 1. Initialize algorithm state
    m_num_vertices = container.num_graph_vertices;
    h_distances.assign(m_num_vertices, distance_t(INFINITY_DIST));
    if (start_node < m_num_vertices) {
        h_distances[start_node] = 0;
    }

    // 2. Prepare host-side input buffers for each kernel
    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;
    big_kernel_input_buffers.resize(container.SPs.size());
    little_kernel_input_buffers.resize(container.DPs.size());

    auto start_time = std::chrono::system_clock::now();
    auto current_time = start_time;

    // --- 2.1: 为 BIG kernels 手动序列化数据 (带 Padding) ---
    for (size_t i = 0; i < big_kernel_input_buffers.size(); ++i) {
        const auto &p_graph = container.SPs[i].partitioned_graph;

        // --- Pack node distances ---
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t word_number =
                (p_graph.num_vertices + dist_per_word - 1) / dist_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            for (int j = 0; j < p_graph.num_vertices; ++j) {
                // **Padding Logic**: 检查加上新数据后是否会跨越 64 字节边界
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed,
                                            0); // 插入0作为 padding
                }

                int global_id = p_graph.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];

                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            big_kernel_input_buffers[i].packed_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_kernel_input_buffers[i].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Preparing data structures big node dist ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;

        // --- Pack edge properties ---
        {
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH) / 8;
            const size_t edges_per_word = bytes_per_word / bytes_per_edge;
            const size_t word_number =
                (p_graph.num_edges + edges_per_word - 1) / edges_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            // Iterate through vertices, then their edges
            for (int v = 0; v < p_graph.num_vertices; ++v) {
                node_id_t src_id = v;
                for (int edge_idx = p_graph.offsets[v];
                     edge_idx < p_graph.offsets[v + 1]; ++edge_idx) {
                    // **Padding Logic**
                    if ((temp_byte_buffer.size() % bytes_per_word) +
                            bytes_per_edge >
                        bytes_per_word) {
                        size_t padding_needed =
                            bytes_per_word -
                            (temp_byte_buffer.size() % bytes_per_word);
                        temp_byte_buffer.insert(temp_byte_buffer.end(),
                                                padding_needed, 0);
                    }

                    char edge_bytes[bytes_per_edge];
                    uint32_t dest_id = p_graph.columns[edge_idx];

                    // Pack dst_id (first NODE_ID_BITWIDTH bits)
                    for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                        edge_bytes[b] = (dest_id >> (8 * b)) & 0xFF;
                    }

                    // Pack src_id (next NODE_ID_BITWIDTH bits)
                    for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                        edge_bytes[(NODE_ID_BITWIDTH / 8) + b] =
                            (src_id >> (8 * b)) & 0xFF;
                    }

                    temp_byte_buffer.insert(temp_byte_buffer.end(), edge_bytes,
                                            edge_bytes + bytes_per_edge);
                }
            }
            big_kernel_input_buffers[i].packed_edge_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_kernel_input_buffers[i].packed_edge_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Preparing data structures edge props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;
    }

    // --- 2.2: 为 LITTLE kernels 手动序列化数据 (带 Padding) ---
    for (size_t i = 0; i < little_kernel_input_buffers.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;

        // --- Pack node distances ---
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t word_number =
                (p_graph.num_vertices + dist_per_word - 1) / dist_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            for (int j = 0; j < p_graph.num_vertices; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = p_graph.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            little_kernel_input_buffers[i].packed_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(little_kernel_input_buffers[i].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Preparing data structures little node dist "
               "("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;

        // --- Pack edge properties ---
        {
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH) / 8;
            const size_t edges_per_word = bytes_per_word / bytes_per_edge;
            const size_t word_number =
                (p_graph.num_edges + edges_per_word - 1) / edges_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            // Iterate through vertices, then their edges
            for (int v = 0; v < p_graph.num_vertices; ++v) {
                node_id_t src_id = v;
                for (int edge_idx = p_graph.offsets[v];
                     edge_idx < p_graph.offsets[v + 1]; ++edge_idx) {
                    if ((temp_byte_buffer.size() % bytes_per_word) +
                            bytes_per_edge >
                        bytes_per_word) {
                        size_t padding_needed =
                            bytes_per_word -
                            (temp_byte_buffer.size() % bytes_per_word);
                        temp_byte_buffer.insert(temp_byte_buffer.end(),
                                                padding_needed, 0);
                    }

                    char edge_bytes[bytes_per_edge];
                    uint32_t dest_id = p_graph.columns[edge_idx];

                    // Pack dst_id (first NODE_ID_BITWIDTH bits)
                    for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                        edge_bytes[b] = (dest_id >> (8 * b)) & 0xFF;
                    }

                    // Pack src_id (next NODE_ID_BITWIDTH bits)
                    for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                        edge_bytes[(NODE_ID_BITWIDTH / 8) + b] =
                            (src_id >> (8 * b)) & 0xFF;
                    }

                    temp_byte_buffer.insert(temp_byte_buffer.end(), edge_bytes,
                                            edge_bytes + bytes_per_edge);
                }
            }
            little_kernel_input_buffers[i].packed_edge_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(little_kernel_input_buffers[i].packed_edge_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Preparing data structures little edge "
               "props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;
    }
}

// --- PHASE 1: BUFFER SETUP ---
// MODIFIED: Buffer sizes are now calculated based on the number of 512-bit
// words required.
void AlgorithmHost::setup_buffers(const PartitionContainer &container) {
    cl_int err;
    std::cout
        << "--- [Host] Phase 1: Setting up HBM buffers for all kernels ---"
        << std::endl;

    // 1.1: Initialize global distance vector on the host

    // 1.2: Clear old buffer handles and resize host-side result vectors
    big_kernel_buffers.clear();
    little_kernel_buffers.clear();
    little_writer_kernel_buffers.clear();
    big_writer_kernel_buffers.clear();
    apply_kernel_node_prop_buffers.clear();
    little_writer_host_outputs.resize(acc.num_little_krnl);
    big_writer_host_outputs.resize(acc.num_big_krnl);

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;

    // --- 1.3: Setup buffers for LITTLE kernels (Dense Partitions) ---
    for (size_t i = 0; i < container.DPs.size(); ++i) {
        printf("Setting up LITTLE kernel %zu buffers...\n", i);
        const auto &p_graph = container.DPs[i].partitioned_graph;
        KernelBuffers buffers;

        cl_mem_ext_ptr_t hbm_ext_edge;
        hbm_ext_edge.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_edge_id[i];
        hbm_ext_edge.obj =
            little_kernel_input_buffers[i].packed_edge_props.data();
        hbm_ext_edge.param = 0;

        // use pre-calculated sizes from Phase 0
        size_t num_edge_words =
            (little_kernel_input_buffers[i].packed_edge_props.size());
        OCL_CHECK(err, buffers.edge_props_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                          CL_MEM_USE_HOST_PTR,
                                      num_edge_words * bytes_per_word,
                                      &hbm_ext_edge, &err));

        // Calculate output buffer size: only distances (no node IDs or end
        // markers) Output is num_dst_vertices * DISTANCE_BITWIDTH
        size_t max_dst_local_id = 0;
        for (size_t e = 0; e < p_graph.num_edges; ++e) {
            int dst = p_graph.columns[e];
            if (((dst & 0x40000000) == 0) && dst > max_dst_local_id) {
                max_dst_local_id = dst;
            }
        }
        size_t num_dst_vertices = max_dst_local_id + 1;
        // Calculate how many distances fit in one 512-bit word
        size_t dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
        size_t num_output_words =
            (num_dst_vertices + dists_per_word - 1) / dists_per_word;

        little_kernel_buffers.push_back(buffers);

        // Create buffers for corresponding hbm_writer kernel
        WriterKernelBuffers writer_buffers;

        // Setup HBM extension for writer kernel input (node distances)
        cl_mem_ext_ptr_t hbm_ext_writer_node;
        hbm_ext_writer_node.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_node_id[i];
        hbm_ext_writer_node.obj =
            little_kernel_input_buffers[i].packed_node_props.data();
        hbm_ext_writer_node.param = 0;
        cl_mem_ext_ptr_t hbm_ext_writer_out;
        hbm_ext_writer_out.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_node_id[i];
        hbm_ext_writer_out.obj = nullptr;
        hbm_ext_writer_out.param = 0;
        // apply_kernel_node_prop_buffers
        cl_mem_ext_ptr_t hbm_ext_apply_node;
        hbm_ext_apply_node.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_node_id[i];
        hbm_ext_apply_node.obj =
            little_kernel_input_buffers[i].packed_node_props.data();
        hbm_ext_apply_node.param = 0;

        // Create node_props_buf for writer kernel (same size as node_props)
        size_t num_dist_words =
            (little_kernel_input_buffers[i].packed_node_props.size());
        OCL_CHECK(err, writer_buffers.node_props_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                          CL_MEM_USE_HOST_PTR,
                                      num_dist_words * bytes_per_word,
                                      &hbm_ext_writer_node, &err));
        // apply_kernel_node_prop_buffers
        OCL_CHECK(
            err,
            apply_kernel_node_prop_buffers.push_back(cl::Buffer(
                acc.context,
                CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                num_dist_words * bytes_per_word, &hbm_ext_apply_node, &err)));
        // Create output buffer for writer kernel
        little_writer_host_outputs[i].resize(num_output_words);
        OCL_CHECK(err, writer_buffers.output_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                                      num_output_words * bytes_per_word,
                                      &hbm_ext_writer_out, &err));
        little_writer_kernel_buffers.push_back(writer_buffers);
    }

    // --- 1.4: Setup buffers for BIG kernels (Sparse Partitions) ---
    for (size_t i = 0; i < container.SPs.size(); ++i) {
        const auto &p_graph = container.SPs[i].partitioned_graph;
        KernelBuffers buffers;

        cl_mem_ext_ptr_t hbm_ext_edge;
        hbm_ext_edge.flags = XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_edge_id[i];
        hbm_ext_edge.obj = big_kernel_input_buffers[i].packed_edge_props.data();
        hbm_ext_edge.param = 0;

        // use pre-calculated sizes from Phase 0
        size_t num_edge_words =
            (big_kernel_input_buffers[i].packed_edge_props.size());
        OCL_CHECK(err, buffers.edge_props_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                          CL_MEM_USE_HOST_PTR,
                                      num_edge_words * bytes_per_word,
                                      &hbm_ext_edge, &err));

        // Calculate output buffer size: only distances (no node IDs or end
        // markers) Output is num_dst_vertices * DISTANCE_BITWIDTH
        size_t max_dst_local_id = 0;
        for (size_t e = 0; e < p_graph.num_edges; ++e) {
            int dst = p_graph.columns[e];
            if (((dst & 0x40000000) == 0) && dst > max_dst_local_id) {
                max_dst_local_id = dst;
            }
        }
        size_t num_dst_vertices = max_dst_local_id + 1;
        // Calculate how many distances fit in one 512-bit word
        size_t dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
        size_t num_output_words =
            (num_dst_vertices + dists_per_word - 1) / dists_per_word;

        big_kernel_buffers.push_back(buffers);

        // Create buffers for corresponding hbm_writer kernel
        WriterKernelBuffers writer_buffers;

        // Setup HBM extension for writer kernel input (node distances)
        cl_mem_ext_ptr_t hbm_ext_writer_node;
        hbm_ext_writer_node.flags =
            XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_node_id[i];
        hbm_ext_writer_node.obj =
            big_kernel_input_buffers[i].packed_node_props.data();
        hbm_ext_writer_node.param = 0;
        cl_mem_ext_ptr_t hbm_ext_writer_out;
        hbm_ext_writer_out.flags =
            XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_node_id[i];
        hbm_ext_writer_out.obj = nullptr;
        hbm_ext_writer_out.param = 0;
        // apply_kernel_node_prop_buffers
        cl_mem_ext_ptr_t hbm_ext_apply_node;
        hbm_ext_apply_node.flags =
            XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_node_id[i];
        hbm_ext_apply_node.obj =
            big_kernel_input_buffers[i].packed_node_props.data();
        hbm_ext_apply_node.param = 0;

        // Create node_props_buf for writer kernel (same size as node_props)
        size_t num_dist_words =
            (big_kernel_input_buffers[i].packed_node_props.size());
        OCL_CHECK(err, writer_buffers.node_props_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                          CL_MEM_USE_HOST_PTR,
                                      num_dist_words * bytes_per_word,
                                      &hbm_ext_writer_node, &err));
        // apply_kernel_node_prop_buffers
        OCL_CHECK(
            err,
            apply_kernel_node_prop_buffers.push_back(cl::Buffer(
                acc.context,
                CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                num_dist_words * bytes_per_word, &hbm_ext_apply_node, &err)));
        // Create output buffer for writer kernel
        big_writer_host_outputs[i].resize(num_output_words);
        OCL_CHECK(err, writer_buffers.output_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                                      num_output_words * bytes_per_word,
                                      &hbm_ext_writer_out, &err));
        big_writer_kernel_buffers.push_back(writer_buffers);
    }

    std::cout << "[SUCCESS] HBM buffers created for " << container.SPs.size()
              << " big and " << container.DPs.size() << " little kernels."
              << std::endl;
}

void AlgorithmHost::update_data(const PartitionContainer &container) {
    std::cout
        << "--- [Host] Phase 2.1: Updating host-side data for new iteration ---"
        << std::endl;

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;

    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        const auto &p_graph = container.SPs[i].partitioned_graph;

        // --- Pack node distances ---
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            std::vector<char> temp_byte_buffer;

            for (int j = 0; j < p_graph.num_vertices; ++j) {
                // **Padding Logic**: 检查加上新数据后是否会跨越 64 字节边界
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed,
                                            0); // 插入0作为 padding
                }

                int global_id = p_graph.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];

                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            big_kernel_input_buffers[i].packed_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_kernel_input_buffers[i].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }
    }

    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;

        // --- Pack node distances ---
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            std::vector<char> temp_byte_buffer;

            for (int j = 0; j < p_graph.num_vertices; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = p_graph.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            little_kernel_input_buffers[i].packed_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(little_kernel_input_buffers[i].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }
    }

    std::cout << "[SUCCESS] Host-side data updated for new iteration."
              << std::endl;
}

void AlgorithmHost::transfer_data_to_fpga(const PartitionContainer &container) {
    cl_int err;
    std::cout << "--- [Host] Phase 2: Packing and transferring data to HBM ---"
              << std::endl;

    // --- 2.3: Transfer all packed data using enqueueMigrateMemObjects ---
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueMigrateMemObjects(
                           {big_kernel_buffers[i].edge_props_buf},
                           0 /* 0 means from host*/));
        OCL_CHECK(err,
                  err = acc.hbm_writer_big_queue[i].enqueueMigrateMemObjects(
                      {big_writer_kernel_buffers[i].node_props_buf},
                      0 /* 0 means from host*/));
        // apply_kernel_node_prop_buffers
        OCL_CHECK(
            err,
            err = acc.apply_queue[i + acc.num_little_krnl]
                      .enqueueMigrateMemObjects(
                          {apply_kernel_node_prop_buffers[i +
                                                          acc.num_little_krnl]},
                          0 /* 0 means from host*/));
    }

    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        OCL_CHECK(err, err = acc.little_gs_queue[i].enqueueMigrateMemObjects(
                           {little_kernel_buffers[i].edge_props_buf},
                           0 /* 0 means from host*/));
        OCL_CHECK(err,
                  err = acc.hbm_writer_little_queue[i].enqueueMigrateMemObjects(
                      {little_writer_kernel_buffers[i].node_props_buf},
                      0 /* 0 means from host*/));
        // apply_kernel_node_prop_buffers
        OCL_CHECK(err, err = acc.apply_queue[i].enqueueMigrateMemObjects(
                           {apply_kernel_node_prop_buffers[i]},
                           0 /* 0 means from host*/));
    }

    // --- 2.6: Wait for all transfers to complete ---
    for (auto &q : acc.big_gs_queue)
        q.finish();
    for (auto &q : acc.little_gs_queue)
        q.finish();
    for (auto &q : acc.apply_queue)
        q.finish();
    for (auto &q : acc.hbm_writer_big_queue)
        q.finish();
    for (auto &q : acc.hbm_writer_little_queue)
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
    std::vector<cl::Event> &little_kernel_events,
    std::vector<cl::Event> &apply_kernel_events,
    std::vector<cl::Event> &little_writer_events,
    std::vector<cl::Event> &big_writer_events) {
    cl_int err;
    // std::cout << "--- [Host] Phase 3: Enqueuing kernel tasks ---" <<
    // std::endl;

    auto enqueue_start = std::chrono::high_resolution_clock::now();

    // 3.2: Enqueue LITTLE HBM writer kernels
    for (size_t i = 0; i < little_writer_kernel_buffers.size(); ++i) {
        auto &writer_kernel = acc.hbm_writer_little_krnls[i];
        const auto &p_graph = container.DPs[i].partitioned_graph;

        int arg_idx = 0;
        OCL_CHECK(err, err = writer_kernel.setArg(
                           arg_idx++,
                           little_writer_kernel_buffers[i].node_props_buf));
        OCL_CHECK(err,
                  err = writer_kernel.setArg(
                      arg_idx++, little_writer_kernel_buffers[i].output_buf));
        OCL_CHECK(err, err = writer_kernel.setArg(arg_idx++, p_graph.num_dsts));
        OCL_CHECK(err, err = acc.hbm_writer_little_queue[i].enqueueTask(
                           writer_kernel, nullptr, &little_writer_events[i]));
    }

    // 3.3: Enqueue BIG HBM writer kernels
    for (size_t i = 0; i < big_writer_kernel_buffers.size(); ++i) {
        auto &writer_kernel = acc.hbm_writer_big_krnls[i];
        const auto &p_graph = container.SPs[i].partitioned_graph;

        int arg_idx = 0;
        OCL_CHECK(err,
                  err = writer_kernel.setArg(
                      arg_idx++, big_writer_kernel_buffers[i].node_props_buf));
        OCL_CHECK(err, err = writer_kernel.setArg(
                           arg_idx++, big_writer_kernel_buffers[i].output_buf));
        OCL_CHECK(err, err = writer_kernel.setArg(arg_idx++, p_graph.num_dsts));
        OCL_CHECK(err, err = acc.hbm_writer_big_queue[i].enqueueTask(
                           writer_kernel, nullptr, &big_writer_events[i]));
    }

    // Enqueue apply kernel
    for (size_t i = 0; i < acc.apply_krnls.size(); ++i) {
        auto &apply_kernel = acc.apply_krnls[i];
        cl::Event &apply_kernel_event = apply_kernel_events[i];
        const auto &p_graph =
            (i < acc.num_little_krnl)
                ? container.DPs[i].partitioned_graph
                : container.SPs[i - acc.num_little_krnl].partitioned_graph;
        int apply_arg_idx = 0;
        OCL_CHECK(err, err = apply_kernel.setArg(
                           apply_arg_idx++, apply_kernel_node_prop_buffers[i]));
        OCL_CHECK(err,
                  err = apply_kernel.setArg(apply_arg_idx++, p_graph.num_dsts));
        OCL_CHECK(err, err = acc.apply_queue[i].enqueueTask(
                           apply_kernel, nullptr, &apply_kernel_event));
    }

    // 3.1: Enqueue BIG kernels (no output parameter, streams to hbm_writer)
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        auto &kernel = acc.big_gs_krnls[i];
        auto &buffers = big_kernel_buffers[i];
        const auto &p_graph = container.SPs[i].partitioned_graph;

        int arg_idx = 0;
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.edge_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_vertices));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_edges));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_dsts));

        cl::Event *event_ptr = &big_kernel_events[i];
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueTask(kernel, nullptr,
                                                             event_ptr));
    }

    // 3.3: Enqueue LITTLE kernels
    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        auto &kernel = acc.little_gs_krnls[i];
        auto &buffers = little_kernel_buffers[i];
        const auto &p_graph = container.DPs[i].partitioned_graph;

        int arg_idx = 0;
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.edge_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_vertices));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_edges));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_dsts));

        cl::Event *event_ptr = &little_kernel_events[i];
        OCL_CHECK(err, err = acc.little_gs_queue[i].enqueueTask(kernel, nullptr,
                                                                event_ptr));
    }

    auto enqueue_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> enqueue_time = enqueue_end - enqueue_start;
    std::cout << "[SUCCESS] All kernel tasks enqueued for one iteration (Time: "
              << enqueue_time.count() << " seconds)" << std::endl;
}

// --- PHASE 4: DATA TRANSFER FROM FPGA ---
void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    std::cout << "--- [Host] Phase 4: Transferring results from HBM ---"
              << std::endl;

    auto transfer_start = std::chrono::high_resolution_clock::now();

    // Read from hbm_writer kernel output buffers instead of big kernel buffers
    for (size_t i = 0; i < little_writer_kernel_buffers.size(); ++i) {
        OCL_CHECK(err,
                  err = acc.hbm_writer_little_queue[i].enqueueReadBuffer(
                      little_writer_kernel_buffers[i].output_buf, CL_FALSE, 0,
                      little_writer_host_outputs[i].size() * sizeof(bus_word_t),
                      little_writer_host_outputs[i].data()));
    }
    for (size_t i = 0; i < big_writer_kernel_buffers.size(); ++i) {
        OCL_CHECK(err,
                  err = acc.hbm_writer_big_queue[i].enqueueReadBuffer(
                      big_writer_kernel_buffers[i].output_buf, CL_FALSE, 0,
                      big_writer_host_outputs[i].size() * sizeof(bus_word_t),
                      big_writer_host_outputs[i].data()));
    }

    // Wait for all transfers to complete
    for (auto &q : acc.hbm_writer_little_queue)
        q.finish();
    for (auto &q : acc.hbm_writer_big_queue)
        q.finish();

    auto transfer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> transfer_time = transfer_end - transfer_start;
    std::cout << "[SUCCESS] All results transferred from HBM (Time: "
              << transfer_time.count() << " seconds)" << std::endl;
}

// --- PHASE 5: CONVERGENCE CHECK AND GLOBAL STATE UPDATE ---
// REWRITTEN: Implements unpacking logic to parse results from 512-bit words.
bool AlgorithmHost::check_convergence_and_update(
    const PartitionContainer &container) {
    bool changed = false;
    std::cout << "--- [Host] Phase 5: Unpacking results and checking for "
                 "convergence ---"
              << std::endl;

    std::map<int, distance_t> min_distances;
    const int dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;

    // 5.1: Unpack and gather results from writer kernels
    // Node IDs are implicit: they are sequential from 0 to num_dsts-1
    auto process_outputs = [&](const auto &p_graph, const auto &words) {
        int local_id = 0; // Implicit node ID counter

        for (const auto &word : words) {
            for (int k = 0; k < dists_per_word && local_id < p_graph.num_dsts;
                 ++k, ++local_id) {
                int bit_offset = k * DISTANCE_BITWIDTH;
                ap_fixed_pod_t dist_pod =
                    word.range(bit_offset + DISTANCE_BITWIDTH - 1, bit_offset);

                if (p_graph.vtx_map_rev.count(local_id) == 0) {
                    continue; // Invalid local ID
                }

                int global_id = p_graph.vtx_map_rev.at(local_id);
                if (global_id >= m_num_vertices) {
                    continue; // Skip if ID is out of bounds
                }

                distance_t new_dist =
                    *reinterpret_cast<distance_t *>(&dist_pod);

                if (min_distances.find(global_id) == min_distances.end() ||
                    new_dist < min_distances[global_id]) {
                    min_distances[global_id] = new_dist;
                }
            }

            if (local_id >= p_graph.num_dsts) {
                break; // All outputs processed
            }
        }
    };

    for (size_t i = 0; i < little_writer_host_outputs.size(); ++i) {
        process_outputs(container.DPs[i].partitioned_graph,
                        little_writer_host_outputs[i]);
    }

    for (size_t i = 0; i < big_writer_host_outputs.size(); ++i) {
        process_outputs(container.SPs[i].partitioned_graph,
                        big_writer_host_outputs[i]);
    }

    // 5.2: Update global distance vector and check for changes
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
            final_distances.push_back(INFINITY_DIST);
        } else {
            final_distances.push_back(dist.to_int());
        }
    }
    return final_distances;
}
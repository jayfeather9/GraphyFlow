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

    // 2. Prepare host-side input buffers for each pipeline
    // Now we have 1 little partition with LITTLE_KERNEL_NUM pipelines
    // and 1 big partition with BIG_KERNEL_NUM pipelines
    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;
    little_kernel_input_buffers.resize(LITTLE_KERNEL_NUM);
    big_kernel_input_buffers.resize(BIG_KERNEL_NUM);
    std::vector<bus_word_t, aligned_allocator<bus_word_t>> big_dst_node_props,
        little_dst_node_props;

    auto start_time = std::chrono::system_clock::now();
    auto current_time = start_time;

    // --- 2.1: Prepare BIG partition data (shared node props, separate edge
    // props per pipeline) ---
    if (!container.SPs.empty()) {
        const auto &big_partition = container.SPs[0];

        // Pack node distances ONCE for the big partition (shared across all big
        // pipelines)
        std::vector<bus_word_t, aligned_allocator<bus_word_t>>
            shared_big_node_props;
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t word_number =
                (big_partition.num_vertices + dist_per_word - 1) /
                dist_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            for (int j = 0; j < big_partition.num_vertices; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }

                int global_id = big_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];

                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            shared_big_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(shared_big_node_props.data(), temp_byte_buffer.data(),
                        temp_byte_buffer.size());

            const size_t dst_word_number =
                (big_partition.num_dsts + dist_per_word - 1) / dist_per_word;
            temp_byte_buffer.clear();
            temp_byte_buffer.reserve(dst_word_number * bytes_per_word);
            for (int j = 0; j < big_partition.num_dsts; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = big_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            big_dst_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_dst_node_props.data(), temp_byte_buffer.data(),
                        temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Prepared shared big node props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;

        // Pack edge properties for EACH big pipeline
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            const auto &pipeline_edges = big_partition.pipeline_edges[pip];

            // Copy shared node props to this pipeline's buffer
            big_kernel_input_buffers[pip].packed_node_props.resize(
                shared_big_node_props.size());
            std::memcpy(big_kernel_input_buffers[pip].packed_node_props.data(),
                        shared_big_node_props.data(),
                        shared_big_node_props.size() * sizeof(bus_word_t));

            // Pack this pipeline's edge properties
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH) / 8;
            const size_t edges_per_word = bytes_per_word / bytes_per_edge;
            const size_t word_number =
                (pipeline_edges.num_edges + edges_per_word - 1) /
                edges_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            // Iterate through vertices, then their edges
            for (int v = 0; v < big_partition.num_vertices; ++v) {
                node_id_t src_id = v;
                for (int edge_idx = pipeline_edges.offsets[v];
                     edge_idx < pipeline_edges.offsets[v + 1]; ++edge_idx) {
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
                    uint32_t dest_id = pipeline_edges.columns[edge_idx];

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
            big_kernel_input_buffers[pip].packed_edge_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_kernel_input_buffers[pip].packed_edge_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Prepared " << BIG_KERNEL_NUM
            << " big pipeline edge props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;
    }

    // --- 2.2: Prepare LITTLE partition data (shared node props, separate edge
    // props per pipeline) ---
    if (!container.DPs.empty()) {
        const auto &little_partition = container.DPs[0];

        // Pack node distances ONCE for the little partition (shared across all
        // little pipelines)
        std::vector<bus_word_t, aligned_allocator<bus_word_t>>
            shared_little_node_props;
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t word_number =
                (little_partition.num_vertices + dist_per_word - 1) /
                dist_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            for (int j = 0; j < little_partition.num_vertices; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = little_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            shared_little_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(shared_little_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());

            const size_t dst_word_number =
                (little_partition.num_dsts + dist_per_word - 1) / dist_per_word;
            temp_byte_buffer.clear();
            temp_byte_buffer.reserve(dst_word_number * bytes_per_word);
            for (int j = 0; j < little_partition.num_dsts; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = little_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            little_dst_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(little_dst_node_props.data(), temp_byte_buffer.data(),
                        temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Prepared shared little node props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;

        // Pack edge properties for EACH little pipeline
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            const auto &pipeline_edges = little_partition.pipeline_edges[pip];

            // Copy shared node props to this pipeline's buffer
            little_kernel_input_buffers[pip].packed_node_props.resize(
                shared_little_node_props.size());
            std::memcpy(
                little_kernel_input_buffers[pip].packed_node_props.data(),
                shared_little_node_props.data(),
                shared_little_node_props.size() * sizeof(bus_word_t));

            // Pack this pipeline's edge properties
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH) / 8;
            const size_t edges_per_word = bytes_per_word / bytes_per_edge;
            const size_t word_number =
                (pipeline_edges.num_edges + edges_per_word - 1) /
                edges_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            // Iterate through vertices, then their edges
            for (int v = 0; v < little_partition.num_vertices; ++v) {
                node_id_t src_id = v;
                for (int edge_idx = pipeline_edges.offsets[v];
                     edge_idx < pipeline_edges.offsets[v + 1]; ++edge_idx) {
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
                    uint32_t dest_id = pipeline_edges.columns[edge_idx];

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
            little_kernel_input_buffers[pip].packed_edge_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(
                little_kernel_input_buffers[pip].packed_edge_props.data(),
                temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Prepared " << LITTLE_KERNEL_NUM
            << " little pipeline edge props ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;
    }
    apply_kernel_node_props.resize(little_dst_node_props.size() +
                                   big_dst_node_props.size());
    std::memcpy(apply_kernel_node_props.data(), little_dst_node_props.data(),
                little_dst_node_props.size() * sizeof(bus_word_t));
    std::memcpy(apply_kernel_node_props.data() + little_dst_node_props.size(),
                big_dst_node_props.data(),
                big_dst_node_props.size() * sizeof(bus_word_t));
    current_time = std::chrono::system_clock::now();
    std::cout
        << "--- [Host] Phase 0: Prepared apply kernel node props ("
        << std::chrono::duration<double>(current_time - start_time).count()
        << " sec) ---" << std::endl;
}

// --- PHASE 1: BUFFER SETUP ---
// MODIFIED: Create separate edge buffers for each pipeline, but share node
// buffers within partition
void AlgorithmHost::setup_buffers(const PartitionContainer &container) {
    cl_int err;
    std::cout
        << "--- [Host] Phase 1: Setting up HBM buffers for all pipelines ---"
        << std::endl;

    // 1.1: Clear old buffer handles and resize host-side result vectors
    big_kernel_buffers.clear();
    little_kernel_buffers.clear();
    writer_kernel_node_prop_buffers.clear();

    // Output buffer sizing: little_dst_num + big_dst_num
    size_t little_dst_num =
        container.DPs.empty() ? 0 : container.DPs[0].num_dsts;
    size_t big_dst_num = container.SPs.empty() ? 0 : container.SPs[0].num_dsts;
    size_t total_dst_num = little_dst_num + big_dst_num;

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;
    size_t dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
    size_t big_dst_words = (big_dst_num + dists_per_word - 1) / dists_per_word;
    size_t little_dst_words =
        (little_dst_num + dists_per_word - 1) / dists_per_word;
    size_t total_output_words = big_dst_words + little_dst_words;

    // --- 1.2: Setup buffers for LITTLE pipelines ---
    if (!container.DPs.empty()) {
        const auto &little_partition = container.DPs[0];

        // Create edge buffers for each little pipeline
        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            KernelBuffers buffers;

            cl_mem_ext_ptr_t hbm_ext_edge;
            hbm_ext_edge.flags =
                XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_edge_id[pip];
            hbm_ext_edge.obj =
                little_kernel_input_buffers[pip].packed_edge_props.data();
            hbm_ext_edge.param = 0;

            size_t num_edge_words =
                little_kernel_input_buffers[pip].packed_edge_props.size();
            OCL_CHECK(err, buffers.edge_props_buf = cl::Buffer(
                               acc.context,
                               CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                   CL_MEM_USE_HOST_PTR,
                               num_edge_words * bytes_per_word, &hbm_ext_edge,
                               &err));

            little_kernel_buffers.push_back(buffers);
        }

        std::cout << "  Created " << LITTLE_KERNEL_NUM
                  << " little pipeline edge buffers." << std::endl;
    }

    // --- 1.3: Setup buffers for BIG pipelines ---
    if (!container.SPs.empty()) {
        const auto &big_partition = container.SPs[0];

        // Create edge buffers for each big pipeline
        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            KernelBuffers buffers;

            cl_mem_ext_ptr_t hbm_ext_edge;
            hbm_ext_edge.flags =
                XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_edge_id[pip];
            hbm_ext_edge.obj =
                big_kernel_input_buffers[pip].packed_edge_props.data();
            hbm_ext_edge.param = 0;

            size_t num_edge_words =
                big_kernel_input_buffers[pip].packed_edge_props.size();
            OCL_CHECK(err, buffers.edge_props_buf = cl::Buffer(
                               acc.context,
                               CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                   CL_MEM_USE_HOST_PTR,
                               num_edge_words * bytes_per_word, &hbm_ext_edge,
                               &err));

            big_kernel_buffers.push_back(buffers);
        }

        std::cout << "  Created " << BIG_KERNEL_NUM
                  << " big pipeline edge buffers." << std::endl;
    }

    // --- 1.4: Setup shared node property buffers for hbm_writer (14 total: 11
    // little + 3 big) --- Create 11 little node prop buffers for hbm_writer
    for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
        cl_mem_ext_ptr_t hbm_ext_node;
        hbm_ext_node.flags =
            XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_node_id[pip];
        hbm_ext_node.obj =
            little_kernel_input_buffers[pip].packed_node_props.data();
        hbm_ext_node.param = 0;

        size_t num_node_words =
            little_kernel_input_buffers[pip].packed_node_props.size();
        cl::Buffer node_buf;
        OCL_CHECK(err, node_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                          CL_MEM_USE_HOST_PTR,
                                      num_node_words * bytes_per_word,
                                      &hbm_ext_node, &err));
        writer_kernel_node_prop_buffers.push_back(node_buf);
    }

    // Create 3 big node prop buffers for hbm_writer
    for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
        cl_mem_ext_ptr_t hbm_ext_node;
        hbm_ext_node.flags = XCL_MEM_TOPOLOGY | acc.big_kernel_hbm_node_id[pip];
        hbm_ext_node.obj =
            big_kernel_input_buffers[pip].packed_node_props.data();
        hbm_ext_node.param = 0;

        size_t num_node_words =
            big_kernel_input_buffers[pip].packed_node_props.size();
        cl::Buffer node_buf;
        OCL_CHECK(err, node_buf =
                           cl::Buffer(acc.context,
                                      CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                          CL_MEM_USE_HOST_PTR,
                                      num_node_words * bytes_per_word,
                                      &hbm_ext_node, &err));
        writer_kernel_node_prop_buffers.push_back(node_buf);
    }

    // --- 1.5: Setup unified output buffer ---
    cl_mem_ext_ptr_t hbm_ext_output;
    hbm_ext_output.flags =
        XCL_MEM_TOPOLOGY |
        acc.little_kernel_hbm_node_id[0]; // Use first HBM bank
    hbm_ext_output.obj = nullptr;
    hbm_ext_output.param = 0;

    writer_kernel_host_outputs.resize(total_output_words, 0);
    OCL_CHECK(err,
              writer_kernel_output_buffer = cl::Buffer(
                  acc.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                  total_output_words * bytes_per_word, &hbm_ext_output, &err));

    // --- 1.6: Setup apply_kernel node prop buffer ---
    cl_mem_ext_ptr_t hbm_ext_apply;
    hbm_ext_apply.flags = XCL_MEM_TOPOLOGY | 30;
    hbm_ext_apply.obj = apply_kernel_node_props.data();
    hbm_ext_apply.param = 0;

    size_t apply_node_words =
        little_kernel_input_buffers[0].packed_node_props.size();
    OCL_CHECK(err, apply_kernel_node_prop_buffer =
                       cl::Buffer(acc.context,
                                  CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX |
                                      CL_MEM_USE_HOST_PTR,
                                  apply_node_words * bytes_per_word,
                                  &hbm_ext_apply, &err));

    std::cout << "[SUCCESS] HBM buffers created: " << LITTLE_KERNEL_NUM
              << " little + " << BIG_KERNEL_NUM << " big pipelines, "
              << "total output size: " << total_output_words << " words."
              << std::endl;
}

void AlgorithmHost::update_data(const PartitionContainer &container) {
    std::cout
        << "--- [Host] Phase 2.1: Updating host-side data for new iteration ---"
        << std::endl;

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;
    std::vector<bus_word_t, aligned_allocator<bus_word_t>> big_dst_node_props,
        little_dst_node_props;

    // Update BIG partition node distances (shared across all big pipelines)
    if (!container.SPs.empty()) {
        const auto &big_partition = container.SPs[0];
        const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
        std::vector<char> temp_byte_buffer;

        for (int j = 0; j < big_partition.num_vertices; ++j) {
            if ((temp_byte_buffer.size() % bytes_per_word) + bytes_per_dist >
                bytes_per_word) {
                size_t padding_needed =
                    bytes_per_word - (temp_byte_buffer.size() % bytes_per_word);
                temp_byte_buffer.insert(temp_byte_buffer.end(), padding_needed,
                                        0);
            }

            int global_id = big_partition.vtx_map_rev.at(j);
            distance_t dist_val = h_distances[global_id];

            const char *data_ptr = reinterpret_cast<const char *>(&dist_val);
            temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                    data_ptr + bytes_per_dist);
        }

        // Update all big pipeline buffers with same node data
        size_t num_words =
            (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word;

        for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
            big_kernel_input_buffers[pip].packed_node_props.resize(num_words,
                                                                   0);
            std::memcpy(big_kernel_input_buffers[pip].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        // update big_dst_node_props for apply kernel
        {
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t dst_word_number =
                (big_partition.num_dsts + dist_per_word - 1) / dist_per_word;
            temp_byte_buffer.clear();
            temp_byte_buffer.reserve(dst_word_number * bytes_per_word);
            for (int j = 0; j < big_partition.num_dsts; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = big_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            big_dst_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_dst_node_props.data(), temp_byte_buffer.data(),
                        temp_byte_buffer.size());
        }
    }

    // Update LITTLE partition node distances (shared across all little
    // pipelines)
    if (!container.DPs.empty()) {
        const auto &little_partition = container.DPs[0];
        const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
        std::vector<char> temp_byte_buffer;

        for (int j = 0; j < little_partition.num_vertices; ++j) {
            if ((temp_byte_buffer.size() % bytes_per_word) + bytes_per_dist >
                bytes_per_word) {
                size_t padding_needed =
                    bytes_per_word - (temp_byte_buffer.size() % bytes_per_word);
                temp_byte_buffer.insert(temp_byte_buffer.end(), padding_needed,
                                        0);
            }
            int global_id = little_partition.vtx_map_rev.at(j);
            distance_t dist_val = h_distances[global_id];
            const char *data_ptr = reinterpret_cast<const char *>(&dist_val);
            temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                    data_ptr + bytes_per_dist);
        }

        // Update all little pipeline buffers with same node data
        size_t num_words =
            (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word;

        for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
            little_kernel_input_buffers[pip].packed_node_props.resize(num_words,
                                                                      0);
            std::memcpy(
                little_kernel_input_buffers[pip].packed_node_props.data(),
                temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        // update little_dst_node_props for apply kernel
        {
            const size_t dist_per_word = bytes_per_word / bytes_per_dist;
            const size_t dst_word_number =
                (little_partition.num_dsts + dist_per_word - 1) / dist_per_word;
            temp_byte_buffer.clear();
            temp_byte_buffer.reserve(dst_word_number * bytes_per_word);
            for (int j = 0; j < little_partition.num_dsts; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_dist >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int global_id = little_partition.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            little_dst_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(little_dst_node_props.data(), temp_byte_buffer.data(),
                        temp_byte_buffer.size());
        }
    }

    apply_kernel_node_props.resize(little_dst_node_props.size() +
                                   big_dst_node_props.size());
    std::memcpy(apply_kernel_node_props.data(), little_dst_node_props.data(),
                little_dst_node_props.size() * sizeof(bus_word_t));
    std::memcpy(apply_kernel_node_props.data() + little_dst_node_props.size(),
                big_dst_node_props.data(),
                big_dst_node_props.size() * sizeof(bus_word_t));

    std::cout << "[SUCCESS] Host-side data updated for new iteration."
              << std::endl;
}

void AlgorithmHost::transfer_data_to_fpga(const PartitionContainer &container) {
    cl_int err;
    std::cout << "--- [Host] Phase 2.2: Transferring data to FPGA HBM ---"
              << std::endl;

    // Transfer edge buffers for all big pipelines
    for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
        OCL_CHECK(err, err = acc.big_gs_queue[pip].enqueueMigrateMemObjects(
                           {big_kernel_buffers[pip].edge_props_buf},
                           0 /* 0 means from host*/));
    }

    // Transfer edge buffers for all little pipelines
    for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
        OCL_CHECK(err, err = acc.little_gs_queue[pip].enqueueMigrateMemObjects(
                           {little_kernel_buffers[pip].edge_props_buf},
                           0 /* 0 means from host*/));
    }

    // Transfer apply_kernel node buffer
    OCL_CHECK(err,
              err = acc.apply_queue.enqueueMigrateMemObjects(
                  {apply_kernel_node_prop_buffer}, 0 /* 0 means from host*/));

    // Transfer hbm_writer node prop buffers
    for (size_t i = 0; i < writer_kernel_node_prop_buffers.size(); ++i) {
        OCL_CHECK(err, err = acc.hbm_writer_queue.enqueueMigrateMemObjects(
                           {writer_kernel_node_prop_buffers[i]},
                           0 /* 0 means from host*/));
    }

    // Wait for all transfers to complete
    for (auto &q : acc.big_gs_queue)
        q.finish();
    for (auto &q : acc.little_gs_queue)
        q.finish();
    acc.apply_queue.finish();
    acc.hbm_writer_queue.finish();

    std::cout
        << "[SUCCESS] All data packed and transferred for current iteration."
        << std::endl;
}

// --- PHASE 3: KERNEL EXECUTION ---
// MODIFIED: Kernel arguments are updated to match the new kernel signature.
void AlgorithmHost::execute_kernel_iteration(
    const PartitionContainer &container) {
    cl_int err;
    // std::cout << "--- [Host] Phase 3: Enqueuing kernel tasks ---" <<
    // std::endl;

    auto enqueue_start = std::chrono::high_resolution_clock::now();

    // Get partition metadata first
    uint32_t little_num_vertices =
        container.DPs.empty() ? 0 : container.DPs[0].num_vertices;
    uint32_t little_num_dsts =
        container.DPs.empty() ? 0 : container.DPs[0].num_dsts;
    uint32_t big_num_vertices =
        container.SPs.empty() ? 0 : container.SPs[0].num_vertices;
    uint32_t big_num_dsts =
        container.SPs.empty() ? 0 : container.SPs[0].num_dsts;
    uint32_t little_dst_offset = 0;
    uint32_t byte_per_word = AXI_BUS_WIDTH / 8;
    uint32_t dists_per_word = AXI_BUS_WIDTH / DISTANCE_BITWIDTH;
    uint32_t little_dst_word_num =
        (little_num_dsts + dists_per_word - 1) / dists_per_word;
    uint32_t big_dst_offset = little_dst_word_num;

    // 3.1: Enqueue BIG gs kernels (one per pipeline, each with different edges,
    // same nodes)
    for (int pip = 0; pip < BIG_KERNEL_NUM; ++pip) {
        auto &kernel = acc.big_gs_krnls[pip];
        auto &buffers = big_kernel_buffers[pip];
        uint32_t pip_num_edges =
            container.SPs.empty()
                ? 0
                : container.SPs[0].pipeline_edges[pip].num_edges;

        int arg_idx = 0;
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.edge_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, big_num_vertices));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, pip_num_edges));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, big_num_dsts));

        OCL_CHECK(err, err = acc.big_gs_queue[pip].enqueueTask(
                           kernel, nullptr, &acc.big_kernel_events[pip]));
    }

    // 3.2: Enqueue LITTLE gs kernels (one per pipeline, each with different
    // edges, same nodes)
    for (int pip = 0; pip < LITTLE_KERNEL_NUM; ++pip) {
        auto &kernel = acc.little_gs_krnls[pip];
        auto &buffers = little_kernel_buffers[pip];
        uint32_t pip_num_edges =
            container.DPs.empty()
                ? 0
                : container.DPs[0].pipeline_edges[pip].num_edges;

        int arg_idx = 0;
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.edge_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, little_num_vertices));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, pip_num_edges));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, little_num_dsts));

        OCL_CHECK(err, err = acc.little_gs_queue[pip].enqueueTask(
                           kernel, nullptr, &acc.little_kernel_events[pip]));
    }

    // 3.3: Enqueue apply_kernel (receives merged streams from little_merger and
    // big_merger)
    {
        auto &apply_kernel = acc.apply_krnl;

        int arg_idx = 0;
        OCL_CHECK(err, err = apply_kernel.setArg(
                           arg_idx++, apply_kernel_node_prop_buffer));
        OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, little_num_dsts));
        OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, big_num_dsts));
        OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, little_dst_offset));
        OCL_CHECK(err, err = apply_kernel.setArg(arg_idx++, big_dst_offset));

        OCL_CHECK(err, err = acc.apply_queue.enqueueTask(
                           apply_kernel, nullptr, &acc.apply_kernel_event));
    }

    // Enqueue hbm_writer kernel
    {
        auto &writer_kernel = acc.hbm_writer_krnl;

        int arg_idx = 0;
        // writer_kernel_node_prop_buffers
        for (const auto &buffer : writer_kernel_node_prop_buffers) {
            OCL_CHECK(err, err = writer_kernel.setArg(arg_idx++, buffer));
        }
        OCL_CHECK(err, err = writer_kernel.setArg(arg_idx++,
                                                  writer_kernel_output_buffer));

        OCL_CHECK(err, err = acc.hbm_writer_queue.enqueueTask(
                           writer_kernel, nullptr, &acc.hbm_writer_event));
    }

    auto enqueue_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> enqueue_time = enqueue_end - enqueue_start;
    std::cout << "[SUCCESS] All kernel tasks enqueued for one iteration (Time: "
              << enqueue_time.count() << " seconds)" << std::endl;
}

// --- PHASE 4: DATA TRANSFER FROM FPGA ---
void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    std::cout << "--- [Host] Phase 4: Transferring results from FPGA ---"
              << std::endl;

    auto transfer_start = std::chrono::high_resolution_clock::now();

    // Read from the single unified output buffer
    OCL_CHECK(err, err = acc.hbm_writer_queue.enqueueReadBuffer(
                       writer_kernel_output_buffer, CL_FALSE, 0,
                       writer_kernel_host_outputs.size() * sizeof(bus_word_t),
                       writer_kernel_host_outputs.data()));

    // Wait for all transfers to complete
    acc.hbm_writer_queue.finish();

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

    // Output layout: [little_dsts][big_dsts]
    // First little_dst_num entries are little partition destinations
    // Next big_dst_num entries are big partition destinations

    int word_idx = 0;
    int dist_in_word = 0;

    // Process little partition destinations [0:little_dst_num]
    if (!container.DPs.empty()) {
        const auto &little_partition = container.DPs[0];
        for (int local_id = 0; local_id < little_partition.num_dsts;
             ++local_id) {
            int bit_offset = dist_in_word * DISTANCE_BITWIDTH;
            ap_fixed_pod_t dist_pod =
                writer_kernel_host_outputs[word_idx].range(
                    bit_offset + DISTANCE_BITWIDTH - 1, bit_offset);

            if (little_partition.vtx_map_rev.count(local_id)) {
                int global_id = little_partition.vtx_map_rev.at(local_id);
                if (global_id < m_num_vertices) {
                    distance_t new_dist =
                        *reinterpret_cast<distance_t *>(&dist_pod);

                    if (min_distances.find(global_id) == min_distances.end() ||
                        new_dist < min_distances[global_id]) {
                        min_distances[global_id] = new_dist;
                    }
                }
            }

            dist_in_word++;
            if (dist_in_word >= dists_per_word) {
                dist_in_word = 0;
                word_idx++;
            }
        }
    }

    if (dist_in_word != 0) {
        dist_in_word = 0;
        word_idx++;
    }

    // Process big partition destinations
    // [little_dst_num:little_dst_num+big_dst_num]
    if (!container.SPs.empty()) {
        const auto &big_partition = container.SPs[0];
        for (int local_id = 0; local_id < big_partition.num_dsts; ++local_id) {
            int bit_offset = dist_in_word * DISTANCE_BITWIDTH;
            ap_fixed_pod_t dist_pod =
                writer_kernel_host_outputs[word_idx].range(
                    bit_offset + DISTANCE_BITWIDTH - 1, bit_offset);

            if (big_partition.vtx_map_rev.count(local_id)) {
                int global_id = big_partition.vtx_map_rev.at(local_id);
                if (global_id < m_num_vertices) {
                    distance_t new_dist =
                        *reinterpret_cast<distance_t *>(&dist_pod);

                    if (min_distances.find(global_id) == min_distances.end() ||
                        new_dist < min_distances[global_id]) {
                        min_distances[global_id] = new_dist;
                    }
                }
            }

            dist_in_word++;
            if (dist_in_word >= dists_per_word) {
                dist_in_word = 0;
                word_idx++;
            }
        }
    }

    // Update global distance vector and check for changes
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
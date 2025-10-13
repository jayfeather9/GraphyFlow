#include "generated_host.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

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
    h_distances.assign(m_num_vertices, distance_t(INFINITY_DIST));
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
        // printf("[BIG] Allocated buffers - offsets: %zu bytes, edges: %zu "
        //        "bytes, dists: %zu bytes\n",
        //        num_offset_words * bytes_per_word,
        //        num_edge_words * bytes_per_word,
        //        num_dist_words * bytes_per_word);
        // fflush(NULL);

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

void AlgorithmHost::transfer_data_to_fpga(const PartitionContainer &container) {
    cl_int err;
    std::cout << "--- [Host] Phase 2: Packing and transferring data to HBM ---"
              << std::endl;

    // 为了保证内存生命周期，在外层定义好所有需要传输的数据容器
    std::vector<std::vector<bus_word_t, aligned_allocator<bus_word_t>>>
        big_packed_node_props(big_kernel_buffers.size());
    std::vector<std::vector<bus_word_t, aligned_allocator<bus_word_t>>>
        big_packed_edge_props(big_kernel_buffers.size());
    std::vector<std::vector<bus_word_t, aligned_allocator<bus_word_t>>>
        big_packed_offsets(big_kernel_buffers.size());

    std::vector<std::vector<bus_word_t, aligned_allocator<bus_word_t>>>
        little_packed_node_props(little_kernel_buffers.size());
    std::vector<std::vector<bus_word_t, aligned_allocator<bus_word_t>>>
        little_packed_edge_props(little_kernel_buffers.size());
    std::vector<std::vector<bus_word_t, aligned_allocator<bus_word_t>>>
        little_packed_offsets(little_kernel_buffers.size());

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;

    // --- 2.1: 为 BIG kernels 手动序列化数据 (带 Padding) ---
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        const auto &p_graph = container.SPs[i].partitioned_graph;

        // --- Pack node distances (ap_fixed<24,8> -> 3 bytes) ---
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

                printf("[BIG]Packed node %d with distance %f\n", global_id,
                       (float)dist_val);
                fflush(nullptr);
            }
            big_packed_node_props[i].resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_packed_node_props[i].data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        // --- Pack edge properties (node_id<24b> + weight<24b> -> 6 bytes) ---
        {
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + WEIGHT_BITWIDTH) / 8;
            std::vector<char> temp_byte_buffer;

            for (size_t j = 0; j < p_graph.num_edges; ++j) {
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
                uint32_t dest_id = p_graph.columns[j];
                edge_bytes[0] = (dest_id >> 0) & 0xFF;
                edge_bytes[1] = (dest_id >> 8) & 0xFF;
                edge_bytes[2] = (dest_id >> 16) & 0xFF;

                weight_t weight_val = (float)p_graph.weights[j];
                std::memcpy(edge_bytes + 3, &weight_val, (WEIGHT_BITWIDTH / 8));

                temp_byte_buffer.insert(temp_byte_buffer.end(), edge_bytes,
                                        edge_bytes + bytes_per_edge);

                // 调试日志
                uint32_t src_global_id = 0;
                uint32_t dst_global_id =
                    p_graph.vtx_map_rev.at(p_graph.columns[j]);
                for (int v = 0; v < p_graph.num_vertices; ++v) {
                    if (p_graph.offsets[v] <= j && j < p_graph.offsets[v + 1]) {
                        src_global_id = p_graph.vtx_map_rev.at(v);
                        break;
                    }
                }
                printf("[BIG]Packed edge: src=%d, dst=%d, weight=%f\n",
                       src_global_id, dst_global_id, (float)weight_val);
                fflush(nullptr);
            }
            big_packed_edge_props[i].resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_packed_edge_props[i].data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        // --- Pack offsets (int32_t -> 4 bytes) ---
        {
            const size_t bytes_per_offset = sizeof(int32_t);
            std::vector<char> temp_byte_buffer;

            for (int j = 0; j < p_graph.num_vertices + 1; ++j) {
                // **Padding Logic**
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_offset >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }

                int32_t offset_val = p_graph.offsets[j];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&offset_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_offset);
            }
            big_packed_offsets[i].resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_packed_offsets[i].data(), temp_byte_buffer.data(),
                        temp_byte_buffer.size());
        }
    }

    // --- 2.2: 为 LITTLE kernels 手动序列化数据 (带 Padding) ---
    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;

        // --- Pack node distances (ap_fixed<24,8> -> 3 bytes) ---
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
                printf("[LITTLE]Packed node %d with distance %f\n", global_id,
                       (float)dist_val);
                fflush(nullptr);
            }
            little_packed_node_props[i].resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(little_packed_node_props[i].data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        // --- Pack edge properties (node_id<24b> + weight<24b> -> 6 bytes) ---
        {
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + WEIGHT_BITWIDTH) / 8;
            std::vector<char> temp_byte_buffer;

            for (size_t j = 0; j < p_graph.num_edges; ++j) {
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
                uint32_t dest_id = p_graph.columns[j];
                edge_bytes[0] = (dest_id >> 0) & 0xFF;
                edge_bytes[1] = (dest_id >> 8) & 0xFF;
                edge_bytes[2] = (dest_id >> 16) & 0xFF;
                weight_t weight_val = (float)p_graph.weights[j];
                std::memcpy(edge_bytes + 3, &weight_val, (WEIGHT_BITWIDTH / 8));
                temp_byte_buffer.insert(temp_byte_buffer.end(), edge_bytes,
                                        edge_bytes + bytes_per_edge);

                uint32_t src_global_id = 0;
                uint32_t dst_global_id =
                    p_graph.vtx_map_rev.at(p_graph.columns[j]);
                for (int v = 0; v < p_graph.num_vertices; ++v) {
                    if (p_graph.offsets[v] <= j && j < p_graph.offsets[v + 1]) {
                        src_global_id = p_graph.vtx_map_rev.at(v);
                        break;
                    }
                }
                printf("[LITTLE]Packed edge: src=%d, dst=%d, weight=%f\n",
                       src_global_id, dst_global_id, (float)weight_val);
                fflush(nullptr);
            }
            little_packed_edge_props[i].resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(little_packed_edge_props[i].data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        // --- Pack offsets (int32_t -> 4 bytes) ---
        {
            const size_t bytes_per_offset = sizeof(int32_t);
            std::vector<char> temp_byte_buffer;

            for (int j = 0; j < p_graph.num_vertices + 1; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) +
                        bytes_per_offset >
                    bytes_per_word) {
                    size_t padding_needed =
                        bytes_per_word -
                        (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(),
                                            padding_needed, 0);
                }
                int32_t offset_val = p_graph.offsets[j];
                const char *data_ptr =
                    reinterpret_cast<const char *>(&offset_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_offset);
            }
            little_packed_offsets[i].resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(little_packed_offsets[i].data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }
    }

    // --- 2.3: 将所有打包好的数据加入传输队列 ---
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueWriteBuffer(
                           big_kernel_buffers[i].node_props_buf, CL_FALSE, 0,
                           big_packed_node_props[i].size() * sizeof(bus_word_t),
                           big_packed_node_props[i].data()));
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueWriteBuffer(
                           big_kernel_buffers[i].edge_props_buf, CL_FALSE, 0,
                           big_packed_edge_props[i].size() * sizeof(bus_word_t),
                           big_packed_edge_props[i].data()));
        OCL_CHECK(err, err = acc.big_gs_queue[i].enqueueWriteBuffer(
                           big_kernel_buffers[i].src_offsets_buf, CL_FALSE, 0,
                           big_packed_offsets[i].size() * sizeof(bus_word_t),
                           big_packed_offsets[i].data()));
    }

    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        OCL_CHECK(err,
                  err = acc.little_gs_queue[i].enqueueWriteBuffer(
                      little_kernel_buffers[i].node_props_buf, CL_FALSE, 0,
                      little_packed_node_props[i].size() * sizeof(bus_word_t),
                      little_packed_node_props[i].data()));
        OCL_CHECK(err,
                  err = acc.little_gs_queue[i].enqueueWriteBuffer(
                      little_kernel_buffers[i].edge_props_buf, CL_FALSE, 0,
                      little_packed_edge_props[i].size() * sizeof(bus_word_t),
                      little_packed_edge_props[i].data()));
        OCL_CHECK(err,
                  err = acc.little_gs_queue[i].enqueueWriteBuffer(
                      little_kernel_buffers[i].src_offsets_buf, CL_FALSE, 0,
                      little_packed_offsets[i].size() * sizeof(bus_word_t),
                      little_packed_offsets[i].data()));
    }

    // --- 2.4: 在所有命令入队后，执行一次全局同步 ---
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

    std::map<int, distance_t> min_distances;
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

                // TODO: maybe buggy
                if (packed_output == 0)
                    continue; // Skip padding

                ap_uint<NODE_ID_BITWIDTH> local_id_pod =
                    packed_output.range(NODE_ID_BITWIDTH - 1, 0);
                ap_fixed_pod_t dist_pod =
                    packed_output.range(bits_per_output - 1, NODE_ID_BITWIDTH);

                int local_id = local_id_pod;
                if (p_graph.vtx_map_rev.count(local_id) == 0)
                    continue; // Invalid local ID

                int global_id = p_graph.vtx_map_rev.at(local_id);
                if (global_id >= m_num_vertices)
                    continue; // Skip if ID is out of bounds (padding)

                distance_t dist_24b =
                    *reinterpret_cast<distance_t *>(&dist_pod);
                distance_t new_dist =
                    dist_24b; // Widen for host-side master copy

                printf("[BIG] Unpacked result: global_id=%d, new_dist=%f\n",
                       global_id, (float)new_dist);
                fflush(nullptr);

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
                ap_fixed_pod_t dist_pod =
                    packed_output.range(bits_per_output - 1, NODE_ID_BITWIDTH);

                int local_id = local_id_pod;
                if (p_graph.vtx_map_rev.count(local_id) == 0)
                    continue;

                int global_id = p_graph.vtx_map_rev.at(local_id);
                if (global_id >= m_num_vertices)
                    continue;

                distance_t dist_24b =
                    *reinterpret_cast<distance_t *>(&dist_pod);
                distance_t new_dist = dist_24b;

                printf("[LITTLE] Unpacked result: global_id=%d, new_dist=%f\n",
                       global_id, (float)new_dist);
                fflush(nullptr);

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
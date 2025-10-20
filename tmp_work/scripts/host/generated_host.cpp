#include "generated_host.h"
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
    if (start_node >= 0 && start_node < m_num_vertices) {
        h_distances[start_node] = 0;
    }

    // 2. Prepare host-side input buffers for each kernel
    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;
    big_kernel_input_buffers.resize(container.SPs.size());
    little_kernel_input_buffers.resize(container.DPs.size());
    hbm_manager_host_buffers.resize(1);

    auto start_time = std::chrono::system_clock::now();
    auto current_time = start_time;


    std::cout<<"DEBUG big:"<<big_kernel_input_buffers.size()<<"little :"<<little_kernel_input_buffers.size()<<std::endl;
    // --- 2.1: 为 BIG kernels 手动序列化数据 (带 Padding) ---
    // DISABLED FOR LITTLE KERNEL TESTING
    /*
    for (size_t i = 0; i < big_kernel_input_buffers.size(); ++i) {
        const auto &p_graph = container.SPs[i].partitioned_graph;

        // --- Pack node distances (ap_fixed<24,8> -> 3 bytes) ---
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
                                            padding_needed,
                                            0); 
                }

                int global_id = p_graph.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];

                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            // 关键修正点：原big kernel的hbm_manager索引也应为0，因为它的大小是1
            hbm_manager_host_buffers[0].packed_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(hbm_manager_host_buffers[0].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Preparing data structures big node dist ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;

        // --- Pack edge properties (node_id<24b> + weight<24b> -> 6 bytes) ---
        {
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + WEIGHT_BITWIDTH) / 8;
            const size_t edges_per_word = bytes_per_word / bytes_per_edge;
            const size_t word_number =
                (p_graph.num_edges + edges_per_word - 1) / edges_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

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
                for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                    edge_bytes[b] = (dest_id >> (8 * b)) & 0xFF;
                }

                weight_t weight_val = (float)p_graph.weights[j];
                std::memcpy(edge_bytes + (NODE_ID_BITWIDTH / 8), &weight_val,
                            (WEIGHT_BITWIDTH / 8));

                temp_byte_buffer.insert(temp_byte_buffer.end(), edge_bytes,
                                        edge_bytes + bytes_per_edge);
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

        // --- Pack source IDs (node_id_t -> 4 bytes) ---
        {
            const size_t bytes_per_id = sizeof(node_id_t);
            const size_t ids_per_word = bytes_per_word / bytes_per_id;
            const size_t word_number =
                (p_graph.num_edges + ids_per_word - 1) / ids_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            for (int j = 0; j < p_graph.num_vertices; ++j) {
                for (int k = p_graph.offsets[j]; k < p_graph.offsets[j + 1];
                     ++k) {
                    node_id_t src_id = j;
                    const char *id_bytes =
                        reinterpret_cast<const char *>(&src_id);
                    for (size_t l = 0; l < bytes_per_id; ++l) {
                        temp_byte_buffer.push_back(id_bytes[l]);
                    }
                }
            }

            big_kernel_input_buffers[i].packed_src_ids.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(big_kernel_input_buffers[i].packed_src_ids.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout
            << "--- [Host] Phase 0: Preparing data structures src ids ("
            << std::chrono::duration<double>(current_time - start_time).count()
            << " sec) ---" << std::endl;
        start_time = current_time;
    }
    */
    
    // --- 2.2: 为 LITTLE kernels 手动序列化数据 (带 Padding) ---
    // ENABLED AND CORRECTED FOR LITTLE KERNEL TESTING
    for (size_t i = 0; i < little_kernel_input_buffers.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;


        
        // --- Pack node distances (ap_fixed<24,8> -> 3 bytes) ---
        // 关键修正: 完全复制big kernel的逻辑，为HBM Manager准备数据
        //要加入4096 边界检查
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
                                            padding_needed,
                                            0); 
                }

                int global_id = p_graph.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];

                const char *data_ptr =
                    reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr,
                                        data_ptr + bytes_per_dist);
            }
            // 关键修正点：目标是hbm_manager_host_buffers[0]，因为只有一个HBM manager
            hbm_manager_host_buffers[0].packed_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word,
                0);
            std::memcpy(hbm_manager_host_buffers[0].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout << "--- [Host] Phase 0: Preparing data structures little node dist ("
                  << std::chrono::duration<double>(current_time - start_time).count()
                  << " sec) ---" << std::endl;
        start_time = current_time;


    
        /* [!!] REVISED AND MERGED PACKING LOGIC [!!]                     */
        /***********************************************************************************/

        // --- 1. 定义对齐和填充常量 ---
        const int BURST_SIZE = 8; 
        const uint32_t SRC_BUFFER_SIZE = 4096;
        
        const node_id_t PSEUDO_SRC_ID = (node_id_t)-1;   // 0xFFFFFFFF, MSB is set
        const uint32_t PSEUDO_DST_ID = (uint32_t)-1;     // 0xFFFFFFFF, MSB is set
        const weight_t PSEUDO_WEIGHT = 0.0f;           // Value doesn't matter, kernel ignores it

        const size_t bytes_per_word = AXI_BUS_WIDTH / 8;

        // --- 2. 准备临时缓冲区 ---
        // 边属性 (dst, weight)
        const size_t bytes_per_edge_prop = (NODE_ID_BITWIDTH + WEIGHT_BITWIDTH) / 8;
        std::vector<char> temp_props_buffer;
        temp_props_buffer.reserve((p_graph.num_edges + p_graph.num_vertices * BURST_SIZE) * bytes_per_edge_prop); 

        // 源ID (src)
        const size_t bytes_per_src_id = sizeof(node_id_t);
        std::vector<char> temp_src_buffer;
        temp_src_buffer.reserve((p_graph.num_edges + p_graph.num_vertices * BURST_SIZE) * bytes_per_src_id);

        // --- 状态跟踪变量 ---
        uint32_t last_src_buffer_idx = 0xFFFFFFFF; // 初始为无效值
        uint64_t total_edges_packed = 0; // 跟踪已打包的边总数

        // --- 预先序列化“伪边”以提高效率 ---
        char pseudo_edge_bytes[bytes_per_edge_prop];
        for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
            pseudo_edge_bytes[b] = (PSEUDO_DST_ID >> (8 * b)) & 0xFF;
        }
        std::memcpy(pseudo_edge_bytes + (NODE_ID_BITWIDTH / 8), &PSEUDO_WEIGHT, (WEIGHT_BITWIDTH / 8));
        const char *pseudo_src_bytes = reinterpret_cast<const char *>(&PSEUDO_SRC_ID);

        // --- 3. 合并的打包循环 (遍历CSR图) ---
        for (int j = 0; j < p_graph.num_vertices; ++j) {
            node_id_t src_id = j; // 'j' 是本地源ID

            // 仅当该顶点有出边时才进行边界检查
            if (p_graph.offsets[j] < p_graph.offsets[j + 1]) {
                uint32_t current_src_buffer_idx = src_id / SRC_BUFFER_SIZE;
                if (last_src_buffer_idx == 0xFFFFFFFF) { // 第一次遇到有边的顶点
                    last_src_buffer_idx = current_src_buffer_idx;
                }

                // 如果 src_id 跨越了4096边界...
                if (current_src_buffer_idx != last_src_buffer_idx) {
                    // ...检查是否需要填充以对齐burst
                    int mod_burst = total_edges_packed % BURST_SIZE;
                    if (mod_burst != 0) {
                        int padding_needed = BURST_SIZE - mod_burst;
                        
                        for (int p = 0; p < padding_needed; ++p) {
                            // 填充 src_ids 缓冲区 (带字节对齐)
                            if ((temp_src_buffer.size() % bytes_per_word) + bytes_per_src_id > bytes_per_word) {
                                temp_src_buffer.insert(temp_src_buffer.end(), bytes_per_word - (temp_src_buffer.size() % bytes_per_word), 0);
                            }
                            temp_src_buffer.insert(temp_src_buffer.end(), pseudo_src_bytes, pseudo_src_bytes + bytes_per_src_id);

                            // 填充 edge_props 缓冲区 (带字节对齐)
                            if ((temp_props_buffer.size() % bytes_per_word) + bytes_per_edge_prop > bytes_per_word) {
                                temp_props_buffer.insert(temp_props_buffer.end(), bytes_per_word - (temp_props_buffer.size() % bytes_per_word), 0);
                            }
                            temp_props_buffer.insert(temp_props_buffer.end(), pseudo_edge_bytes, pseudo_edge_bytes + bytes_per_edge_prop);
                        }
                        total_edges_packed += padding_needed;
                    }
                    last_src_buffer_idx = current_src_buffer_idx;
                }
            }

            // 4. 打包该顶点的所有真实边
            for (int k = p_graph.offsets[j]; k < p_graph.offsets[j + 1]; ++k) {
                // --- A. 打包 src_id (带字节对齐) ---
                if ((temp_src_buffer.size() % bytes_per_word) + bytes_per_src_id > bytes_per_word) {
                    temp_src_buffer.insert(temp_src_buffer.end(), bytes_per_word - (temp_src_buffer.size() % bytes_per_word), 0);
                }
                const char *id_bytes = reinterpret_cast<const char *>(&src_id);
                temp_src_buffer.insert(temp_src_buffer.end(), id_bytes, id_bytes + bytes_per_src_id);

                // --- B. 打包 edge_prop (dst, weight) (带字节对齐) ---
                if ((temp_props_buffer.size() % bytes_per_word) + bytes_per_edge_prop > bytes_per_word) {
                    temp_props_buffer.insert(temp_props_buffer.end(), bytes_per_word - (temp_props_buffer.size() % bytes_per_word), 0);
                }
                
                char edge_bytes[bytes_per_edge_prop];
                uint32_t dest_id = p_graph.columns[k];
                for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                    edge_bytes[b] = (dest_id >> (8 * b)) & 0xFF;
                }
                weight_t weight_val = (float)p_graph.weights[k];
                std::memcpy(edge_bytes + (NODE_ID_BITWIDTH / 8), &weight_val, (WEIGHT_BITWIDTH / 8));
                temp_props_buffer.insert(temp_props_buffer.end(), edge_bytes, edge_bytes + bytes_per_edge_prop);

                total_edges_packed++;
            }
        }

        // --- 5. 最终填充 (修复 size==0 错误) ---
        int padding_needed = 0;
        if (total_edges_packed == 0) {
            padding_needed = BURST_SIZE; 
        } else if (total_edges_packed % BURST_SIZE != 0) {
            padding_needed = BURST_SIZE - (total_edges_packed % BURST_SIZE);
        }

        if (padding_needed > 0) {
            for (int p = 0; p < padding_needed; ++p) {
                // 填充 src_ids 缓冲区 (带字节对齐)
                if ((temp_src_buffer.size() % bytes_per_word) + bytes_per_src_id > bytes_per_word) {
                    temp_src_buffer.insert(temp_src_buffer.end(), bytes_per_word - (temp_src_buffer.size() % bytes_per_word), 0);
                }
                temp_src_buffer.insert(temp_src_buffer.end(), pseudo_src_bytes, pseudo_src_bytes + bytes_per_src_id);

                // 填充 edge_props 缓冲区 (带字节对齐)
                if ((temp_props_buffer.size() % bytes_per_word) + bytes_per_edge_prop > bytes_per_word) {
                    temp_props_buffer.insert(temp_props_buffer.end(), bytes_per_word - (temp_props_buffer.size() % bytes_per_word), 0);
                }
                temp_props_buffer.insert(temp_props_buffer.end(), pseudo_edge_bytes, pseudo_edge_bytes + bytes_per_edge_prop);
            }
        }

        // --- 6. 将数据从temp缓冲区复制到最终的主机缓冲区 ---
        little_kernel_input_buffers[i].packed_edge_props.resize(
            (temp_props_buffer.size() + bytes_per_word - 1) / bytes_per_word, 0);
        std::memcpy(little_kernel_input_buffers[i].packed_edge_props.data(),
                    temp_props_buffer.data(), temp_props_buffer.size());
        
        little_kernel_input_buffers[i].packed_src_ids.resize(
            (temp_src_buffer.size() + bytes_per_word - 1) / bytes_per_word, 0);
        std::memcpy(little_kernel_input_buffers[i].packed_src_ids.data(),
                    temp_src_buffer.data(), temp_src_buffer.size());
        
        /***********************************************************************************/
        /* [!!] END OF REVISION [!!]                            */



        /*
        // --- Pack edge properties (node_id<24b> + weight<24b> -> 6 bytes) ---
        // 关键修正: 完全复制big kernel的逻辑
        // 要加入4096边界检查
        {
            const size_t bytes_per_edge =
                (NODE_ID_BITWIDTH + WEIGHT_BITWIDTH) / 8;
            const size_t edges_per_word = bytes_per_word / bytes_per_edge;
            const size_t word_number = (p_graph.num_edges + edges_per_word - 1) /
                                       edges_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

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
                for (int b = 0; b < NODE_ID_BITWIDTH / 8; ++b) {
                    edge_bytes[b] = (dest_id >> (8 * b)) & 0xFF;
                }

                weight_t weight_val = (float)p_graph.weights[j];
                std::memcpy(edge_bytes + (NODE_ID_BITWIDTH / 8), &weight_val, (WEIGHT_BITWIDTH / 8));
                temp_byte_buffer.insert(temp_byte_buffer.end(), edge_bytes,
                                        edge_bytes + bytes_per_edge);
            }
            little_kernel_input_buffers[i].packed_edge_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) /
                bytes_per_word, 0);
            std::memcpy(little_kernel_input_buffers[i].packed_edge_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }

        current_time = std::chrono::system_clock::now();
        std::cout << "--- [Host] Phase 0: Preparing data structures little edge props ("
                  << std::chrono::duration<double>(current_time - start_time).count()
                  << " sec) ---" << std::endl;
        start_time = current_time;

        // --- Pack source IDs (node_id_t -> 4 bytes) ---
        // 关键修正: 完全复制big kernel的逻辑
        // 要加入4096边界检查
        {
            const size_t bytes_per_id = sizeof(node_id_t);
            const size_t ids_per_word = bytes_per_word / bytes_per_id;
            const size_t word_number =
                (p_graph.num_edges + ids_per_word - 1) / ids_per_word;
            std::vector<char> temp_byte_buffer;
            temp_byte_buffer.reserve(word_number * bytes_per_word);

            for (int j = 0; j < p_graph.num_vertices; ++j) {
                for (int k = p_graph.offsets[j]; k < p_graph.offsets[j + 1];
                     ++k) {
                    node_id_t src_id = j;
                    const char *id_bytes =
                        reinterpret_cast<const char *>(&src_id);
                    for (size_t l = 0; l < bytes_per_id; ++l) {
                        temp_byte_buffer.push_back(id_bytes[l]);
                    }
                }
            }

            little_kernel_input_buffers[i].packed_src_ids.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) /
                    bytes_per_word,
                0);
            std::memcpy(little_kernel_input_buffers[i].packed_src_ids.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }



        current_time = std::chrono::system_clock::now();
        std::cout << "--- [Host] Phase 0: Preparing data structures little src ids ("
                  << std::chrono::duration<double>(current_time - start_time).count()
                  << " sec) ---" << std::endl;
        start_time = current_time;
        */



    }
        
}

void AlgorithmHost::setup_buffers(const PartitionContainer &container) {
    cl_int err;
    std::cout << "--- [Host] Phase 1: Setting up HBM buffers for all kernels ---" << std::endl;

    big_kernel_buffers.clear();
    little_kernel_buffers.clear();
    hbm_manager_buffers.clear(); // Clear HBM manager buffers as well
    big_kernel_host_outputs.resize(container.SPs.size());
    little_kernel_host_outputs.resize(container.DPs.size());

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;

    // --- 1.3: Setup buffers for BIG kernels (Sparse Partitions) ---
    // DISABLED FOR LITTLE KERNEL TESTING
    /*
    for (size_t i = 0; i < container.SPs.size(); ++i) {
        // ... (BIG KERNEL LOGIC IS COMMENTED OUT)
    }
    */
    
    // --- Setup buffer for HBM Manager Kernel (Now for Little Kernels) ---
    // Assuming we have at least one little kernel to test
    if (!container.DPs.empty()) {
        KernelBuffers hbm_buffers;
        cl_mem_ext_ptr_t hbm_ext_in2;
        // This HBM bank should be accessible by the little kernels
        hbm_ext_in2.flags = XCL_MEM_TOPOLOGY | 2; // Example HBM bank
        hbm_ext_in2.obj = nullptr;
        hbm_ext_in2.param = 0;
        
        size_t num_dist_words = hbm_manager_host_buffers[0].packed_node_props.size();
        OCL_CHECK(err,
                  hbm_buffers.node_props_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_dist_words * bytes_per_word, &hbm_ext_in2, &err));

        hbm_manager_buffers.push_back(hbm_buffers);
    }


    // --- 1.4: Setup buffers for LITTLE kernels (Dense Partitions) ---
    // ENABLED FOR LITTLE KERNEL TESTING
    for (size_t i = 0; i < container.DPs.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;
        KernelBuffers buffers;

        cl_mem_ext_ptr_t hbm_ext_in0, hbm_ext_in1, hbm_ext_out;
        hbm_ext_in0.flags = XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_input_id[i];
        hbm_ext_in0.obj = nullptr;
        hbm_ext_in0.param = 0;
        hbm_ext_in1.flags = XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_input_id[i];
        hbm_ext_in1.obj = nullptr;
        hbm_ext_in1.param = 0;
        hbm_ext_out.flags = XCL_MEM_TOPOLOGY | acc.little_kernel_hbm_output_id[i];
        hbm_ext_out.obj = nullptr;
        hbm_ext_out.param = 0;

        size_t num_id_words = little_kernel_input_buffers[i].packed_src_ids.size();
        size_t num_edge_words = little_kernel_input_buffers[i].packed_edge_props.size();

        OCL_CHECK(err,
                  buffers.src_ids_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_id_words * bytes_per_word, &hbm_ext_in0, &err));
        OCL_CHECK(err,
                  buffers.edge_props_buf = cl::Buffer(
                      acc.context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_edge_words * bytes_per_word, &hbm_ext_in1, &err));
        
        // Note: node_props_buf is no longer set here, it's handled by the HBM manager

        size_t bits_per_output = NODE_ID_BITWIDTH + DISTANCE_BITWIDTH + OUT_END_MARKER_BITWIDTH;
        size_t max_dst_local_id = 0;
        for (size_t e = 0; e < p_graph.num_edges; ++e) {
            int dst = p_graph.columns[e];
            if (dst > max_dst_local_id) {
                max_dst_local_id = dst;
            }
        }
        size_t num_dst_vertices = max_dst_local_id + 1;
        size_t vert_num_in_word = AXI_BUS_WIDTH / bits_per_output;
        size_t num_output_words = (num_dst_vertices + vert_num_in_word - 1) / vert_num_in_word + 1;

        little_kernel_host_outputs[i].resize(num_output_words);
        OCL_CHECK(err,
                  buffers.output_buf = cl::Buffer(
                      acc.context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
                      num_output_words * bytes_per_word, &hbm_ext_out, &err));

        little_kernel_buffers.push_back(buffers);
    }

    std::cout << "[SUCCESS] HBM buffers created for " << container.DPs.size() << " little kernels and 1 HBM manager." << std::endl;
}

void AlgorithmHost::update_data(const PartitionContainer &container) {
    std::cout << "--- [Host] Phase 2.1: Updating host-side data for new iteration ---" << std::endl;

    const size_t bytes_per_word = AXI_BUS_WIDTH / 8;

    // DISABLED FOR LITTLE KERNEL TESTING
    /*
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        // ... (BIG KERNEL LOGIC IS COMMENTED OUT)
    }
    */

    // ENABLED for LITTLE KERNEL TESTING
    // This loop now only updates the HBM manager's buffer, which is correct for subsequent iterations.
    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;
        {
            const size_t bytes_per_dist = DISTANCE_BITWIDTH / 8;
            std::vector<char> temp_byte_buffer;
            for (int j = 0; j < p_graph.num_vertices; ++j) {
                if ((temp_byte_buffer.size() % bytes_per_word) + bytes_per_dist > bytes_per_word) {
                    size_t padding_needed = bytes_per_word - (temp_byte_buffer.size() % bytes_per_word);
                    temp_byte_buffer.insert(temp_byte_buffer.end(), padding_needed, 0);
                }
                int global_id = p_graph.vtx_map_rev.at(j);
                distance_t dist_val = h_distances[global_id];
                const char *data_ptr = reinterpret_cast<const char *>(&dist_val);
                temp_byte_buffer.insert(temp_byte_buffer.end(), data_ptr, data_ptr + bytes_per_dist);
            }
            hbm_manager_host_buffers[0].packed_node_props.resize(
                (temp_byte_buffer.size() + bytes_per_word - 1) / bytes_per_word, 0);
                
            std::memcpy(hbm_manager_host_buffers[0].packed_node_props.data(),
                        temp_byte_buffer.data(), temp_byte_buffer.size());
        }
    }

    std::cout << "[SUCCESS] Host-side data updated for new iteration." << std::endl;
}

void AlgorithmHost::transfer_data_to_fpga(const PartitionContainer &container) {
    cl_int err;
    std::cout << "--- [Host] Phase 2: Packing and transferring data to HBM ---" << std::endl;

    // DISABLED FOR LITTLE KERNEL TESTING
    /*
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        // ... (BIG KERNEL LOGIC IS COMMENTED OUT)
    }
    */
    
    // Transfer data for HBM Manager (for little kernels)
    if (!hbm_manager_buffers.empty()) {
        OCL_CHECK(err,
              err = acc.little_gs_queue[0].enqueueWriteBuffer(//只有一个little kernel
                  hbm_manager_buffers[0].node_props_buf, CL_FALSE, 0,
                  hbm_manager_host_buffers[0].packed_node_props.size() * sizeof(bus_word_t),
                  hbm_manager_host_buffers[0].packed_node_props.data()));
    }

    // ENABLED FOR LITTLE KERNEL TESTING
    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        // Note: node_props_buf is no longer transferred here.
        OCL_CHECK(err,
                  err = acc.little_gs_queue[i].enqueueWriteBuffer(
                      little_kernel_buffers[i].edge_props_buf, CL_FALSE, 0,
                      little_kernel_input_buffers[i].packed_edge_props.size() * sizeof(bus_word_t),
                      little_kernel_input_buffers[i].packed_edge_props.data()));
        OCL_CHECK(err,
                  err = acc.little_gs_queue[i].enqueueWriteBuffer(
                      little_kernel_buffers[i].src_ids_buf, CL_FALSE, 0,
                      little_kernel_input_buffers[i].packed_src_ids.size() * sizeof(bus_word_t),
                      little_kernel_input_buffers[i].packed_src_ids.data()));
    }

    // --- 2.4: 在所有命令入队后，执行一次全局同步 ---
    for (auto &q : acc.big_gs_queue) q.finish(); // Stays for safety, though should be empty
    acc.hbm_manager_queue.finish();
    for (auto &q : acc.little_gs_queue) q.finish();

    std::cout << "[SUCCESS] All data packed and transferred for current iteration." << std::endl;
}

void AlgorithmHost::execute_kernel_iteration(
    const PartitionContainer &container,
    std::vector<cl::Event> &big_kernel_events,
    std::vector<cl::Event> &little_kernel_events) {
    cl_int err;
    std::cout << "--- [Host] Phase 3: Enqueuing kernel tasks ---" << std::endl;

    // DISABLED FOR LITTLE KERNEL TESTING
    /*
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        // ... (BIG KERNEL LOGIC IS COMMENTED OUT)
    }
    */

    // Enqueue HBM Manager (for little kernels)
    // if (!hbm_manager_buffers.empty()) {
    //     auto &kernel = acc.hbm_manager_krnl;
    //     auto &buffers = hbm_manager_buffers[0];
    //     
    //     // This assumes the number of vertices for the manager is the total number
    //     // or a relevant subset for all little kernels. Using m_num_vertices for now.
    //     int num_nodes_for_manager = m_num_vertices;
// 
    //     int arg_idx = 0;
    //     OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.node_props_buf));
    //     OCL_CHECK(err, err = kernel.setArg(arg_idx++, num_nodes_for_manager));
// 
    //     // Use the first little kernel's event to chain the manager
    //     cl::Event *event_ptr = &little_kernel_events[0];
    //     OCL_CHECK(err, err = acc.hbm_manager_queue.enqueueTask(kernel, nullptr, event_ptr));
    // }


    // ENABLED AND CORRECTED FOR LITTLE KERNEL TESTING
    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        auto &kernel = acc.little_gs_krnls[i];
        auto &buffers = little_kernel_buffers[i];
        const auto &p_graph = container.DPs[i].partitioned_graph;

        auto &hbm_buffers = hbm_manager_buffers[0];
        int num_nodes_for_manager = m_num_vertices;

        int arg_idx = 0;
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.src_ids_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.edge_props_buf));
        // 关键修正: little kernel不再直接接收node_props_buf
        // OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.node_props_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, buffers.output_buf));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_vertices));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, p_graph.num_edges));



        // Assuming dst_num is required, using num_vertices as a placeholder.
        // You might need to calculate the actual number of destination vertices.
        size_t max_dst_local_id = 0;
        for (size_t e = 0; e < p_graph.num_edges; ++e) {
            if (p_graph.columns[e] > max_dst_local_id) max_dst_local_id = p_graph.columns[e];
        }
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, (int)(max_dst_local_id + 1)));
        OCL_CHECK(err, err = kernel.setArg(arg_idx++, hbm_buffers.node_props_buf));

        cl::Event *event_ptr = &little_kernel_events[i];
        OCL_CHECK(err, err = acc.little_gs_queue[i].enqueueTask(kernel, nullptr, event_ptr));

        
    }
    std::cout << "[SUCCESS] All kernel tasks enqueued for one iteration." << std::endl;
}

void AlgorithmHost::transfer_data_from_fpga() {
    cl_int err;
    std::cout << "--- [Host] Phase 4: Transferring results from HBM ---" << std::endl;
    
    // DISABLED
    /*
    for (size_t i = 0; i < big_kernel_buffers.size(); ++i) {
        OCL_CHECK(err,
                  err = acc.big_gs_queue[i].enqueueReadBuffer(
                      big_kernel_buffers[i].output_buf, CL_FALSE, 0,
                      big_kernel_host_outputs[i].size() * sizeof(bus_word_t),
                      big_kernel_host_outputs[i].data()));
    }
    */

    // ENABLED
    for (size_t i = 0; i < little_kernel_buffers.size(); ++i) {
        OCL_CHECK(err,
                  err = acc.little_gs_queue[i].enqueueReadBuffer(
                      little_kernel_buffers[i].output_buf, CL_FALSE, 0,
                      little_kernel_host_outputs[i].size() * sizeof(bus_word_t),
                      little_kernel_host_outputs[i].data()));
    }

    // Wait for all transfers to complete
    for (auto &q : acc.big_gs_queue) q.finish();
    for (auto &q : acc.little_gs_queue) q.finish();
    std::cout << "[SUCCESS] All results transferred from HBM." << std::endl;
}

bool AlgorithmHost::check_convergence_and_update(const PartitionContainer &container) {
    bool changed = false;
    std::cout << "--- [Host] Phase 5: Unpacking results and checking for convergence ---" << std::endl;

    std::map<int, distance_t> min_distances;
    const int bits_per_output = NODE_ID_BITWIDTH + DISTANCE_BITWIDTH + OUT_END_MARKER_BITWIDTH;
    const int outputs_per_word = AXI_BUS_WIDTH / bits_per_output;

    // DISABLED
    /*
    for (size_t i = 0; i < big_kernel_host_outputs.size(); ++i) {
        // ... (BIG KERNEL LOGIC IS COMMENTED OUT)
    }
    */

    // ENABLED
    for (size_t i = 0; i < little_kernel_host_outputs.size(); ++i) {
        const auto &p_graph = container.DPs[i].partitioned_graph;
        for (const auto &word : little_kernel_host_outputs[i]) {
            out_end_marker_t end_flag = 0;
            for (int k = 0; k < outputs_per_word; ++k) {
                int bit_offset = k * bits_per_output;
                ap_uint<bits_per_output> packed_output = word.range(bit_offset + bits_per_output - 1, bit_offset);

                ap_uint<NODE_ID_BITWIDTH> local_id_pod = packed_output.range(NODE_ID_BITWIDTH - 1, 0);
                ap_fixed_pod_t dist_pod = packed_output.range(NODE_ID_BITWIDTH + DISTANCE_BITWIDTH - 1, NODE_ID_BITWIDTH);
                end_flag = packed_output.range(bits_per_output - 1, NODE_ID_BITWIDTH + DISTANCE_BITWIDTH);

                if (end_flag != 0) break;

                int local_id = local_id_pod;
                if (p_graph.vtx_map_rev.count(local_id) == 0) continue;

                int global_id = p_graph.vtx_map_rev.at(local_id);
                if (global_id >= m_num_vertices) continue;

                distance_t new_dist = *reinterpret_cast<distance_t *>(&dist_pod);

                if (min_distances.find(global_id) == min_distances.end() || new_dist < min_distances[global_id]) {
                    min_distances[global_id] = new_dist;
                }
            }
            if (end_flag != 0) break;
        }
    }

    for (auto const &[global_id, new_dist] : min_distances) {
        if (global_id < m_num_vertices && new_dist < h_distances[global_id]) {
            h_distances[global_id] = new_dist;
            changed = true;
        }
    }

    if (changed) {
        std::cout << "[INFO] Distances updated. Preparing for next iteration." << std::endl;
    } else {
        std::cout << "[INFO] No distance updates. Algorithm has converged." << std::endl;
    }

    return !changed;
}

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
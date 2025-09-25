#include "graph_preprocess.h"
#include "host_config.h"
#include "iostream"
#include "common.h"

#define INF 16384

// helper functions：
/**
 * @brief 将CSR格式的图转换为COO格式（即 EDGE_TYPE 的 vector）
 * @param graph 输入的CSR图
 * @return 包含所有边的 std::vector<EDGE_TYPE>
 */
std::vector<EDGE_TYPE> csrToCoo(const GraphCSR& graph) {
    std::vector<EDGE_TYPE> all_edges;
    all_edges.reserve(graph.num_edges); // 预分配内存以提高效率

    for (int u = 0; u < graph.num_vertices; ++u) {
        for (int i = graph.offsets[u]; i < graph.offsets[u + 1]; ++i) {
            int v = graph.columns[i];
            int w = graph.weights[i];

            EDGE_TYPE edge;
            edge.src.id = u;
            edge.dst.id = v;
            
            // --- *** 核心修正点：将所有整数值转换为 ap_fixed 的二进制模式 *** ---

            // 1. 处理 weight
            ap_fixed<32, 16> weight_fp = w;
            edge.weight = *reinterpret_cast<int32_t*>(&weight_fp);

            // 2. 处理 src.distance (保留您原来的 if/else 逻辑)
            int src_dist_val = (edge.src.id == 0) ? 0 : INF;
            ap_fixed<32, 16> src_dist_fp = src_dist_val;
            edge.src.distance = *reinterpret_cast<int32_t*>(&src_dist_fp);
            
            // 3. 处理 dst.distance (保留您原来的 if/else 逻辑)
            int dst_dist_val = (edge.dst.id == 0) ? 0 : INF;
            ap_fixed<32, 16> dst_dist_fp = dst_dist_val;
            edge.dst.distance = *reinterpret_cast<int32_t*>(&dst_dist_fp);
            
            // --- *** 修正结束 *** ---
            
            all_edges.push_back(edge);
        }
    }
    return all_edges;
}

/**
 * @brief 将一个边的vector打包成多个批处理，并直接填充到目标vector中。
 * @param edges 输入的边列表。
 * @param batches [输出] 用于接收批处理数据的、使用对齐内存的vector的引用。
 */
// **********************************************************
// *** 修正点 1: 修改函数签名，不再返回值，而是通过引用填充 ***
void createBatches(const std::vector<EDGE_TYPE>& edges, 
                   std::vector<BATCH_TYPE, aligned_allocator<BATCH_TYPE>>& batches) {
    
    batches.clear(); // 确保开始前目标 vector 是空的
    if (edges.empty()) {
        return;
    }

    BATCH_TYPE current_batch;
    int edges_in_batch = 0;

    for (const auto& edge : edges) {
        current_batch.data[edges_in_batch] = edge;
        edges_in_batch++;

        if (edges_in_batch == PE_NUM) {
            current_batch.end_pos = PE_NUM;
            current_batch.end_flag = false;
            batches.push_back(current_batch); // 直接填充到传入的 vector 中
            edges_in_batch = 0;
        }
    }

    // 处理最后一个可能未满的batch
    if (edges_in_batch > 0) {
        current_batch.end_pos = edges_in_batch;
        current_batch.end_flag = true;
        batches.push_back(current_batch);
    } 
    // 如果整个流恰好被PE_NUM整除，那么最后一个已满的batch也应该是结束batch
    else if (!batches.empty()) {
        batches.back().end_flag = true;
    }
    // 函数现在是 void，不需要 return 语句
}
// **********************************************************


partition_container_dt partitionGraph (const GraphCSR* graph) {

    // --- 1: CSR to COO ---
    std::vector<edge_t> all_edges = csrToCoo(*graph);

    // --- 2: 将 COO vector 分割成多个部分 ---

    //方案一：均分
    const int num_partitions = LITTLE_KERNEL_NUM + BIG_KERNEL_NUM;
    size_t total_edges = all_edges.size();
    size_t base_partition_size = total_edges / num_partitions;
    size_t remainder = total_edges % num_partitions;

    std::vector<std::vector<edge_t>> coo_parts(num_partitions);
    auto current_iter = all_edges.begin();
    for (int i = 0; i < num_partitions; ++i) {
        size_t part_size = base_partition_size + (i < remainder ? 1 : 0);
        auto end_iter = current_iter + part_size;
        coo_parts[i].assign(current_iter, end_iter);
        current_iter = end_iter;
    }
    

    /*
    // --- 2: [DEBUG] 修改分区逻辑：将所有边放入第一个分区 ---
    const int num_partitions = LITTLE_KERNEL_NUM + BIG_KERNEL_NUM; // 仍然是 3
    
    // 创建一个包含 num_partitions 个空 vector 的 vector
    std::vector<std::vector<edge_t>> coo_parts(num_partitions);

    // 检查确保至少有一个分区存在，然后将所有边复制到第一个分区
    if (num_partitions > 0) {
        coo_parts[0] = all_edges; 
    }
    
    // 其他分区 (coo_parts[1], coo_parts[2], ...) 自动保持为空。

    */


    // --- 3: 创建并填充 partition_container_dt ---
    std::cout << "Populating the Partition Container with batched data..." << std::endl;

    partition_container_dt container;
    container.num_graph_vertices = graph->num_vertices;
    container.num_graph_edges = graph->num_edges;

    // --- 为小核（Dense Partitions）创建分区描述符 ---
    for (size_t i = 0; i < LITTLE_KERNEL_NUM; ++i) {
        
        partition_descriptor_dt pd;

        // **********************************************************
        // *** 修正点 2: 调用新版 createBatches，直接填充 pd.batch_array_host ***
        createBatches(coo_parts[i], pd.batch_array_host);
        // **********************************************************

        // 填充元数据
        pd.num_edges = coo_parts[i].size();
        pd.num_vertices = graph->num_vertices;
        pd.kernel_id = i;
        pd.is_dense = true;
        
        container.DP.push_back(pd);
    }

    // --- 为大核（Sparse Partitions）创建分区描述符 ---
    for (size_t i = 0; i < BIG_KERNEL_NUM; ++i) {
        size_t part_index = i + LITTLE_KERNEL_NUM;
        
        partition_descriptor_dt pd;

        // **********************************************************
        // *** 修正点 2: 调用新版 createBatches，直接填充 pd.batch_array_host ***
        createBatches(coo_parts[part_index], pd.batch_array_host);
        // **********************************************************
        
        // 填充元数据
        pd.num_edges = coo_parts[part_index].size();
        pd.num_vertices = graph->num_vertices;
        pd.kernel_id = part_index;
        pd.is_dense = false;
        
        container.SP.push_back(pd);
    }
    
    // 更新容器中的分区数量
    container.num_sparse_partitions = container.SP.size();
    container.num_dense_partitions = container.DP.size();
    
    std::cout << "Partition Container populated successfully." << std::endl;

    return container;
}

/*
//reorder vertices according to the outdegree of the vertices...
void reorderGraph(CSR* csr){
   
}
*/
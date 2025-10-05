#include "graphyflow_kernel.h"

void memory_loader(
    int instantiate_idx,
    const int *src_offsets, const edge_des_burst_t *edge_des_bursts,
    const int *node_distances, int num_nodes, int num_edges,
    hls::stream<edge_batch_t> &response_to_318,
    hls::stream<struct_ibu_14_t> &all_node_distances_to_343);

void final_writeback(
    int instantiate_idx,
    hls::stream<struct_sbu_19_t> &in_stream,
    KernelOutputBatch *out_o_0_342);

void axi_to_ori_edge_batch_wrapper(
    hls::stream<edge_batch_axi_t> &axi_stream,
    hls::stream<edge_batch_t> &ori_stream) {
    while (true) {
#pragma HLS PIPELINE II=1
        edge_batch_axi_t axi_data = axi_stream.read();
        ori_stream.write(axi2ori(axi_data));
        if (axi_data.last) {
            break;
        }
    }
}

void ori_to_axi_edge_batch_wrapper(
    int instantiate_idx,
    hls::stream<edge_batch_t> &ori_stream,
    hls::stream<edge_batch_axi_t> &axi_stream) {
#pragma HLS function_instantiate variable = instantiate_idx
    while (true) {
#pragma HLS PIPELINE II=1
        edge_batch_t ori_data = ori_stream.read();
        // printf("[%d] Edge batch end_flag: %d, end_pos: %d\n", instantiate_idx, ori_data.end_flag, ori_data.end_pos);
        edge_batch_axi_t axi_data = ori2axi(ori_data);
        axi_data.last = ori_data.end_flag;
        axi_stream.write(axi_data);
        if (ori_data.end_flag) {
            break;
        }
    }
}

void ori_to_axi_struct_ibu_14_wrapper(
    int instantiate_idx,
    hls::stream<struct_ibu_14_t> &ori_stream,
    hls::stream<struct_ibu_14_axi_t> &axi_stream) {
#pragma HLS function_instantiate variable = instantiate_idx
    while (true) {
#pragma HLS PIPELINE II=1
        struct_ibu_14_t ori_data = ori_stream.read();
        // printf("[%d] Node distances batch end_flag: %d, end_pos: %d\n", instantiate_idx, ori_data.end_flag, ori_data.end_pos);
        struct_ibu_14_axi_t axi_data = ori2axi(ori_data);
        axi_data.last = ori_data.end_flag;
        axi_stream.write(axi_data);
        if (ori_data.end_flag) {
            break;
        }
    }
}

void axi_to_ori_struct_sbu_19_wrapper(
    int instantiate_idx,
    hls::stream<struct_sbu_19_axi_t> &axi_stream,
    hls::stream<struct_sbu_19_t> &ori_stream) {
#pragma HLS function_instantiate variable = instantiate_idx
    while (true) {
#pragma HLS PIPELINE II=1
        struct_sbu_19_axi_t axi_data = axi_stream.read();
        ori_stream.write(axi2ori(axi_data));
        // printf("[%d] Result batch end_flag: %d, end_pos: %d\n", instantiate_idx, axi2ori(axi_data).end_flag, axi2ori(axi_data).end_pos);
        if (axi_data.last) {
            break;
        }
    }
}

extern "C" void global_controller(
    // input i/o
    const int *src_offsets_1,
    const int *src_offsets_2,
    const int *src_offsets_3,
    const int *src_offsets_4,
    const edge_des_burst_t *edge_des_bursts_1,
    const edge_des_burst_t *edge_des_bursts_2,
    const edge_des_burst_t *edge_des_bursts_3,
    const edge_des_burst_t *edge_des_bursts_4,
    const int *node_distances_1, 
    const int *node_distances_2, 
    const int *node_distances_3, 
    const int *node_distances_4,
    // output i/o
    KernelOutputBatch *output_ptr_1,
    KernelOutputBatch *output_ptr_2,
    KernelOutputBatch *output_ptr_3,
    KernelOutputBatch *output_ptr_4,
    // graph metadata
    int num_nodes_1,
    int num_nodes_2,
    int num_nodes_3,
    int num_nodes_4,
    int num_edges_1,
    int num_edges_2,
    int num_edges_3,
    int num_edges_4,
    // streams to/from graphyflow kernels
    hls::stream<edge_batch_axi_t> &edge_batches_1,
    hls::stream<edge_batch_axi_t> &edge_batches_2,
    hls::stream<edge_batch_axi_t> &edge_batches_3,
    hls::stream<edge_batch_axi_t> &edge_batches_4,
    hls::stream<struct_ibu_14_axi_t> &node_distances_stream_1, 
    hls::stream<struct_ibu_14_axi_t> &node_distances_stream_2, 
    hls::stream<struct_ibu_14_axi_t> &node_distances_stream_3, 
    hls::stream<struct_ibu_14_axi_t> &node_distances_stream_4,
    hls::stream<struct_sbu_19_axi_t> &result_stream_1,
    hls::stream<struct_sbu_19_axi_t> &result_stream_2,
    hls::stream<struct_sbu_19_axi_t> &result_stream_3,
    hls::stream<struct_sbu_19_axi_t> &result_stream_4
) {
    #pragma HLS INTERFACE m_axi port = src_offsets_1 offset = slave bundle = gmem0
    #pragma HLS INTERFACE m_axi port = src_offsets_2 offset = slave bundle = gmem1
    #pragma HLS INTERFACE m_axi port = src_offsets_3 offset = slave bundle = gmem2
    #pragma HLS INTERFACE m_axi port = src_offsets_4 offset = slave bundle = gmem3
    
    #pragma HLS INTERFACE m_axi port = edge_des_bursts_1 offset = slave bundle = gmem4
    #pragma HLS INTERFACE m_axi port = edge_des_bursts_2 offset = slave bundle = gmem5
    #pragma HLS INTERFACE m_axi port = edge_des_bursts_3 offset = slave bundle = gmem6
    #pragma HLS INTERFACE m_axi port = edge_des_bursts_4 offset = slave bundle = gmem7

    #pragma HLS INTERFACE m_axi port = node_distances_1 offset = slave bundle = gmem8
    #pragma HLS INTERFACE m_axi port = node_distances_2 offset = slave bundle = gmem9
    #pragma HLS INTERFACE m_axi port = node_distances_3 offset = slave bundle = gmem10
    #pragma HLS INTERFACE m_axi port = node_distances_4 offset = slave bundle = gmem11

    #pragma HLS INTERFACE m_axi port = output_ptr_1 offset = slave bundle = gmem0
    #pragma HLS INTERFACE m_axi port = output_ptr_2 offset = slave bundle = gmem1
    #pragma HLS INTERFACE m_axi port = output_ptr_3 offset = slave bundle = gmem2
    #pragma HLS INTERFACE m_axi port = output_ptr_4 offset = slave bundle = gmem3

    #pragma HLS INTERFACE s_axilite port = src_offsets_1 bundle = control
    #pragma HLS INTERFACE s_axilite port = src_offsets_2 bundle = control
    #pragma HLS INTERFACE s_axilite port = src_offsets_3 bundle = control
    #pragma HLS INTERFACE s_axilite port = src_offsets_4 bundle = control

    #pragma HLS INTERFACE s_axilite port = edge_des_bursts_1 bundle = control
    #pragma HLS INTERFACE s_axilite port = edge_des_bursts_2 bundle = control
    #pragma HLS INTERFACE s_axilite port = edge_des_bursts_3 bundle = control
    #pragma HLS INTERFACE s_axilite port = edge_des_bursts_4 bundle = control

    #pragma HLS INTERFACE s_axilite port = node_distances_1 bundle = control
    #pragma HLS INTERFACE s_axilite port = node_distances_2 bundle = control
    #pragma HLS INTERFACE s_axilite port = node_distances_3 bundle = control
    #pragma HLS INTERFACE s_axilite port = node_distances_4 bundle = control

    #pragma HLS INTERFACE s_axilite port = output_ptr_1 bundle = control
    #pragma HLS INTERFACE s_axilite port = output_ptr_2 bundle = control
    #pragma HLS INTERFACE s_axilite port = output_ptr_3 bundle = control
    #pragma HLS INTERFACE s_axilite port = output_ptr_4 bundle = control

    #pragma HLS INTERFACE s_axilite port = num_nodes_1 bundle = control
    #pragma HLS INTERFACE s_axilite port = num_nodes_2 bundle = control
    #pragma HLS INTERFACE s_axilite port = num_nodes_3 bundle = control
    #pragma HLS INTERFACE s_axilite port = num_nodes_4 bundle = control
    #pragma HLS INTERFACE s_axilite port = num_edges_1 bundle = control
    #pragma HLS INTERFACE s_axilite port = num_edges_2 bundle = control
    #pragma HLS INTERFACE s_axilite port = num_edges_3 bundle = control
    #pragma HLS INTERFACE s_axilite port = num_edges_4 bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    #pragma HLS DATAFLOW

    hls::stream<edge_batch_t> edge_batches_1_ori;
    hls::stream<edge_batch_t> edge_batches_2_ori;
    hls::stream<edge_batch_t> edge_batches_3_ori;
    hls::stream<edge_batch_t> edge_batches_4_ori;
    hls::stream<struct_ibu_14_t> node_distances_stream_1_ori;
    hls::stream<struct_ibu_14_t> node_distances_stream_2_ori;
    hls::stream<struct_ibu_14_t> node_distances_stream_3_ori;
    hls::stream<struct_ibu_14_t> node_distances_stream_4_ori;
    hls::stream<struct_sbu_19_t> result_stream_1_ori;
    hls::stream<struct_sbu_19_t> result_stream_2_ori;
    hls::stream<struct_sbu_19_t> result_stream_3_ori;
    hls::stream<struct_sbu_19_t> result_stream_4_ori;

    memory_loader(1, src_offsets_1, edge_des_bursts_1, node_distances_1, num_nodes_1, num_edges_1, edge_batches_1_ori, node_distances_stream_1_ori);
    memory_loader(2, src_offsets_2, edge_des_bursts_2, node_distances_2, num_nodes_2, num_edges_2, edge_batches_2_ori, node_distances_stream_2_ori);
    memory_loader(3, src_offsets_3, edge_des_bursts_3, node_distances_3, num_nodes_3, num_edges_3, edge_batches_3_ori, node_distances_stream_3_ori);
    memory_loader(4, src_offsets_4, edge_des_bursts_4, node_distances_4, num_nodes_4, num_edges_4, edge_batches_4_ori, node_distances_stream_4_ori);

    ori_to_axi_edge_batch_wrapper(1, edge_batches_1_ori, edge_batches_1);
    ori_to_axi_edge_batch_wrapper(2, edge_batches_2_ori, edge_batches_2);
    ori_to_axi_edge_batch_wrapper(3, edge_batches_3_ori, edge_batches_3);
    ori_to_axi_edge_batch_wrapper(4, edge_batches_4_ori, edge_batches_4);

    ori_to_axi_struct_ibu_14_wrapper(1, node_distances_stream_1_ori, node_distances_stream_1);
    ori_to_axi_struct_ibu_14_wrapper(2, node_distances_stream_2_ori, node_distances_stream_2);
    ori_to_axi_struct_ibu_14_wrapper(3, node_distances_stream_3_ori, node_distances_stream_3);
    ori_to_axi_struct_ibu_14_wrapper(4, node_distances_stream_4_ori, node_distances_stream_4);

    axi_to_ori_struct_sbu_19_wrapper(1, result_stream_1, result_stream_1_ori);
    axi_to_ori_struct_sbu_19_wrapper(2, result_stream_2, result_stream_2_ori);
    axi_to_ori_struct_sbu_19_wrapper(3, result_stream_3, result_stream_3_ori);
    axi_to_ori_struct_sbu_19_wrapper(4, result_stream_4, result_stream_4_ori);

    final_writeback(1, result_stream_1_ori, output_ptr_1);
    final_writeback(2, result_stream_2_ori, output_ptr_2);
    final_writeback(3, result_stream_3_ori, output_ptr_3);
    final_writeback(4, result_stream_4_ori, output_ptr_4);
}

// --- PHASE 2.1: UMC Sub-module: Node Property Loader ---

/**
 * @brief Loads all node distances from DDR into a URAM cache.
 * It reads data in wide AXI bus bursts for efficiency.
 *
 * @param node_distances_ddr  DDR pointer for node distances.
 * @param node_distance_cache On-chip URAM cache for node distances
 * (write-only).
 * @param num_nodes           Total number of nodes.
 * @param load_finished_signal Stream to signal completion to the edge loader.
 */
static void
node_property_loader(const int *node_distances_ddr,
                     hls::stream<node_distance_burst_t> &node_distance_burst_stream_0,
                     hls::stream<node_distance_burst_t> &node_distance_burst_stream_1,
                     int num_nodes) {
    // Use ap_uint for wide bus access
    const ap_uint<AXI_BUS_WIDTH> *wide_bus_ptr =
        reinterpret_cast<const ap_uint<AXI_BUS_WIDTH> *>(node_distances_ddr);
    const int num_wide_reads =
        (num_nodes + NUM_WORDS_PER_BUS - 1) / NUM_WORDS_PER_BUS;
    int sent_pack_cnt = 0;
    int total_pack_cnt = (num_nodes + PE_NUM - 1) / PE_NUM;

LOAD_NODES_LOOP:
LOOP_NODE_PROP_LOADER_848:
    for (int i = 0; i < num_wide_reads; ++i) {
#pragma HLS PIPELINE II = 1
        ap_uint<AXI_BUS_WIDTH> wide_word = wide_bus_ptr[i];
        node_distance_burst_t burst;
        int batch_idx = i * NUM_WORDS_PER_BUS / PE_NUM;
    UNPACK_NODES_LOOP:
    LOOP_NODE_PROP_LOADER_UNPACK_852:
        for (int j = 0; j < NUM_WORDS_PER_BUS; j += PE_NUM) {
#pragma HLS UNROLL
            for (int pe = 0; pe < PE_NUM; ++pe) {
#pragma HLS UNROLL
                int node_idx = i * NUM_WORDS_PER_BUS + j + pe;
                if (node_idx < num_nodes) {
                    burst.data[pe] = wide_word.range((j + pe + 1) * DATA_TYPE_WIDTH - 1,
                                                     (j + pe) * DATA_TYPE_WIDTH);
                }
            }
            if (sent_pack_cnt < total_pack_cnt) {
                node_distance_burst_stream_0.write(burst);
                // node_distance_burst_stream_1.write(burst);
                sent_pack_cnt++;
            }
        }
    }

    sent_pack_cnt = 0;
    LOAD_NODES_LOOP_2:
LOOP_NODE_PROP_LOADER_848_2:
    for (int i = 0; i < num_wide_reads; ++i) {
#pragma HLS PIPELINE II = 1
        ap_uint<AXI_BUS_WIDTH> wide_word = wide_bus_ptr[i];
        node_distance_burst_t burst;
        int batch_idx = i * NUM_WORDS_PER_BUS / PE_NUM;
    UNPACK_NODES_LOOP_2:
    LOOP_NODE_PROP_LOADER_UNPACK_852_2:
        for (int j = 0; j < NUM_WORDS_PER_BUS; j += PE_NUM) {
#pragma HLS UNROLL
            for (int pe = 0; pe < PE_NUM; ++pe) {
#pragma HLS UNROLL
                int node_idx = i * NUM_WORDS_PER_BUS + j + pe;
                if (node_idx < num_nodes) {
                    burst.data[pe] = wide_word.range((j + pe + 1) * DATA_TYPE_WIDTH - 1,
                                                     (j + pe) * DATA_TYPE_WIDTH);
                }
            }
            if (sent_pack_cnt < total_pack_cnt) {
                // node_distance_burst_stream_0.write(burst);
                node_distance_burst_stream_1.write(burst);
                sent_pack_cnt++;
            }
        }
    }

}

// --- PHASE 2.2: UMC Sub-module: Edge Loader and Dispatcher ---

// edge_prop_loader
/**
 * @brief Load edge properties from DDR and dispatch to downstream modules.
 *
 * @param edge_descriptors DDR pointer for edge descriptors.
 * @param edge_stream     Stream to output edge descriptors.
 * @param num_edges       Total number of edges.
 */
static void
edge_descriptor_loader(const edge_des_burst_t *edge_des_bursts,
                       hls::stream<edge_descriptor_batch_t> &edge_stream,
                       int num_edges) {
#pragma HLS dependence variable = edge_des_bursts inter false
    // read 8 edges at one time and push
    const int num_batches = (num_edges + PE_NUM - 1) / PE_NUM;
LOAD_EDGES_LOOP:
LOOP_EDGE_DESC_LOADER_872:
    for (int i = 0; i < num_batches; ++i) {
#pragma HLS PIPELINE II = 1
        edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
#pragma HLS dependence variable = edge_batch inter false direction = WAW
        edge_des_burst_t burst = edge_des_bursts[i];
        edge_batch.end_pos = 0;
    LOAD_EDGE_BATCH_LOOP:
    LOOP_EDGE_DESC_LOADER_LOAD_BATCH_877:
        for (int j = 0; j < PE_NUM; ++j) {
#pragma HLS UNROLL
            int edge_idx = i * PE_NUM + j;
            if (edge_idx < num_edges) {
                edge_batch.edges[edge_batch.end_pos] = burst.edges[j];
                edge_batch.end_pos++;
            }
        }
        edge_stream.write(edge_batch);
    }
}

static void src_offset_loader(const int *src_offsets_ddr,
                               hls::stream<int> &src_offsets_stream,
                               int num_nodes) {
    const ap_uint<AXI_BUS_WIDTH> *wide_bus_ptr =
        reinterpret_cast<const ap_uint<AXI_BUS_WIDTH> *>(src_offsets_ddr);
    const int num_wide_reads =
        (num_nodes + 1 + NUM_WORDS_PER_BUS - 1) / NUM_WORDS_PER_BUS;
#pragma HLS dependence variable = wide_bus_ptr inter false

    // read src offsets and push to stream
LOAD_SRC_OFFSETS_LOOP:
LOOP_SRC_OFFSET_LOADER_881:
    for (int i = 0; i <= num_wide_reads; ++i) {
#pragma HLS PIPELINE II = 1
        ap_uint<AXI_BUS_WIDTH> wide_word = wide_bus_ptr[i];
        // src_offsets_stream.write(src_offsets_ddr[i]);
        for (int j = 0; j < NUM_WORDS_PER_BUS; j++) {
#pragma HLS UNROLL
            int node_idx = i * NUM_WORDS_PER_BUS + j;
            int cur_data;
            if (node_idx <= num_nodes) {
                cur_data = wide_word.range((j + 1) * DATA_TYPE_WIDTH - 1,
                                           j * DATA_TYPE_WIDTH);
                src_offsets_stream.write(cur_data);
            }
        }
    }
}

/**
 * @brief Reads CSR graph data, fetches corresponding source node distances from
 * the URAM cache, and serves batches of processed edges to downstream modules.
 */
static void edge_property_loader_and_dispatcher(
    hls::stream<int> &src_offsets_cache_stream,
    hls::stream<edge_descriptor_batch_t> &edge_stream,
    hls::stream<node_distance_burst_t> &node_distance_burst_stream,
    int num_nodes,
    hls::stream<edge_batch_t> &response_stream) {
    // // --- PHASE A: Wait for node properties to be fully cached on-chip ---
    // (void)load_finished_signal.read();
    // --- PHASE B: Cache src_offsets on-chip for fast access ---
//     int src_offsets_cache[MAX_NUM + 1];
// #pragma HLS BIND_STORAGE variable = src_offsets_cache type = RAM_1P impl = BRAM

// CACHE_OFFSETS_LOOP:
// LOOP_EDGE_PROP_LOADER_CACHE_OFFSETS_884:
//     for (int i = 0; i <= num_nodes; ++i) {
// #pragma HLS PIPELINE II = 1
//         // src_offsets_cache[i] = src_offsets_ddr[i];
//         src_offsets_cache_stream.write(src_offsets_ddr[i]);
//     }

    // --- PHASE C: Process and dispatch all edges unconditionally ---
    edge_batch_t current_batch;
#pragma HLS ARRAY_PARTITION variable = current_batch.weights complete dim = 0
#pragma HLS ARRAY_PARTITION variable =                                         \
    current_batch.src_distances complete dim = 0
#pragma HLS ARRAY_PARTITION variable = current_batch.dst_ids complete dim = 0
#pragma HLS dependence variable = current_batch inter false direction = WAW
    current_batch.end_pos = 0;
    current_batch.end_flag = false;

    edge_descriptor_batch_t edge_batch;
    edge_batch.end_pos = 0;
    int edge_batch_pos = 0;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
#pragma HLS dependence variable = edge_batch inter false

    node_distance_burst_t node_distance_burst;
#pragma HLS ARRAY_PARTITION variable = node_distance_burst.data complete dim = 0
#pragma HLS dependence variable = node_distance_burst inter false

    int start_edge_idx, end_edge_idx;
    start_edge_idx = src_offsets_cache_stream.read();

    int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
LOOP_EDGE_PROP_LOADER_MAX_BURST_891:
    for (int node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         ++node_burst_idx) {
        node_distance_burst = node_distance_burst_stream.read();
        int32_t base_idx = node_burst_idx << LOG_PE_NUM;
    LOOP_EDGE_PROP_LOADER_PROCESS_NODES_894:
        for (int pe_idx = 0; pe_idx < PE_NUM; ++pe_idx) {
            int32_t u = base_idx + pe_idx;
            if (u >= num_nodes) {
                break;
            }
            int32_t src_dist = node_distance_burst.data[pe_idx];
            // int start_edge_idx = src_offsets_cache[u];
            // int end_edge_idx = src_offsets_cache[u + 1];
            end_edge_idx = src_offsets_cache_stream.read();

        LOOP_EDGE_PROP_LOADER_PROCESS_EDGES_900:
            for (int e_idx = start_edge_idx; e_idx < end_edge_idx; ++e_idx) {
    #pragma HLS PIPELINE II = 1
                if (edge_batch_pos == edge_batch.end_pos) {
                    edge_batch = edge_stream.read();
                    edge_batch_pos = 0;
                }
                edge_descriptor_t edge = edge_batch.edges[edge_batch_pos++];
                int batch_idx = current_batch.end_pos;
                current_batch.weights[batch_idx] = edge.weight;
                current_batch.src_distances[batch_idx] = src_dist;
                current_batch.dst_ids[batch_idx] = edge.dst_id;
                current_batch.end_pos++;

                // ap_fixed<32, 16> src_dist_fp = *reinterpret_cast<ap_fixed<32, 16>
                // *>(&src_dist); ap_fixed<32, 16> weight_fp =
                // *reinterpret_cast<ap_fixed<32, 16> *>(&edge.weight);

                // printf("DEBUG: Edge (src=%d, dst=%d, weight=%.2f) with src_dist=%.2f\n",
                //        u, edge.dst_id, (float)weight_fp, (float)src_dist_fp);

                if (current_batch.end_pos == PE_NUM) {
                    response_stream.write(current_batch);
                    current_batch.end_pos = 0;
                }
            }

            start_edge_idx = end_edge_idx;
        }
    }

    // --- PHASE D: Send the final batch (can be partial) with end flag ---
    current_batch.end_flag = true;
    response_stream.write(current_batch);
}

// --- PHASE 2.3: UMC Sub-module: Node Property Responder ---

/**
 * @brief Responds to requests for node distances from the URAM cache.
 *
 * @param node_distance_cache On-chip URAM cache (read-only).
 * @param num_nodes           Total number of nodes.
 * @param response_stream     Stream of batched node distance responses.
 */
static void
node_property_responder(hls::stream<node_distance_burst_t> &node_distance_burst_stream,
                        int num_nodes,
                        hls::stream<struct_ibu_14_t> &all_distances_stream) {
    /**
     * @brief Proactively broadcasts all node distances in ascending order of
     * node ID.
     */
    struct_ibu_14_t dist_batch;
    dist_batch.end_pos = 0;
    dist_batch.end_flag = false;

// BROADCAST_LOOP:
// LOOP_NODE_PROP_RESPONDER_BROADCAST_936:
//     for (int i = 0; i < num_nodes; i++) {
// #pragma HLS PIPELINE II = 1
//         dist_batch.data[dist_batch.end_pos] = node_distance_cache[i];
//         dist_batch.end_pos++;
//         if (dist_batch.end_pos == PE_NUM) {
//             all_distances_stream.write(dist_batch);
//             dist_batch.end_pos = 0;
//         }
//     }
    int32_t max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
LOOP_NODE_PROP_RESPONDER_MAX_BURST_940:
    for (int node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         ++node_burst_idx) {
#pragma HLS PIPELINE II = 1
        node_distance_burst_t node_distance_burst = node_distance_burst_stream.read();
        int32_t base_idx = node_burst_idx << LOG_PE_NUM;
    LOOP_NODE_PROP_RESPONDER_PROCESS_NODES_943:
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; ++pe_idx) {
#pragma HLS UNROLL
            int32_t node_idx = base_idx + pe_idx;
            if (node_idx >= num_nodes) {
                break;
            }
            dist_batch.data[dist_batch.end_pos] = node_distance_burst.data[pe_idx];
            dist_batch.end_pos++;
        }
        all_distances_stream.write(dist_batch);
        dist_batch.end_pos = 0;
    }

    // Send the final batch (can be partial) with the end flag set.
    dist_batch.end_flag = true;
    all_distances_stream.write(dist_batch);
}

// --- PHASE 2.4: UMC Top-Level Dataflow Function ---

void memory_loader(
    int instantiate_idx,
    const int *src_offsets, const edge_des_burst_t *edge_des_bursts,
    const int *node_distances, int num_nodes, int num_edges,
    hls::stream<edge_batch_t> &response_to_318,
    hls::stream<struct_ibu_14_t> &all_node_distances_to_343) {
#pragma HLS function_instantiate variable = instantiate_idx
#pragma HLS DATAFLOW

    // On-chip caches
//     static int32_t node_distance_cache_for_edge_loader[MAX_NUM];
// #pragma HLS BIND_STORAGE variable = node_distance_cache_for_edge_loader type = \
//     RAM_T2P impl = URAM
//     static int32_t node_distance_cache_for_responder[MAX_NUM];
// #pragma HLS BIND_STORAGE variable = node_distance_cache_for_responder type =   \
//     RAM_T2P impl = URAM
    hls::stream<node_distance_burst_t> node_distance_burst_stream_0;
#pragma HLS STREAM variable = node_distance_burst_stream_0 depth = 12
    hls::stream<node_distance_burst_t> node_distance_burst_stream_1;
#pragma HLS STREAM variable = node_distance_burst_stream_1 depth = 12

    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 12

    hls::stream<int> src_offsets_cache_stream;
#pragma HLS STREAM variable = src_offsets_cache_stream depth = 32

    // Internal Signal Stream
//     static hls::stream<bool> node_loader_finished;
// #pragma HLS STREAM variable = node_loader_finished depth = 2

    src_offset_loader(src_offsets, src_offsets_cache_stream, num_nodes);

    node_property_loader(node_distances, node_distance_burst_stream_0,
                         node_distance_burst_stream_1, num_nodes);

    edge_descriptor_loader(edge_des_bursts, edge_stream, num_edges);

    edge_property_loader_and_dispatcher(
        src_offsets_cache_stream, edge_stream, node_distance_burst_stream_0,
        num_nodes, response_to_318);

    node_property_responder(node_distance_burst_stream_1, num_nodes,
                            all_node_distances_to_343);
}

static void final_convert(
    hls::stream<struct_sbu_19_t> &in_stream,
    hls::stream<KernelOutputBatch> &converted_stream) {
LOOP_FINAL_CONVERT_1033:
    while (true) {
#pragma HLS PIPELINE II=1
        struct_sbu_19_t in_batch = in_stream.read();
        KernelOutputBatch out_batch;
        out_batch.end_flag = in_batch.end_flag;
        out_batch.end_pos = in_batch.end_pos;
    LOOP_FINAL_CONVERT_UNROLL_1038:
        for (int i = 0; i < PE_NUM; ++i) {
#pragma HLS UNROLL
            ap_fixed<32, 16> dist_fp =
                *reinterpret_cast<ap_fixed<32, 16> *>(&in_batch.data[i].ele_0);
            out_batch.data[i].distance = (float)dist_fp;
            out_batch.data[i].id = in_batch.data[i].ele_1;
        }
        converted_stream.write(out_batch);
        if (in_batch.end_flag) {
            break;
        }
    }
}

static void final_write(
    hls::stream<KernelOutputBatch> &converted_stream,
    KernelOutputBatch *out_o_0_342) {
    #pragma HLS dependence variable=out_o_0_342 inter false
    int32_t i = 0;
LOOP_FINAL_WRITE_1055:
    while (true) {
#pragma HLS PIPELINE II=1
        KernelOutputBatch out_batch = converted_stream.read();
        out_o_0_342[i] = out_batch;
        if (out_batch.end_flag) {
            break;
        }
        i = (i + 1);
    }
}

void final_writeback(
    int instantiate_idx,
    hls::stream<struct_sbu_19_t> &in_stream,
    KernelOutputBatch *out_o_0_342) {
#pragma HLS function_instantiate variable = instantiate_idx
#pragma HLS DATAFLOW
    hls::stream<KernelOutputBatch> converted_stream;
#pragma HLS STREAM variable = converted_stream depth = 12

    final_convert(in_stream, converted_stream);
    final_write(converted_stream, out_o_0_342);
}

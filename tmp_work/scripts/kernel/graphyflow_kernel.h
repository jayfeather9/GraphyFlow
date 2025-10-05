#ifndef __GRAPHYFLOW_KERNEL_H__
#define __GRAPHYFLOW_KERNEL_H__

#include <ap_fixed.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PE_NUM 8
#define LOG_PE_NUM 3
#define MAX_NUM 32768
#define L 4

#define AXI_BUS_WIDTH 512
#define DATA_TYPE_WIDTH 32
#define NUM_WORDS_PER_BUS (AXI_BUS_WIDTH / DATA_TYPE_WIDTH)
// NEED TO ASSERT NUM_WORDS_PER_BUS % PE_NUM == 0 IN PYTHON SCRIPT
// AND AXI_BUS_WIDTH % DATA_TYPE_WIDTH == 0

// --- Graph Type Definitions ---
typedef uint16_t edge_id_t;
typedef uint16_t node_id_t;
typedef uint32_t ap_fixed_pod_t;

/**
 * @brief A structure to hold a batch of edge properties ready for processing.
 * This is the data packet sent from the UMC to the downstream modules.
 */
struct __attribute__((packed)) edge_batch_t {
    int32_t weights[PE_NUM];
    int32_t src_distances[PE_NUM];
    node_id_t dst_ids[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

// Describes a single edge in CSR format for the host and kernel
struct __attribute__((packed)) edge_descriptor_t {
    node_id_t dst_id;
    int32_t weight;
};

struct __attribute__((packed)) edge_des_burst_t {
    edge_descriptor_t edges[PE_NUM];
};

struct __attribute__((packed)) node_distance_burst_t {
    int32_t data[PE_NUM];
};

struct __attribute__((packed)) edge_descriptor_batch_t {
    edge_descriptor_t edges[PE_NUM];
    uint8_t end_pos;
};

// --- Struct Type Definitions ---
struct __attribute__((packed)) struct_ebu_4_t {
    edge_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ini_7_t {
    int32_t ele_0;
    node_id_t ele_1;
    int32_t ele_2;
};

struct __attribute__((packed)) struct_ibu_14_t {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_nbu_16_t {
    node_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_in_17_t {
    int32_t ele_0;
    node_id_t ele_1;
};

struct __attribute__((packed)) struct_bbu_21_t {
    bool data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    int32_t id;
};

struct __attribute__((packed)) opt_struct_ini_7_t_t {
    struct_ini_7_t data;
    bool valid;
};

struct __attribute__((packed)) struct_sbu_12_t {
    struct_ini_7_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_sbu_19_t {
    struct_in_17_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) kt_pair_141_t {
    int32_t key;
    struct_in_17_t transform;
};

struct __attribute__((packed)) struct_sb_38_t {
    struct_in_17_t ele_0;
    bool ele_1;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_obu_10_t {
    opt_struct_ini_7_t_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) net_wrapper_kt_pair_141_t_t {
    kt_pair_141_t data;
    bool end_flag;
};

struct __attribute__((packed)) struct_kbu_30_t {
    kt_pair_141_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

// --- AXI Stream Type Definitions and Conversion Functions ---

#define PAD8(x) (((x) + 7) & ~7)

// ap_axiu<width, user, id, dest>
#define EDGE_BATCH_T_WIDTH (PAD8(PE_NUM * 32 + PE_NUM * 32 + PE_NUM * 16 + 1 + 8))
typedef ap_axiu<EDGE_BATCH_T_WIDTH, 0, 0, 0> edge_batch_axi_t;

inline edge_batch_t axi2ori(edge_batch_axi_t axi) {
    edge_batch_t ori;
    ap_uint<EDGE_BATCH_T_WIDTH> data = axi.data;
    int offset = 0;
    for (int i = 0; i < PE_NUM; ++i) {
        ori.weights[i] = data.range(offset + 31, offset);
        offset += 32;
    }
    for (int i = 0; i < PE_NUM; ++i) {
        ori.src_distances[i] = data.range(offset + 31, offset);
        offset += 32;
    }
    for (int i = 0; i < PE_NUM; ++i) {
        ori.dst_ids[i] = data.range(offset + 15, offset);
        offset += 16;
    }
    ori.end_flag = data.range(offset, offset);
    offset += 1;
    ori.end_pos = data.range(offset + 7, offset);
    return ori;
}

inline edge_batch_axi_t ori2axi(edge_batch_t ori) {
    edge_batch_axi_t axi;
    ap_uint<EDGE_BATCH_T_WIDTH> data = 0;
    int offset = 0;
    for (int i = 0; i < PE_NUM; ++i) {
        data.range(offset + 31, offset) = ori.weights[i];
        offset += 32;
    }
    for (int i = 0; i < PE_NUM; ++i) {
        data.range(offset + 31, offset) = ori.src_distances[i];
        offset += 32;
    }
    for (int i = 0; i < PE_NUM; ++i) {
        data.range(offset + 15, offset) = ori.dst_ids[i];
        offset += 16;
    }
    data.range(offset, offset) = ori.end_flag;
    offset += 1;
    data.range(offset + 7, offset) = ori.end_pos;
    axi.data = data;
    return axi;
}

#define STRUCT_IBU_14_T_WIDTH (PAD8(PE_NUM * 32 + 1 + 8))
typedef ap_axiu<STRUCT_IBU_14_T_WIDTH, 0, 0, 0> struct_ibu_14_axi_t;

inline struct_ibu_14_t axi2ori(struct_ibu_14_axi_t axi) {
    struct_ibu_14_t ori;
    ap_uint<STRUCT_IBU_14_T_WIDTH> data = axi.data;
    int offset = 0;
    for (int i = 0; i < PE_NUM; ++i) {
        ori.data[i] = data.range(offset + 31, offset);
        offset += 32;
    }
    ori.end_flag = data.range(offset, offset);
    offset += 1;
    ori.end_pos = data.range(offset + 7, offset);
    return ori;
}

inline struct_ibu_14_axi_t ori2axi(struct_ibu_14_t ori) {
    struct_ibu_14_axi_t axi;
    ap_uint<STRUCT_IBU_14_T_WIDTH> data = 0;
    int offset = 0;
    for (int i = 0; i < PE_NUM; ++i) {
        data.range(offset + 31, offset) = ori.data[i];
        offset += 32;
    }
    data.range(offset, offset) = ori.end_flag;
    offset += 1;
    data.range(offset + 7, offset) = ori.end_pos;
    axi.data = data;
    return axi;
}

#define STRUCT_SBU_19_T_WIDTH (PAD8(PE_NUM * (32 + 16) + 1 + 8))
typedef ap_axiu<STRUCT_SBU_19_T_WIDTH, 0, 0, 0> struct_sbu_19_axi_t;

inline struct_sbu_19_t axi2ori(struct_sbu_19_axi_t axi) {
    struct_sbu_19_t ori;
    ap_uint<STRUCT_SBU_19_T_WIDTH> data = axi.data;
    int offset = 0;
    for (int i = 0; i < PE_NUM; ++i) {
        ori.data[i].ele_0 = data.range(offset + 31, offset);
        offset += 32;
        ori.data[i].ele_1 = data.range(offset + 15, offset);
        offset += 16;
    }
    ori.end_flag = data.range(offset, offset);
    offset += 1;
    ori.end_pos = data.range(offset + 7, offset);
    return ori;
}

inline struct_sbu_19_axi_t ori2axi(struct_sbu_19_t ori) {
    struct_sbu_19_axi_t axi;
    ap_uint<STRUCT_SBU_19_T_WIDTH> data = 0;
    int offset = 0;
    for (int i = 0; i < PE_NUM; ++i) {
        data.range(offset + 31, offset) = ori.data[i].ele_0;
        offset += 32;
        data.range(offset + 15, offset) = ori.data[i].ele_1;
        offset += 16;
    }
    data.range(offset, offset) = ori.end_flag;
    offset += 1;
    data.range(offset + 7, offset) = ori.end_pos;
    axi.data = data;
    return axi;
}

// --- Function Prototypes ---
void Reduc_141_pre_process(
    hls::stream<struct_ibu_14_t> &i_global_data_0,
    hls::stream<struct_nbu_16_t> &i_global_data_1,
    hls::stream<struct_ibu_14_t> &i_global_data_2,
    hls::stream<struct_ibu_14_t> &i_global_data_3,
    hls::stream<struct_ibu_14_t> &intermediate_key,
    hls::stream<struct_sbu_19_t> &intermediate_transform);
void Reduc_141_unit_reduce(
    hls::stream<net_wrapper_kt_pair_141_t_t> (&kt_wrap_item)[PE_NUM],
    hls::stream<struct_sbu_19_t> &o_0);
void Colle_65(hls::stream<struct_obu_10_t> &i_0,
              hls::stream<struct_sbu_12_t> &o_0);
void Scatt_270(hls::stream<struct_sbu_12_t> &i_0,
               hls::stream<struct_ibu_14_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1,
               hls::stream<struct_ibu_14_t> &o_2);
void Memor_267(hls::stream<struct_ibu_14_t> &o_0_node_id,
               hls::stream<struct_nbu_16_t> &i_0_node_id);
void CopyC_283(hls::stream<struct_nbu_16_t> &i_0,
               hls::stream<struct_nbu_16_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1);
void Condi_61(hls::stream<struct_sbu_12_t> &i_data,
              hls::stream<struct_bbu_21_t> &i_cond,
              hls::stream<struct_obu_10_t> &o_0);
void fused_op_312(hls::stream<struct_ibu_14_t> &i_0,
                  hls::stream<struct_nbu_16_t> &i_1,
                  hls::stream<struct_ibu_14_t> &i_2,
                  hls::stream<struct_bbu_21_t> &o_0,
                  hls::stream<struct_sbu_12_t> &o_1);
/**
 * @brief Acts as a client to the UMC to fetch edge data.
 * It initiates the edge processing pipeline by sending requests to the UMC
 * and unpacks the received edge batches for downstream modules.
 *
 * @param o_0_edge_weight         Output stream for edge weights.
 * @param o_0_edge_src_distance   Output stream for source node distances.
 * @param o_0_edge_dst            Output stream for destination node IDs.
 * @param request_to_umc          Stream to send requests for edge batches to
 * the UMC.
 * @param response_from_umc       Stream to receive processed edge batches from
 * the UMC.
 */
void Memor_318(hls::stream<struct_ibu_14_t> &o_0_edge_weight,
               hls::stream<struct_ibu_14_t> &o_0_edge_src_distance,
               hls::stream<struct_nbu_16_t> &o_0_edge_dst,
               hls::stream<edge_batch_t> &response_from_umc);
void CopyC_350(hls::stream<struct_nbu_16_t> &i_0,
               hls::stream<struct_nbu_16_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1);
void Memor_343(hls::stream<struct_ibu_14_t> &o_0_node_distance,
               hls::stream<struct_nbu_16_t> &i_0_node_id,
               hls::stream<struct_ibu_14_t> &all_node_distances_from_umc);
void fused_op_338(hls::stream<struct_ibu_14_t> &i_0,
                  hls::stream<struct_ibu_14_t> &i_1,
                  hls::stream<struct_nbu_16_t> &i_2,
                  hls::stream<struct_sbu_19_t> &o_0);
void Scatt_346(hls::stream<struct_sbu_19_t> &i_0,
               hls::stream<struct_ibu_14_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1);

/**
 * @brief Unified Memory Controller (UMC) top-level dataflow function.
 * Orchestrates loading graph data from DDR into on-chip memory and serves
 * requests from other processing modules.
 *
 * @param src_offsets       DDR pointer to CSR source offsets.
 * @param edge_descriptors  DDR pointer to edge data.
 * @param node_distances    DDR pointer to node distances.
 * @param num_nodes         Total number of nodes.
 * @param response_to_318   Stream to send processed edge batches.
 * @param response_to_343   Stream to send node distances.
 */
// void UnifiedMemoryController(
//     const int *src_offsets, const edge_des_burst_t *edge_des_bursts,
//     const int *node_distances, int num_nodes, int num_edges,
//     hls::stream<edge_batch_t> &response_to_318,
//     hls::stream<struct_ibu_14_t> &all_node_distances_to_343);

// static void graphyflow_dataflow(
//     // UMC inputs from DDR
//     const int *src_offsets, const edge_des_burst_t *edge_des_bursts,
//     const int *node_distances, int num_nodes, int num_edges,
//     // Final output stream
//     hls::stream<struct_sbu_19_t> &o_0_342_stream);

extern "C" void graphyflow_kernel(
    // input stream
    hls::stream<edge_batch_axi_t> &umc_edge_resp_stream_axi,
    hls::stream<struct_ibu_14_axi_t> &umc_all_node_distances_stream_axi,
    // Final output stream
    hls::stream<struct_sbu_19_axi_t> &o_0_342_stream_axi);

// --- Top-Level Function Prototype ---
/**
 * @brief Top-level kernel function for the graph processing accelerator.
 * @param src_offsets       Pointer to the CSR offsets array for source nodes.
 * @param edge_descriptors  Pointer to the array of edge data (destination and
 * weight).
 * @param node_distances    Pointer to the array of node distances (readable and
 * writable).
 * @param num_nodes         The total number of nodes in the graph.
 */
// extern "C" void graphyflow(const int *src_offsets,
//                            const edge_des_burst_t *edge_des_bursts,
//                            int *node_distances, int num_nodes, int num_edges,
//                            KernelOutputBatch *o_0_342);

#endif // __GRAPHYFLOW_KERNEL_H__

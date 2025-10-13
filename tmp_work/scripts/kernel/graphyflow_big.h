#ifndef __GRAPHYFLOW_GRAPHYFLOW_BIG_H__
#define __GRAPHYFLOW_GRAPHYFLOW_BIG_H__

#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <stdint.h>
#include <ap_axi_sdata.h>
#include <string.h>

#define PE_NUM 8
#define LOG_PE_NUM 3
#define MAX_NUM 32768
#define L 4

// --- New Bitwidth Definitions for HLS Synthesis ---
#define NODE_ID_BITWIDTH 24
#define DISTANCE_BITWIDTH 24
#define DISTANCE_INTEGER_PART 8
#define WEIGHT_BITWIDTH 24
#define WEIGHT_INTEGER_PART 8

// --- New Memory Word and Bus Definitions ---
#define AXI_BUS_WIDTH 512
#define DATA_TYPE_WIDTH 32 // Kept from original for some legacy calculations, may be removed later.
#define NUM_WORDS_PER_BUS (AXI_BUS_WIDTH / DATA_TYPE_WIDTH)

#define REDUCE_MEM_WIDTH 72
typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;
typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;

// --- New Packing-related Constants ---
// Number of distances that can be packed into a single reduce memory word.
#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)


// --- Redefinition of Core Graph Types for HLS ---
// These typedefs override the standard integer types from common.h for synthesis.
typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;
typedef ap_uint<32> edge_id_t; // edge_id_t is not customized yet, keep as is.
typedef ap_uint<DISTANCE_BITWIDTH> ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types
typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;

// --- Struct Type Definitions (UNCHANGED) ---
// The definitions of these structs remain the same, but the underlying
// types (node_id_t, ap_fixed_pod_t) are now custom-width, not uint32_t.
struct __attribute__((packed)) struct_ana_3_t {
    ap_fixed_pod_t ele_0;
    node_id_t ele_1;
    ap_fixed_pod_t ele_2;
};

struct __attribute__((packed)) struct_abu_9_t {
    ap_fixed_pod_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_nbu_11_t {
    node_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ibu_14_t {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_an_15_t {
    ap_fixed_pod_t ele_0;
    node_id_t ele_1;
};

struct __attribute__((packed)) struct_ebu_20_t {
    edge_id_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) node_with_prop_t {
    ap_fixed_pod_t prop;
    node_id_t node_id;
};

struct __attribute__((packed)) node_distance_burst_t {
    ap_fixed_pod_t data[PE_NUM];
};

struct __attribute__((packed)) edge_batch_t {
    ap_fixed_pod_t weights[PE_NUM];
    ap_fixed_pod_t src_distances[PE_NUM];
    node_id_t dsts[PE_NUM];
    int32_t end_pos;
    bool end_flag;
};

struct __attribute__((packed)) node_dist_batch_t {
    ap_fixed_pod_t data[PE_NUM];
    uint8_t end_pos;
    bool end_flag;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    node_id_t id;
};

struct __attribute__((packed)) struct_sbu_7_t {
    struct_ana_3_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_sbu_17_t {
    struct_an_15_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) internal_end_data_batch_t {
    node_with_prop_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) kt_pair_105_t {
    int32_t key;
    node_with_prop_t transform;
};

struct __attribute__((packed)) struct_nb_58_t {
    node_with_prop_t ele_0;
    bool ele_1;
};

struct __attribute__((packed)) edge_des_burst_t {
    node_with_prop_t edges[PE_NUM];
};

struct __attribute__((packed)) edge_descriptor_batch_t {
    node_with_prop_t edges[PE_NUM];
    int32_t end_pos;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_kbu_50_t {
    kt_pair_105_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) net_wrapper_kt_pair_105_t_t {
    kt_pair_105_t data;
    bool end_flag;
};

// --- Function Prototypes ---
static void node_property_loader(const int32_t* node_distances_ddr, hls::stream<node_distance_burst_t> &node_distance_burst_stream_0, hls::stream<node_distance_burst_t> &node_distance_burst_stream_1, int32_t num_nodes);
static void edge_descriptor_loader(const edge_des_burst_t* edge_des_bursts, hls::stream<edge_descriptor_batch_t> &edge_stream, int32_t num_edges);
static void src_offset_loader(const int32_t* src_offsets_ddr, hls::stream<int32_t> &src_offsets_stream, int32_t num_nodes);
static void edge_property_loader_and_dispatcher(hls::stream<int32_t> &src_offsets_cache_stream, hls::stream<edge_descriptor_batch_t> &edge_stream, hls::stream<node_distance_burst_t> &node_distance_burst_stream, int32_t num_nodes, hls::stream<edge_batch_t> &response_stream);
static void node_property_responder(hls::stream<node_distance_burst_t> &node_distance_burst_stream, int32_t num_nodes, hls::stream<node_dist_batch_t> &all_distances_stream);
static void final_convert(hls::stream<internal_end_data_batch_t> &in_stream, hls::stream<KernelOutputBatch> &converted_stream);
static void final_write(hls::stream<KernelOutputBatch> &converted_stream, KernelOutputBatch* out_o_0_342);
static void Reduc_105_pre_process(hls::stream<struct_ibu_14_t> &i_global_data_0, hls::stream<struct_nbu_11_t> &i_global_data_1, hls::stream<struct_abu_9_t> &i_global_data_2, hls::stream<struct_abu_9_t> &i_global_data_3, hls::stream<struct_ibu_14_t> &intermediate_key, hls::stream<internal_end_data_batch_t> &intermediate_transform);
static void Reduc_105_unit_reduce(hls::stream<net_wrapper_kt_pair_105_t_t> (&kt_wrap_item)[PE_NUM], hls::stream<internal_end_data_batch_t> &o_0);
static void Scatt_234(hls::stream<struct_sbu_7_t> &i_0, hls::stream<struct_abu_9_t> &o_0, hls::stream<struct_nbu_11_t> &o_1, hls::stream<struct_abu_9_t> &o_2);
static void Memor_231(hls::stream<struct_ibu_14_t> &o_0_node_id, hls::stream<struct_nbu_11_t> &i_0_node_id);
static void CopyC_247(hls::stream<struct_nbu_11_t> &i_0, hls::stream<struct_nbu_11_t> &o_0, hls::stream<struct_nbu_11_t> &o_1);
static void fused_op_269(hls::stream<struct_abu_9_t> &i_0, hls::stream<struct_nbu_11_t> &i_1, hls::stream<struct_abu_9_t> &i_2, hls::stream<struct_sbu_7_t> &o_0);
static void Memor_274(hls::stream<edge_batch_t> &i_0_edge_id, hls::stream<struct_abu_9_t> &o_0_edge_src_distance, hls::stream<struct_nbu_11_t> &o_0_edge_dst, hls::stream<struct_abu_9_t> &o_0_edge_weight);
static void Memor_299(hls::stream<node_dist_batch_t> &i_all_node_distances, hls::stream<struct_abu_9_t> &o_0_node_distance, hls::stream<struct_nbu_11_t> &i_0_node_id);
static void Scatt_302(hls::stream<internal_end_data_batch_t> &i_0, hls::stream<struct_abu_9_t> &o_0, hls::stream<struct_nbu_11_t> &o_1);
static void CopyC_306(hls::stream<struct_nbu_11_t> &i_0, hls::stream<struct_nbu_11_t> &o_0, hls::stream<struct_nbu_11_t> &o_1);
static void fused_op_294(hls::stream<struct_abu_9_t> &i_0, hls::stream<struct_abu_9_t> &i_1, hls::stream<struct_nbu_11_t> &i_2, hls::stream<internal_end_data_batch_t> &o_0);
static void memory_loader(int32_t instantiate_idx, const int32_t* src_offsets, const edge_des_burst_t* edge_des_bursts, const int32_t* node_distances, int32_t num_nodes, int32_t num_edges, hls::stream<edge_batch_t> &response_to_318, hls::stream<node_dist_batch_t> &all_node_distances_to_343);
static void graphyflow_big_dataflow(hls::stream<edge_batch_t> &response_to_318, hls::stream<node_dist_batch_t> &all_node_distances_to_343, hls::stream<internal_end_data_batch_t> &internal_end_stream);
static void final_writeback(int32_t instantiate_idx, hls::stream<internal_end_data_batch_t> &internal_end_stream, KernelOutputBatch* out_o_0_342);

// --- Top-Level Function Prototype ---
extern "C" void graphyflow_big(
    const int32_t* src_offsets,
    const edge_des_burst_t* edge_des_bursts,
    const int32_t* node_distances,
    KernelOutputBatch* o_0_342,
    int32_t num_nodes,
    int32_t num_edges
);

#endif // __GRAPHYFLOW_GRAPHYFLOW_BIG_H__

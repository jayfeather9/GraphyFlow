#ifndef __GRAPHYFLOW_GRAPHYFLOW_H__
#define __GRAPHYFLOW_GRAPHYFLOW_H__

#include <ap_fixed.h>
#include <hls_stream.h>
#include <stdint.h>

#include <string.h>

#define PE_NUM 8
#define MAX_NUM 32768
#define L 4

// --- Graph Type Definitions ---
typedef uint16_t edge_id_t;
typedef uint16_t node_id_t;

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
void Memor_318(hls::stream<struct_ibu_14_t> &o_0_edge_weight,
               hls::stream<struct_ebu_4_t> &i_0_edge_id,
               hls::stream<struct_ibu_14_t> &o_0_edge_src_distance,
               hls::stream<struct_nbu_16_t> &o_0_edge_dst);
void CopyC_350(hls::stream<struct_nbu_16_t> &i_0,
               hls::stream<struct_nbu_16_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1);
void Memor_343(hls::stream<struct_ibu_14_t> &o_0_node_distance,
               hls::stream<struct_nbu_16_t> &i_0_node_id);
void fused_op_338(hls::stream<struct_ibu_14_t> &i_0,
                  hls::stream<struct_ibu_14_t> &i_1,
                  hls::stream<struct_nbu_16_t> &i_2,
                  hls::stream<struct_sbu_19_t> &o_0);
void Scatt_346(hls::stream<struct_sbu_19_t> &i_0,
               hls::stream<struct_ibu_14_t> &o_0,
               hls::stream<struct_nbu_16_t> &o_1);

// --- Top-Level Function Prototype ---
extern "C" void graphyflow(const struct_ebu_4_t *i_0_edge_id_320,
                           KernelOutputBatch *o_0_342, int *stop_flag,
                           uint16_t input_length_in_batches);

#endif // __GRAPHYFLOW_GRAPHYFLOW_H__

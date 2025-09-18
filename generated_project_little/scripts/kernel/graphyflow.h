#ifndef __GRAPHYFLOW_GRAPHYFLOW_H__
#define __GRAPHYFLOW_GRAPHYFLOW_H__

#include <hls_stream.h>
#include <ap_fixed.h>
#include <stdint.h>

#include <string.h>

#define PE_NUM 8
#define MAX_NUM 32768
#define L 4

// --- Struct Type Definitions ---
struct __attribute__((packed)) node_t {
    int32_t distance;
    int32_t id;
};

struct __attribute__((packed)) struct_ibu_10_t {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_bbu_15_t {
    bool data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) KernelOutputData {
    float distance;
    int32_t id;
};

struct __attribute__((packed)) edge_t {
    int32_t weight;
    node_t src;
    node_t dst;
};

struct __attribute__((packed)) struct_nbu_8_t {
    node_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ini_11_t {
    int32_t ele_0;
    node_t ele_1;
    int32_t ele_2;
};

struct __attribute__((packed)) struct_in_19_t {
    int32_t ele_0;
    node_t ele_1;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_ebu_6_t {
    edge_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_sbu_13_t {
    struct_ini_11_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) opt_struct_ini_11_t_t {
    struct_ini_11_t data;
    bool valid;
};

struct __attribute__((packed)) struct_sbu_21_t {
    struct_in_19_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) kt_pair_141_t {
    int32_t key;
    struct_in_19_t transform;
};

struct __attribute__((packed)) struct_sb_34_t {
    struct_in_19_t ele_0;
    bool ele_1;
};

struct __attribute__((packed)) struct_obu_18_t {
    opt_struct_ini_11_t_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) struct_kbu_27_t {
    kt_pair_141_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

// --- Function Prototypes ---
void Reduc_141_pre_process(hls::stream<struct_sbu_13_t> &i_0, hls::stream<struct_ibu_10_t> &intermediate_key, hls::stream<struct_sbu_21_t> &intermediate_transform);
void Reduc_141_unit_reduce(hls::stream<struct_kbu_27_t> &in_kt_pair_stream, hls::stream<struct_sbu_21_t> &o_0);
void Unary_6(hls::stream<struct_ebu_6_t> &i_0, hls::stream<struct_nbu_8_t> &o_0);
void Unary_9(hls::stream<struct_ebu_6_t> &i_0, hls::stream<struct_nbu_8_t> &o_0);
void CopyC_12(hls::stream<struct_ebu_6_t> &i_0, hls::stream<struct_ebu_6_t> &o_0, hls::stream<struct_ebu_6_t> &o_1);
void Unary_16(hls::stream<struct_ebu_6_t> &i_0, hls::stream<struct_ibu_10_t> &o_0);
void CopyC_19(hls::stream<struct_ebu_6_t> &i_0, hls::stream<struct_ebu_6_t> &o_0, hls::stream<struct_ebu_6_t> &o_1);
void Unary_23(hls::stream<struct_nbu_8_t> &i_0, hls::stream<struct_ibu_10_t> &o_0);
void Gathe_27(hls::stream<struct_ibu_10_t> &i_0, hls::stream<struct_nbu_8_t> &i_1, hls::stream<struct_ibu_10_t> &i_2, hls::stream<struct_sbu_13_t> &o_0);
void CopyC_57(hls::stream<struct_sbu_13_t> &i_0, hls::stream<struct_sbu_13_t> &o_0, hls::stream<struct_sbu_13_t> &o_1);
void Scatt_32(hls::stream<struct_sbu_13_t> &i_0, hls::stream<struct_ibu_10_t> &o_2);
void BinOp_48(hls::stream<struct_ibu_10_t> &i_0, hls::stream<struct_bbu_15_t> &o_0);
void Condi_61(hls::stream<struct_sbu_13_t> &i_data, hls::stream<struct_bbu_15_t> &i_cond, hls::stream<struct_obu_18_t> &o_0);
void Colle_65(hls::stream<struct_obu_18_t> &i_0, hls::stream<struct_sbu_13_t> &o_0);
void Scatt_151(hls::stream<struct_sbu_21_t> &i_0, hls::stream<struct_ibu_10_t> &o_0, hls::stream<struct_nbu_8_t> &o_1);
void Unary_161(hls::stream<struct_nbu_8_t> &i_0, hls::stream<struct_ibu_10_t> &o_0);
void BinOp_164(hls::stream<struct_ibu_10_t> &i_0, hls::stream<struct_ibu_10_t> &i_1, hls::stream<struct_ibu_10_t> &o_0);
void CopyC_168(hls::stream<struct_nbu_8_t> &i_0, hls::stream<struct_nbu_8_t> &o_0, hls::stream<struct_nbu_8_t> &o_1);
void Gathe_173(hls::stream<struct_ibu_10_t> &i_0, hls::stream<struct_nbu_8_t> &i_1, hls::stream<struct_sbu_21_t> &o_0);

// --- Top-Level Function Prototype ---
extern "C" void graphyflow(
    const struct_ebu_6_t* i_0_20,
    KernelOutputBatch* o_0_176,
    int* stop_flag,
    uint16_t input_length_in_batches
);

#endif // __GRAPHYFLOW_GRAPHYFLOW_H__

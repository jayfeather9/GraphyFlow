// REPLACE THE ENTIRE FILE CONTENT WITH THIS:

#ifndef __GRAPHYFLOW_GRAPHYFLOW_H__
#define __GRAPHYFLOW_GRAPHYFLOW_H__

// Include the shared data structures
#include "common.h"
#include <hls_stream.h>

#include <ap_fixed.h>
#include <hls_stream.h>
#include <stdint.h>

#define PE_NUM 8
#define MAX_NUM 256
#define L 4

// --- Struct Type Definitions --

typedef struct {
    ap_fixed<32, 16> data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
} struct_abu_11_t;

typedef struct {
    bool data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
} struct_bbu_16_t;

typedef struct {
    int32_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
} struct_ibu_24_t;

typedef struct {
    node_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
} struct_nbu_9_t;

typedef struct {
    ap_fixed<32, 16> ele_0;
    node_t ele_1;
    ap_fixed<32, 16> ele_2;
} struct_ana_12_t;

typedef struct {
    struct_ana_12_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
} struct_sbu_14_t;

typedef struct {
    struct_ana_12_t data;
    bool valid;
} opt_struct_ana_12_t_t;

typedef struct {
    int32_t key;
    struct_an_20_t transform;
} kt_pair_141_t;

typedef struct {
    struct_an_20_t ele_0;
    bool ele_1;
} struct_sb_41_t;

typedef struct {
    opt_struct_ana_12_t_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
} struct_obu_19_t;

typedef struct {
    kt_pair_141_t data;
    bool end_flag;
    uint8_t end_pos;
} net_wrapper_kt_pair_141_t_t;

typedef struct {
    kt_pair_141_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
} struct_kbu_33_t;

// --- Function Prototypes ---
void Reduc_141_pre_process(
    hls::stream<struct_sbu_14_t> &i_0,
    hls::stream<struct_ibu_24_t> &intermediate_key,
    hls::stream<struct_sbu_22_t> &intermediate_transform);
void Reduc_141_unit_reduce(
    hls::stream<net_wrapper_kt_pair_141_t_t> (&kt_wrap_item)[PE_NUM],
    hls::stream<struct_sbu_22_t> &o_0);
void Unary_6(hls::stream<struct_ebu_7_t> &i_0,
             hls::stream<struct_nbu_9_t> &o_0);
void Unary_9(hls::stream<struct_ebu_7_t> &i_0,
             hls::stream<struct_nbu_9_t> &o_0);
void CopyC_12(hls::stream<struct_ebu_7_t> &i_0,
              hls::stream<struct_ebu_7_t> &o_0,
              hls::stream<struct_ebu_7_t> &o_1);
void Unary_16(hls::stream<struct_ebu_7_t> &i_0,
              hls::stream<struct_abu_11_t> &o_0);
void CopyC_19(hls::stream<struct_ebu_7_t> &i_0,
              hls::stream<struct_ebu_7_t> &o_0,
              hls::stream<struct_ebu_7_t> &o_1);
void Unary_23(hls::stream<struct_nbu_9_t> &i_0,
              hls::stream<struct_abu_11_t> &o_0);
void Gathe_27(hls::stream<struct_abu_11_t> &i_0,
              hls::stream<struct_nbu_9_t> &i_1,
              hls::stream<struct_abu_11_t> &i_2,
              hls::stream<struct_sbu_14_t> &o_0);
void CopyC_57(hls::stream<struct_sbu_14_t> &i_0,
              hls::stream<struct_sbu_14_t> &o_0,
              hls::stream<struct_sbu_14_t> &o_1);
void Scatt_32(hls::stream<struct_sbu_14_t> &i_0,
              hls::stream<struct_abu_11_t> &o_2);
void BinOp_48(hls::stream<struct_abu_11_t> &i_0,
              hls::stream<struct_bbu_16_t> &o_0);
void Condi_61(hls::stream<struct_sbu_14_t> &i_data,
              hls::stream<struct_bbu_16_t> &i_cond,
              hls::stream<struct_obu_19_t> &o_0);
void Colle_65(hls::stream<struct_obu_19_t> &i_0,
              hls::stream<struct_sbu_14_t> &o_0);
void Scatt_151(hls::stream<struct_sbu_22_t> &i_0,
               hls::stream<struct_abu_11_t> &o_0,
               hls::stream<struct_nbu_9_t> &o_1);
void Unary_161(hls::stream<struct_nbu_9_t> &i_0,
               hls::stream<struct_abu_11_t> &o_0);
void BinOp_164(hls::stream<struct_abu_11_t> &i_0,
               hls::stream<struct_abu_11_t> &i_1,
               hls::stream<struct_abu_11_t> &o_0);
void CopyC_168(hls::stream<struct_nbu_9_t> &i_0,
               hls::stream<struct_nbu_9_t> &o_0,
               hls::stream<struct_nbu_9_t> &o_1);
void Gathe_173(hls::stream<struct_abu_11_t> &i_0,
               hls::stream<struct_nbu_9_t> &i_1,
               hls::stream<struct_sbu_22_t> &o_0);

// --- Top-Level Function Prototype ---
// The interface is changed to pointers for memory-mapped access
extern "C" void
graphyflow(const struct_ebu_7_t *i_0_20,    // Input from global memory
           KernelOutputBatch *o_0_176,      // Output to global memory
           int *stop_flag,                  // Flag to signal convergence
           uint16_t input_length_in_batches // Number of batches to process
);

#endif // __GRAPHYFLOW_GRAPHYFLOW_H__
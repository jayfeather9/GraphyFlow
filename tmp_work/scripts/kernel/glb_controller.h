#ifndef __GLB_CONTROLLER_H__
#define __GLB_CONTROLLER_H__

#include "graphyflow.h"
#include "mem_controller.h"

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
    hls::stream<edge_batch_t> &edge_batches_1,
    hls::stream<edge_batch_t> &edge_batches_2,
    hls::stream<edge_batch_t> &edge_batches_3,
    hls::stream<edge_batch_t> &edge_batches_4,
    hls::stream<struct_ibu_14_t> &node_distances_1, 
    hls::stream<struct_ibu_14_t> &node_distances_2, 
    hls::stream<struct_ibu_14_t> &node_distances_3, 
    hls::stream<struct_ibu_14_t> &node_distances_4,
    // output i/o
    hls::stream<struct_sbu_19_t> &result_stream_1,
    hls::stream<struct_sbu_19_t> &result_stream_2,
    hls::stream<struct_sbu_19_t> &result_stream_3,
    hls::stream<struct_sbu_19_t> &result_stream_4,
    KernelOutputBatch *output_ptr_1,
    KernelOutputBatch *output_ptr_2,
    KernelOutputBatch *output_ptr_3,
    KernelOutputBatch *output_ptr_4,
    // graph metadata
    int num_nodes, 
    int num_edges
);

#endif // __GLB_CONTROLLER_H__
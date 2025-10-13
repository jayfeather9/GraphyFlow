#ifndef __GRAPHYFLOW_KERNEL_TYPES_H__
#define __GRAPHYFLOW_KERNEL_TYPES_H__

#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <stdint.h>
#include <ap_axi_sdata.h>
#include <string.h>

// --- 1. Customizable Width Macros ---
// Use these macros to easily change data types widths in the future.
#define NODE_ID_WIDTH 24
#define PROP_WIDTH 24
#define PROP_INT_WIDTH 8 // 8 integer bits for a range up to 255

// --- 2. Vectorization / Packing Size ---
#define TRIO_SIZE 3

// --- 3. Kernel-specific Type Definitions ---
// Use a `_k` suffix to distinguish from host-side types if they differ.
typedef ap_uint<NODE_ID_WIDTH> node_id_t_k;
typedef ap_fixed<PROP_WIDTH, PROP_INT_WIDTH> prop_t_k;
typedef ap_uint<PROP_WIDTH> prop_pod_t_k; // POD type for easy bit manipulation

// --- 4. Core Data Structures ---

// Represents a single node with its associated property (e.g., distance).
// Total size: 24 (prop) + 24 (node_id) = 48 bits.
struct __attribute__((packed)) NodeWithProp {
    prop_pod_t_k prop;
    node_id_t_k node_id;
};

// The new fundamental unit of data processing.
// Holds TRIO_SIZE elements to be processed in a vectorized manner.
// Total size: 48 bits/element * 3 elements = 144 bits.
struct __attribute__((packed)) DataTrio {
    NodeWithProp elements[TRIO_SIZE];
};


// --- 5. AXI Bus Configuration ---
#define AXI_BUS_WIDTH 512
#define DATA_TYPE_WIDTH 32 // Kept for legacy calculations, but new logic is based on struct sizes.
#define NUM_WORDS_PER_BUS (AXI_BUS_WIDTH / DATA_TYPE_WIDTH)


// --- 6. Streaming Batch Structures (Updated for DataTrio) ---
// These structures define the data packets that flow between processing modules.
// Each now carries a payload of `DataTrio` for each of the PE_NUM parallel lanes.

struct __attribute__((packed)) Trio_sbu_t { // Carries a trio of (prop, node_id, prop)
    DataTrio data[PE_NUM];
    bool end_flag;
    uint8_t end_pos; // Number of valid PEs
};

struct __attribute__((packed)) Trio_abu_t { // Carries a trio of properties
    DataTrio data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) Trio_nbu_t { // Carries a trio of node_ids
    DataTrio data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) Trio_ibu_t { // Carries a trio of integers (used for keys)
    DataTrio data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

// Represents a batch of edges, now vectorized with DataTrio
struct __attribute__((packed)) edge_batch_t {
    DataTrio weights;
    DataTrio src_distances;
    DataTrio dsts;
    int32_t end_pos; // Number of valid trios in this batch
    bool end_flag;
};

// Represents a batch of node distances from memory
struct __attribute__((packed)) node_dist_batch_t {
    DataTrio data[PE_NUM];
    uint8_t end_pos;
    bool end_flag;
};

// Final kernel output batch structure
struct __attribute__((packed)) KernelOutputData {
    float distance;
    node_id_t_k id;
};

struct __attribute__((packed)) KernelOutputBatch {
    KernelOutputData data[PE_NUM * TRIO_SIZE]; // Flattened output
    bool end_flag;
    uint8_t end_pos; // Number of valid data elements
};

// --- 7. Reduction and Sorting Network Structures ---
struct __attribute__((packed)) kt_pair_t {
    int32_t key;
    DataTrio transform;
};

struct __attribute__((packed)) Trio_nb_t { // (DataTrio, bool) pair
    DataTrio ele_0;
    bool ele_1;
};

struct __attribute__((packed)) Trio_kbu_t { // Batch of (key, transform) pairs
    kt_pair_t data[PE_NUM];
    bool end_flag;
    uint8_t end_pos;
};

struct __attribute__((packed)) net_wrapper_kt_pair_t {
    kt_pair_t data;
    bool end_flag;
};


#endif // __GRAPHYFLOW_KERNEL_TYPES_H__
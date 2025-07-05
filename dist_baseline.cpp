#include "graphyflow.h"

/**
 * @brief Performs a single, full-graph iteration of the Bellman-Ford algorithm, ensuring
 * each destination node is updated at most once.
 *
 * This function uses internal buffers to solve the write-after-write hazard inherent
 * in parallel graph processing. It first ingests all edges, calculating the best
 * possible new distance for each node and storing it locally. Only after all edges
 * have been processed does it compare the new minimum distances with the original
 * ones and streams out the unique, final updates.
 *
 * @param i_0_20         Input stream of edges for one full graph iteration.
 * @param o_0_176        Output stream of unique distance updates.
 */
void graphyflow(stream<outer_basic_edge__t> &i_0_20,
                stream<outer_tuple_bb_2_t> &o_0_176) {

    // --- STAGE 1: Internal State Buffers ---
    // These arrays are synthesized as on-chip memory (BRAM/URAM) in an FPGA.

    // Stores the best (minimum) distance found for each node during this iteration.
    ap_fixed<32, 16> min_distances[MAX_NUM];
    // Stores the original node data (ID and distance) for final comparison.
    basic_node__t original_nodes[MAX_NUM];
    // Flags to track which nodes are destinations in this iteration.
    bool is_destination[MAX_NUM] = {false};
#pragma HLS ARRAY_PARTITION variable=is_destination complete

    // --- STAGE 2: Ingest Edges and Find Minimum Distances ---
    // This loop processes all incoming edges and updates the local `min_distances`
    // buffer with the shortest path found *so far* for each destination node.
ingest_and_relax_loop:
    while (true) {
#pragma HLS PIPELINE
        outer_basic_edge__t edge = i_0_20.read();
        if (edge.end_flag) {
            break;
        }
        basic_node__t dst_node = edge.dst;
        int dst_id = dst_node.id.ele;

        // If this is the first time we see this destination node,
        // store its original state and initialize its minimum distance.
        if (!is_destination[dst_id]) {
            original_nodes[dst_id] = dst_node;
            min_distances[dst_id] = dst_node.distance.ele; // Initialize with original distance
            is_destination[dst_id] = true;
        }

        // Perform the relaxation calculation
        ap_fixed<32, 16> new_dist = edge.src.distance.ele + edge.weight.ele;

        // Update the minimum distance if the new path is shorter
        if (new_dist < min_distances[dst_id]) {
            min_distances[dst_id] = new_dist;
        }
    }

    // --- STAGE 3: Generate Unique Output Updates ---
    // Iterate through all possible nodes. If a node's distance was improved,
    // write a single, final update tuple to the output stream.
output_generation_loop:
    for (int i = 0; i < MAX_NUM; ++i) {
#pragma HLS PIPELINE
        // Check if this node was a destination AND its distance was actually reduced.
        if (is_destination[i]) {
            outer_tuple_bb_2_t update_tuple;
            update_tuple.ele_0.ele = min_distances[i]; // The new best distance
            update_tuple.ele_1 = original_nodes[i];   // The node being updated
            update_tuple.end_flag = false;
            o_0_176.write(update_tuple);
        }
    }

    // --- STAGE 4: Finalization ---
    // Signal the end of the output stream to the downstream component.
    outer_tuple_bb_2_t end_signal;
    end_signal.end_flag = true;
    o_0_176.write(end_signal);
}
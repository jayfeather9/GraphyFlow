#include "graphyflow_big.h"

static void src_id_loader(const bus_word_t *node_ids_ddr,
                          hls::stream<node_id_burst_t> &src_id_burst_stream_1,
                          hls::stream<node_id_burst_t> &src_id_burst_stream_2,
                          int32_t num_nodes) {
    const int num_ids_per_word = AXI_BUS_WIDTH / NODE_ID_BITWIDTH;
    const int num_wide_reads =
        (num_nodes + num_ids_per_word - 1) / num_ids_per_word;

    int nodes_read = 0;
    int burst_idx = 0;
    node_id_burst_t burst1, burst2;
LOOP_SIL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 2
        bus_word_t wide_word = node_ids_ddr[i];

    LOOP_SIL_UNPACK:
        for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
            if (nodes_read + j < num_nodes) {
                node_id_t cur_id = wide_word.range(
                    (j + 1) * NODE_ID_BITWIDTH - 1, j * NODE_ID_BITWIDTH);
                burst1.data[j] = cur_id;
                // printf("Loaded node ID %d at burst %d, position %d\n",
                // (int)cur_id, burst_idx, j); fflush(NULL);
            }
        }
        bool burst2_valid = false;
        for (int j = 8; j < 16; j++) {
#pragma HLS UNROLL
            if (nodes_read + j < num_nodes) {
                burst2.data[j - 8] = wide_word.range(
                    (j + 1) * NODE_ID_BITWIDTH - 1, j * NODE_ID_BITWIDTH);
                burst2_valid |= true;
                // printf("Loaded node ID %d at burst %d, position %d\n",
                // (int)burst2.data[j - 8], burst_idx + 1, j - 8); fflush(NULL);
            }
        }
        src_id_burst_stream_1.write(burst1);
        src_id_burst_stream_2.write(burst1);
        if (burst2_valid) {
            src_id_burst_stream_1.write(burst2);
            src_id_burst_stream_2.write(burst2);
        }
        nodes_read += num_ids_per_word;
    }
}

static void
edge_descriptor_loader(const bus_word_t *edge_props_ddr,
                       hls::stream<node_id_burst_t> &stream_src_ids,
                       hls::stream<edge_descriptor_batch_t> &edge_stream,
                       int32_t num_edges) {
    const int bits_per_edge = NODE_ID_BITWIDTH + WEIGHT_BITWIDTH;
    const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
    const int num_wide_reads =
        (num_edges + edges_per_word - 1) / edges_per_word;

    int edges_read = 0;
    edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
    edge_batch.end_pos = 0;

    node_id_burst_t src_id_burst;
#pragma HLS ARRAY_PARTITION variable = src_id_burst.data complete dim = 0

#if (NODE_ID_BITWIDTH == 32) && (WEIGHT_BITWIDTH == 32)
LOOP_EDL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = edge_props_ddr[i];
    LOOP_EDL_UNPACK:
        for (int j = 0; j < edges_per_word; j++) {
#pragma HLS UNROLL
            if (edges_read + j < num_edges) {
                ap_uint<bits_per_edge> packed_edge = wide_word.range(
                    (j + 1) * bits_per_edge - 1, j * bits_per_edge);
                edge_t edge;
                node_id_t src_id;
                edge.dst_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);
                edge.src_id =
                    packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);
                src_id = edge.src_id;

                edge_batch.edges[j] = edge;
                src_id_burst.data[j] = src_id;
            }
        }
        stream_src_ids.write(src_id_burst);
        edges_read += edges_per_word;
        edge_batch.end_pos = (edges_read <= num_edges)
                                 ? edges_per_word
                                 : (num_edges % edges_per_word);
        edge_stream.write(edge_batch);
        edge_batch.end_pos = 0;
    }
#else
// Add support for other bitwidth combinations if needed.
#error                                                                         \
    "edge_descriptor_loader currently only supports 32-bit node_id and 32-bit weight."
#endif
}

ap_uint<4> count_end_ones(ap_uint<PE_NUM> valid_mask) {
#pragma HLS INLINE
    ap_uint<4> count = 0;
    switch (valid_mask) {
    case 0:
        count = 0;
        break;
    case 1:
        count = 1;
        break;
    case 3:
        count = 2;
        break;
    case 7:
        count = 3;
        break;
    case 15:
        count = 4;
        break;
    case 31:
        count = 5;
        break;
    case 63:
        count = 6;
        break;
    case 127:
        count = 7;
        break;
    case 255:
        count = 8;
        break;
    default:
        break;
    }
    return count;
}

static void
dist_req_packer(hls::stream<node_id_burst_t> &src_id_burst_stream,
                hls::stream<distance_req_pack_t> &distance_req_pack_stream,
                int32_t num_nodes) {

    const int max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_idx_max = 0;

LOOP_DRP_SEND_REQ:
    for (int32_t node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         node_burst_idx += 1) {
#pragma HLS PIPELINE II = 1
        ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cache_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cache_idx complete dim = 0
        node_id_burst_t node_id_burst = src_id_burst_stream.read();
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cache_idx[pe_idx] = node_id_burst.data[pe_idx] >> LOG_DIST_PER_WORD;
            // printf("PE %d requests node ID %d (cache idx %d)\n", pe_idx,
            // (int)node_id_burst.data[pe_idx], (int)cache_idx[pe_idx]);
            // fflush(NULL);
        }

        ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cache_idx_diffs[PE_NUM];
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cache_idx_diffs[pe_idx] = cache_idx[pe_idx] - last_idx_max;
            // printf("PE %d cache idx diff: %d\n", pe_idx,
            // (int)cache_idx_diffs[pe_idx]); fflush(NULL);
        }

        // if not all diffs are zero, send a req_pack
        if (cache_idx_diffs[PE_NUM - 1]) {
            ap_uint<PE_NUM> valid_mask;
            for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
                if (cache_idx_diffs[pe_idx] == 0) {
                    valid_mask[pe_idx] = 1;
                } else {
                    valid_mask[pe_idx] = 0;
                }
            }

            ap_uint<4> num_unread = count_end_ones(valid_mask);
            // printf("Packing req for %d unread PEs\n", (int)num_unread);
            // fflush(NULL);

            distance_req_pack_t req_pack;
            req_pack.offset = num_unread;
            req_pack.end_flag = false;

            for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
                req_pack.node_ids[pe_idx] = node_id_burst.data[pe_idx];
                // printf("Req pack PE %d node ID: %d\n", pe_idx,
                // (int)req_pack.node_ids[pe_idx]); fflush(NULL);
            }

            distance_req_pack_stream.write(req_pack);
        }

        last_idx_max = cache_idx[PE_NUM - 1];
    }

    distance_req_pack_t end_req_pack;
    end_req_pack.end_flag = true;
    end_req_pack.offset = 8;
    distance_req_pack_stream.write(end_req_pack);
}

static void
cacheline_req_sender(hls::stream<distance_req_pack_t> &distance_req_pack_stream,
                     hls::stream<cacheline_req_t> &cacheline_req_stream) {

    cacheline_req_t cache_req;
    cache_req.end_flag = false;
    cache_req.idx = 0;
    cache_req.target_pe = 0;
    cacheline_req_stream.write(cache_req);

    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cacheline_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cacheline_idx complete dim = 0

LOOP_SEND_CACHE_REQ:
    while (true) {
#pragma HLS PIPELINE II = 1
        distance_req_pack_t req_pack = distance_req_pack_stream.read();
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cacheline_idx[pe_idx] =
                req_pack.node_ids[pe_idx] >> LOG_DIST_PER_WORD;
        }

        {
        LOOP_SEND_CACHE_REQ_INNER:
            for (ap_uint<4> i = req_pack.offset; i < PE_NUM; i++) {
#pragma HLS PIPELINE II = 1 rewind
#pragma HLS unroll factor = 1
                cache_req.idx = cacheline_idx[i];
                cache_req.target_pe = i;
                cache_req.end_flag = req_pack.end_flag;
                cacheline_req_stream.write(cache_req);
                // printf("Sent cacheline req for idx %d to PE %d\n",
                // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
            }
        }

        if (req_pack.end_flag) {
            break;
        }
    }
    cache_req.end_flag = true;
    cacheline_req_stream.write(cache_req);
}

// --- 1. Memory Helper Functions ---
// --- MODIFIED: Reads 512-bit words and unpacks 24-bit distance values.
static void node_property_loader(
    const bus_word_t *node_distances_ddr,
    hls::stream<cacheline_req_t> &cacheline_req_stream,
    hls::stream<cacheline_resp_t> &cacheline_resp_stream,
    // hls::stream<node_distance_burst_t> &node_distance_burst_stream_1,
    int32_t num_nodes) {

    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx = -1;
    bus_word_t last_cacheline;
    cacheline_resp_t cache_resp;
    bool end_flag_get = false;

    // Stream 0
LOOP_NPL_S0_READ:
    while (true) {
#pragma HLS PIPELINE II = 1
        if (!cacheline_req_stream.empty()) {
            // printf("Waiting for cacheline request...\n");fflush(NULL);
            cacheline_req_t cache_req = cacheline_req_stream.read();
            // printf("Received cacheline request for idx %d from PE %d\n",
            // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
            if (cache_req.end_flag) {
                cache_resp.end_flag = true;
                end_flag_get = true;
            } else {
                cache_resp.end_flag = false;
                if (cache_req.idx == last_cache_idx) {
                    cache_resp.data = last_cacheline;
                } else {
                    cache_resp.data = node_distances_ddr[cache_req.idx];
                }
            }

            last_cacheline = cache_resp.data;
            last_cache_idx = cache_req.idx;
            cache_resp.target_pe = cache_req.target_pe;
            cacheline_resp_stream.write(cache_resp);
            // printf("Sent cacheline response for idx %d to PE %d\n",
            // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
            if (end_flag_get) {
                break;
            }
        }
    }
}

static void
node_prop_resp_receiver(hls::stream<cacheline_resp_t> &cacheline_resp_stream,
                        hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM]) {

    cacheline_resp_t cache_resp = cacheline_resp_stream.read();
    bus_word_t first_line = cache_resp.data;
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        cacheline_streams[pe_idx].write(first_line);
    }

LOOP_RECEIVE_CACHE_RESP:
    while (true) {
#pragma HLS PIPELINE II = 1
        if (!cacheline_resp_stream.empty()) {
            cache_resp = cacheline_resp_stream.read();
            if (cache_resp.end_flag) {
                break;
            }
            cacheline_streams[cache_resp.target_pe].write(cache_resp.data);
        }
    }
}

ap_fixed_pod_t get_val_from_bus(const bus_word_t bus, int offset) {
#pragma HLS INLINE
    switch (offset) {
    case 0:
        return bus.range(31, 0);
    case 1:
        return bus.range(63, 32);
    case 2:
        return bus.range(95, 64);
    case 3:
        return bus.range(127, 96);
    case 4:
        return bus.range(159, 128);
    case 5:
        return bus.range(191, 160);
    case 6:
        return bus.range(223, 192);
    case 7:
        return bus.range(255, 224);
    case 8:
        return bus.range(287, 256);
    case 9:
        return bus.range(319, 288);
    case 10:
        return bus.range(351, 320);
    case 11:
        return bus.range(383, 352);
    case 12:
        return bus.range(415, 384);
    case 13:
        return bus.range(447, 416);
    case 14:
        return bus.range(479, 448);
    case 15:
        return bus.range(511, 480);
    default:
        return 0;
    }
}

static void
merge_node_props(hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM],
                 hls::stream<edge_descriptor_batch_t> &edge_stream,
                 //  hls::stream<node_id_burst_t> &src_id_burst_stream,
                 hls::stream<edge_batch_t> &edge_batch_stream,
                 uint32_t edge_num) {
    bus_word_t last_cacheline[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = last_cacheline complete dim = 0
    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> last_cache_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = last_cache_idx complete dim = 0

// Init first cacheline for each PE
LOOP_INIT_CACHELINE:
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        last_cacheline[pe_idx] = cacheline_streams[pe_idx].read();
        last_cache_idx[pe_idx] = 0;
    }

    const uint32_t scatter_size = (edge_num + PE_NUM - 1) / PE_NUM;
// printf("Merging node properties for %d edges (%d scatter batches)\n",
// edge_num, scatter_size); fflush(NULL);
LOOP_SCATTER_EDGES:
    for (int32_t edge_batch_idx = 0; edge_batch_idx < scatter_size;
         edge_batch_idx++) {
#pragma HLS PIPELINE II = 1
        edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
        edge_batch = edge_stream.read();
        //         node_id_burst_t src_id_burst = src_id_burst_stream.read();
        // #pragma HLS ARRAY_PARTITION variable = src_id_burst.data complete dim
        // = 0
        edge_batch_t out_batch;
#pragma HLS ARRAY_PARTITION variable = out_batch.src_distances complete dim = 0
#pragma HLS ARRAY_PARTITION variable = out_batch.weights complete dim = 0
#pragma HLS ARRAY_PARTITION variable = out_batch.dsts complete dim = 0
        out_batch.end_flag = false;
        out_batch.end_pos = edge_batch.end_pos;
        bus_word_t cur_last_cacheline;
        ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cur_last_cache_idx;
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cacheline_idx =
                (edge_batch.edges[pe_idx].src_id >> LOG_DIST_PER_WORD);
            uint32_t offset =
                (edge_batch.edges[pe_idx].src_id & (DIST_PER_WORD - 1));
            if (pe_idx < edge_batch.end_pos) {
                bus_word_t cacheline;
                if (cacheline_idx == last_cache_idx[pe_idx]) {
                    cacheline = last_cacheline[pe_idx];
                } else {
                    cacheline = cacheline_streams[pe_idx].read();
                }

                // ap_fixed_pod_t prop = cacheline.range(
                //     31 + (offset << 5), offset << 5);
                ap_fixed_pod_t prop = get_val_from_bus(cacheline, offset);

                out_batch.src_distances[pe_idx] = prop;
                distance_t tmp_dist = 1.0;
                out_batch.weights[pe_idx] =
                    (*reinterpret_cast<ap_fixed_pod_t *>(&tmp_dist));
                out_batch.dsts[pe_idx] = edge_batch.edges[pe_idx].dst_id;

                if (pe_idx == PE_NUM - 1) {
                    cur_last_cacheline = cacheline;
                    cur_last_cache_idx = cacheline_idx;
                }
            }
        }
        edge_batch_stream.write(out_batch);
        // printf("Sent edge batch with %d entries\n", (int)out_batch.end_pos);
        // fflush(NULL);
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            last_cacheline[pe_idx] = cur_last_cacheline;
            last_cache_idx[pe_idx] = cur_last_cache_idx;
        }
    }
    // Send end marker
    edge_batch_t end_batch;
    end_batch.end_flag = true;
    end_batch.end_pos = 0;
    edge_batch_stream.write(end_batch);
}

static void
node_property_responder(hls::stream<node_dist_pkt_t> &node_dist_stream,
                        int32_t num_nodes,
                        hls::stream<node_dist_batch_t> &all_distances_stream) {
    node_dist_batch_t dist_batch;
    dist_batch.end_flag = false;
    int32_t nodes_read = 0;

LOOP_FOR_14:
    while (true) {
#pragma HLS PIPELINE II = 1
        // Read packet from stream
        node_dist_pkt_t node_dist_pkt = node_dist_stream.read();

        // If we hit the end marker, exit the loop
        if (node_dist_pkt.last) {
            break;
        }

    LOOP_FOR_13:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            // Direct indexing with pe_idx to avoid race conditions in UNROLL
            dist_batch.data[pe_idx] =
                node_dist_pkt.data.range((pe_idx + 1) * DISTANCE_BITWIDTH - 1,
                                         pe_idx * DISTANCE_BITWIDTH);
        }
        // Always send PE_NUM distances per batch (the downstream will know how
        // many are valid based on num_nodes)
        dist_batch.end_pos = (nodes_read + PE_NUM < num_nodes)
                                 ? PE_NUM
                                 : (num_nodes - nodes_read);
        nodes_read += dist_batch.end_pos;
        all_distances_stream.write(dist_batch);
    }

    dist_batch.end_flag = true;
    dist_batch.end_pos = 0;
    all_distances_stream.write(dist_batch);
}

// --- REWRITTEN: New final_writeback function packs only distances (no node
// IDs) into 512-bit words. Node IDs are implicit: they are sequential from 0 to
// num_dsts-1.
static void
pack_distances_to_bus_words(hls::stream<internal_end_data_batch_t> &in_stream,
                            hls::stream<write_burst_pkt_t> &output_stream) {
    const int dists_per_batch = PE_NUM; // 8 distances per batch
    ap_uint<256> first_half, second_half;
    bool has_pending_half = false;
    int pkt_idx = 0;

LOOP_PACK_TO_BUS:
    while (true) {
#pragma HLS PIPELINE II = 1

        internal_end_data_batch_t in_batch = in_stream.read();

        if (in_batch.end_pos == 0 && in_batch.end_flag) {
            break;
        }

        // Pack current batch into 256-bit half
        ap_uint<256> current_half;
    LOOP_PACK_BATCH:
        for (int i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            ap_fixed_pod_t distance = in_batch.data[i].prop;
            current_half.range((i + 1) * DISTANCE_BITWIDTH - 1,
                               i * DISTANCE_BITWIDTH) = distance;
        }

        // Alternate between first_half and second_half
        if (!has_pending_half) {
            // Store as first half
            first_half = current_half;
            has_pending_half = true;
        } else {
            // Combine with first_half to form complete bus_word
            second_half = current_half;

            bus_word_t word;
            word.range(255, 0) = first_half;
            word.range(511, 256) = second_half;

            write_burst_pkt_t pkt;
            pkt.data = word;
            pkt.dest = pkt_idx;
            pkt.last = false;
            pkt_idx++;

            output_stream.write(pkt);
            has_pending_half = false;
        }

        if (in_batch.end_flag) {
            break;
        }
    }

    // Handle remaining half (if total_reads is odd)
    if (has_pending_half) {
        bus_word_t word;
        word.range(255, 0) = first_half;
        word.range(511, 256) = 0; // Zero-padding for the second half
        write_burst_pkt_t pkt;
        pkt.data = word;
        pkt.dest = pkt_idx;
        pkt.last = false;
        pkt_idx++;
        output_stream.write(pkt);
    }
    // printf("Finished packing distances to bus words.\n"); fflush(NULL);

    write_burst_pkt_t pkt;
    pkt.last = true;
    output_stream.write(pkt);
}

// Write bus words from stream to DDR memory
// Writes exactly the number of words needed to cover dst_num distances
// static void write_bus_words_to_ddr(hls::stream<bus_word_t> &in_bus_stream,
//                                    bus_word_t *out_ddr, int32_t dst_num) {
//     const int dists_per_word =
//         AXI_BUS_WIDTH / DISTANCE_BITWIDTH; // 16 distances per 512-bit word
//     int total_words = (dst_num + dists_per_word - 1) / dists_per_word;
//     int word_idx = 0;

// LOOP_WRITE_TO_DDR:
//     while (true) {
// #pragma HLS PIPELINE II = 1
//         if (!in_bus_stream.empty()) {
//             bus_word_t word = in_bus_stream.read();
//             out_ddr[word_idx] = word;
//             // printf("Wrote bus word %d to DDR.\n", word_idx); fflush(NULL);
//             // printf("Total words to write: %d\n", total_words);
//             fflush(NULL); word_idx++; if (word_idx >= total_words) {
//                 break;
//             }
//         }
//     }
// }

// --- 2. Utility Network Functions ---
static void stream_zipper_0(
    hls::stream<struct_ibu_14_t> &in_key_batch_stream,
    hls::stream<internal_end_data_batch_t> &in_transform_batch_stream,
    hls::stream<struct_kbu_50_t> &out_pair_batch_stream) {
    struct_ibu_14_t key_batch;
    internal_end_data_batch_t transform_batch;
    struct_kbu_50_t out_batch;
LOOP_WHILE_19:
    while (true) {
#pragma HLS PIPELINE
        key_batch = in_key_batch_stream.read();
        transform_batch = in_transform_batch_stream.read();
    LOOP_FOR_18:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch.data[i].key = key_batch.data[i];
            out_batch.data[i].transform = transform_batch.data[i];
        }
        out_batch.end_flag = key_batch.end_flag;
        out_batch.end_pos = key_batch.end_pos;
        out_pair_batch_stream.write(out_batch);
        if (key_batch.end_flag) {
            break;
        }
    }
}

static void
demux_1(hls::stream<struct_kbu_50_t> &in_batch_stream,
        hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8]) {
    struct_kbu_50_t in_batch;
LOOP_WHILE_22:
    while (true) {
#pragma HLS PIPELINE
        in_batch = in_batch_stream.read();
        net_wrapper_kt_pair_105_t_t wrapper_data;
    LOOP_FOR_20:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if ((i < in_batch.end_pos)) {
                wrapper_data.data = in_batch.data[i];
                wrapper_data.end_flag = false;
                out_streams[i].write(wrapper_data);
            }
        }
        if (in_batch.end_flag) {
            break;
        }
    }
    // Propagate end_flag to all output streams
    net_wrapper_kt_pair_105_t_t end_wrapper;
    end_wrapper.end_flag = true;
LOOP_FOR_21:
    for (uint32_t i = 0; i < 8; i++) {
#pragma HLS UNROLL
        out_streams[i].write(end_wrapper);
    }
}

static void sender_2(int32_t i, hls::stream<net_wrapper_kt_pair_105_t_t> &in1,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &in2,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &out1,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &out2,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &out3,
                     hls::stream<net_wrapper_kt_pair_105_t_t> &out4) {
#pragma HLS function_instantiate variable = i
    bool in1_end_flag = false;
    bool in2_end_flag = false;
LOOP_WHILE_23:
    while (true) {
#pragma HLS PIPELINE II = 1
        if ((!in1.empty())) {
            net_wrapper_kt_pair_105_t_t data1;
            data1 = in1.read();
            if ((!data1.end_flag)) {
                if (((data1.data.key >> i) & 1)) {
                    out2.write(data1);
                } else {
                    out1.write(data1);
                }
            } else {
                in1_end_flag = true;
            }
        }
        if ((!in2.empty())) {
            net_wrapper_kt_pair_105_t_t data2;
            data2 = in2.read();
            if ((!data2.end_flag)) {
                if (((data2.data.key >> i) & 1)) {
                    out4.write(data2);
                } else {
                    out3.write(data2);
                }
            } else {
                in2_end_flag = true;
            }
        }
        if ((in1_end_flag & in2_end_flag)) {
            net_wrapper_kt_pair_105_t_t data;
            data.end_flag = true;
            out1.write(data);
            out2.write(data);
            out3.write(data);
            out4.write(data);
            in1_end_flag = false;
            in2_end_flag = false;
            break;
        }
    }
}

static void receiver_2(int32_t i,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &out1,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &out2,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &in1,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &in2,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &in3,
                       hls::stream<net_wrapper_kt_pair_105_t_t> &in4) {
#pragma HLS function_instantiate variable = i
    bool in1_end_flag = false;
    bool in2_end_flag = false;
    bool in3_end_flag = false;
    bool in4_end_flag = false;
LOOP_WHILE_24:
    while (true) {
#pragma HLS PIPELINE II = 1
        if ((!in1.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in1.read();
            if ((!data.end_flag)) {
                out1.write(data);
            } else {
                in1_end_flag = true;
            }
        } else if ((!in3.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in3.read();
            if ((!data.end_flag)) {
                out1.write(data);
            } else {
                in3_end_flag = true;
            }
        }
        if ((!in2.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in2.read();
            if ((!data.end_flag)) {
                out2.write(data);
            } else {
                in2_end_flag = true;
            }
        } else if ((!in4.empty())) {
            net_wrapper_kt_pair_105_t_t data;
            data = in4.read();
            if ((!data.end_flag)) {
                out2.write(data);
            } else {
                in4_end_flag = true;
            }
        }
        if ((((in1_end_flag & in2_end_flag) & in3_end_flag) & in4_end_flag)) {
            net_wrapper_kt_pair_105_t_t data;
            data.end_flag = true;
            out1.write(data);
            out2.write(data);
            break;
        }
    }
}

static void switch2x2_2(int32_t i,
                        hls::stream<net_wrapper_kt_pair_105_t_t> &in1,
                        hls::stream<net_wrapper_kt_pair_105_t_t> &in2,
                        hls::stream<net_wrapper_kt_pair_105_t_t> &out1,
                        hls::stream<net_wrapper_kt_pair_105_t_t> &out2) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_1;
#pragma HLS STREAM variable = l1_1 depth = 2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_2;
#pragma HLS STREAM variable = l1_2 depth = 2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_3;
#pragma HLS STREAM variable = l1_3 depth = 2
    hls::stream<net_wrapper_kt_pair_105_t_t> l1_4;
#pragma HLS STREAM variable = l1_4 depth = 2
    sender_2(i, in1, in2, l1_1, l1_2, l1_3, l1_4);
    receiver_2(i, out1, out2, l1_1, l1_2, l1_3, l1_4);
}

static void
omega_switch_2(hls::stream<net_wrapper_kt_pair_105_t_t> (&in_streams)[8],
               hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8]) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_0[8];
#pragma HLS STREAM variable = stream_stage_0 depth = 2
    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_1[8];
#pragma HLS STREAM variable = stream_stage_1 depth = 2
    switch2x2_2(2, in_streams[0], in_streams[1], stream_stage_0[0],
                stream_stage_0[1]);
    switch2x2_2(2, in_streams[2], in_streams[3], stream_stage_0[2],
                stream_stage_0[3]);
    switch2x2_2(2, in_streams[4], in_streams[5], stream_stage_0[4],
                stream_stage_0[5]);
    switch2x2_2(2, in_streams[6], in_streams[7], stream_stage_0[6],
                stream_stage_0[7]);
    switch2x2_2(1, stream_stage_0[0], stream_stage_0[4], stream_stage_1[0],
                stream_stage_1[1]);
    switch2x2_2(1, stream_stage_0[1], stream_stage_0[5], stream_stage_1[2],
                stream_stage_1[3]);
    switch2x2_2(1, stream_stage_0[2], stream_stage_0[6], stream_stage_1[4],
                stream_stage_1[5]);
    switch2x2_2(1, stream_stage_0[3], stream_stage_0[7], stream_stage_1[6],
                stream_stage_1[7]);
    switch2x2_2(0, stream_stage_1[0], stream_stage_1[4], out_streams[0],
                out_streams[1]);
    switch2x2_2(0, stream_stage_1[1], stream_stage_1[5], out_streams[2],
                out_streams[3]);
    switch2x2_2(0, stream_stage_1[2], stream_stage_1[6], out_streams[4],
                out_streams[5]);
    switch2x2_2(0, stream_stage_1[3], stream_stage_1[7], out_streams[6],
                out_streams[7]);
}

// --- 3. DFIR Component Functions ---
static void
Reduc_105_pre_process(hls::stream<edge_batch_t> &response_to_318,
                      hls::stream<struct_kbu_50_t> &reduce_105_z2d_pair) {
    edge_batch_t edge_batch_data;
    struct_kbu_50_t out_batch_data;
    bool end_flag;
LOOP_WHILE_26:
    while (true) {
#pragma HLS PIPELINE
        edge_batch_data = response_to_318.read();
    LOOP_FOR_25:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            kt_pair_105_t kt_pair;
            kt_pair.key = edge_batch_data.dsts[i];
            kt_pair.transform.node_id = edge_batch_data.dsts[i];
            ap_fixed_pod_t new_dist;
            distance_t lhs_68 = *reinterpret_cast<distance_t *>(
                &edge_batch_data.src_distances[i]);
            distance_t rhs_68 =
                *reinterpret_cast<distance_t *>(&edge_batch_data.weights[i]);
            distance_t temp_BinOp_68_o_0_ap_result;
            temp_BinOp_68_o_0_ap_result = (lhs_68 + rhs_68);
            ap_fixed_pod_t fused_temp_BinOp_68_o_0 =
                *reinterpret_cast<ap_fixed_pod_t *>(
                    &temp_BinOp_68_o_0_ap_result);
            // Inlining Gathe_179
            kt_pair.transform.prop = fused_temp_BinOp_68_o_0;
            out_batch_data.data[i] = kt_pair;
        }
        out_batch_data.end_flag = edge_batch_data.end_flag;
        out_batch_data.end_pos = edge_batch_data.end_pos;
        reduce_105_z2d_pair.write(out_batch_data);
        end_flag = edge_batch_data.end_flag;
        if (end_flag) {
            break;
        }
    }
}

inline ap_fixed_pod_t get_raw_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> bits;
    switch (idx) {
    case 0:
        bits = word.range(DISTANCE_BITWIDTH - 1, 0);
        break;
    case 1:
        bits = word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH);
        break;
    case 2:
        bits =
            word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1));
        break;
    default:
        bits = 0;
        break;
    }
    return bits;
}

inline distance_t get_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_fixed_pod_t raw_val = get_raw_val(word, idx);
    distance_t val = *reinterpret_cast<distance_t *>(&raw_val);
    return val;
}

inline void set_val(reduce_word_t &word, int idx, distance_t val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits =
        *reinterpret_cast<ap_uint<DISTANCE_BITWIDTH> *>(&val);
    switch (idx) {
    case 0:
        word.range(DISTANCE_BITWIDTH - 1, 0) = val_bits;
        break;
    case 1:
        word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH) = val_bits;
        break;
    case 2:
        word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1)) =
            val_bits;
        break;
    default:
        break;
    }
}

static void Reduc_105_unit_reduce(
    hls::stream<net_wrapper_kt_pair_105_t_t> (&kt_wrap_item)[PE_NUM],
    hls::stream<internal_end_data_batch_t> &o_0, int32_t dst_num) {
    // --- Phase 1: Memory Declaration ---

    // URAM for packed distance data (3 distances per 72-bit word)
    const int MEM_SIZE = (MAX_NUM >> LOG_PE_NUM) / DISTANCES_PER_REDUCE_WORD;
    reduce_word_t prop_mem[PE_NUM][MEM_SIZE];
#pragma HLS BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM
#pragma HLS ARRAY_PARTITION variable = prop_mem complete dim = 1
#pragma HLS dependence variable = prop_mem inter false direction = WAW
#pragma HLS dependence variable = prop_mem inter false direction = RAW

    //     // BRAM for individual validity flags (fast access)
    //     bool prop_valid[PE_NUM][MAX_NUM >> LOG_PE_NUM];
    // #pragma HLS BIND_STORAGE variable = prop_valid type = RAM_2P impl = BRAM
    // #pragma HLS ARRAY_PARTITION variable = prop_valid complete dim = 1
    // #pragma HLS dependence variable = prop_valid inter false direction = WAW
    // #pragma HLS dependence variable = prop_valid inter false direction = RAW

    // BRAM for pre-calculated address mapping (avoids division/modulo)
    // 16 bits: 14 for word_addr, 2 for pack_idx
    typedef ap_uint<16> addr_map_t;
    addr_map_t key_to_addr_map[PE_NUM][MAX_NUM >> LOG_PE_NUM];
#pragma HLS BIND_STORAGE variable = key_to_addr_map type = RAM_1P impl = BRAM
#pragma HLS ARRAY_PARTITION variable = key_to_addr_map complete dim = 1

    // Latency-hiding cache for recently accessed URAM words
    reduce_word_t cache_data_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
    reduce_word_t tmp_cache_data_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable = tmp_cache_data_buffer complete dim = 0
    int cache_addr_buffer[PE_NUM][L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0
    int tmp_cache_addr_buffer[PE_NUM][L];
#pragma HLS ARRAY_PARTITION variable = tmp_cache_addr_buffer complete dim = 0

    distance_t MAX_DISTANCE = (distance_t)(16384.0);
    const ap_fixed_pod_t MAX_DISTANCE_POD =
        *reinterpret_cast<ap_fixed_pod_t *>(&MAX_DISTANCE);
    const reduce_word_t MAX_REDUCE_WORD =
        (((reduce_word_t)MAX_DISTANCE_POD << DISTANCE_BITWIDTH) |
         ((reduce_word_t)MAX_DISTANCE_POD));

// --- Phase 2: Initialization ---
LOOP_INIT_VALID:
    for (int i = 0; i < (MAX_NUM >> LOG_PE_NUM); i++) {
#pragma HLS PIPELINE II = 1
        // Populate the address map
        addr_map_t map_val;
        map_val.range(15, 2) = i / DISTANCES_PER_REDUCE_WORD; // word_addr
        map_val.range(1, 0) = i % DISTANCES_PER_REDUCE_WORD;  // pack_idx
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            key_to_addr_map[pe][i] = map_val;
        }
    }

LOOP_INIT_MEM:
    for (int i = 0; i < MEM_SIZE; i++) {
#pragma HLS PIPELINE II = 1
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            prop_mem[pe][i] = MAX_REDUCE_WORD; // Initialize distances to max
        }
    }

LOOP_INIT_CACHE_ADDR:
    for (int i = 0; i < L + 1; i++) {
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            cache_addr_buffer[pe][i] = -1; // Invalidate cache
        }
    }

    // --- Phase 3: Aggregation Loop ---
    bool all_end_flags[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = all_end_flags complete dim = 0
    for (int i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
        all_end_flags[i] = false;
    }

LOOP_AGGREGATE:
    while (true) {
#pragma HLS PIPELINE II = 1
        for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
            net_wrapper_kt_pair_105_t_t kt_elem;
            if (!all_end_flags[pe] && kt_wrap_item[pe].read_nb(kt_elem)) {
                if (kt_elem.end_flag) {
                    all_end_flags[pe] = true;
                } else {
                    int key = kt_elem.data.key >> LOG_PE_NUM;
                    ap_fixed_pod_t incoming_dist_pod =
                        kt_elem.data.transform.prop;

                    addr_map_t map_val = key_to_addr_map[pe][key];
                    int word_addr = map_val.range(15, 2);
                    int pack_idx = map_val.range(1, 0);

                    reduce_word_t current_word = prop_mem[pe][word_addr];

                    // Check cache first
                    for (int i = L; i >= 0; --i) {
#pragma HLS UNROLL
                        if (cache_addr_buffer[pe][i] == word_addr) {
                            current_word = cache_data_buffer[pe][i];
                            break;
                        }
                    }

                    for (int i = 0; i < L; i++) {
#pragma HLS UNROLL
                        cache_addr_buffer[pe][i] = cache_addr_buffer[pe][i + 1];
                        cache_data_buffer[pe][i] = cache_data_buffer[pe][i + 1];
                    }

                    distance_t new_dist_fp;
                    distance_t incoming_dist_fp =
                        *reinterpret_cast<distance_t *>(&incoming_dist_pod);

                    // printf("[BIG] PE %d processing key %d (word_addr %d,
                    // pack_idx %d) with incoming_dist %f\n", pe, key,
                    // word_addr, pack_idx, (float)incoming_dist_fp);
                    // fflush(NULL);

                    distance_t old_dist_fp = get_val(current_word, pack_idx);
                    // printf("[BIG]  Old distance: %f\n",
                    // (float)old_dist_fp);
                    new_dist_fp = (old_dist_fp < incoming_dist_fp)
                                      ? old_dist_fp
                                      : incoming_dist_fp;

                    ap_fixed_pod_t new_dist_pod =
                        *reinterpret_cast<ap_fixed_pod_t *>(&new_dist_fp);
                    // current_word.range(end_bit, start_bit) = new_dist_pod;
                    set_val(current_word, pack_idx, new_dist_fp);

                    // Write back to URAM and update cache
                    prop_mem[pe][word_addr] = current_word;
                    cache_addr_buffer[pe][L] = word_addr;
                    cache_data_buffer[pe][L] = current_word;
                }
            }
        }
        bool end_flag = true;
        for (int i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            end_flag = (end_flag & all_end_flags[i]);
        }
        if (end_flag) {
            break;
        }
    }

    // --- Phase 4: High-Performance Drain Loop ---
    internal_end_data_batch_t data_pack;
#pragma HLS ARRAY_PARTITION variable = data_pack.data complete dim = 0
    data_pack.end_flag = 0;

    const int32_t num_keys_per_pe = (dst_num + PE_NUM - 1) >> LOG_PE_NUM;
    int real_addr = 0;

LOOP_DRAIN_ADDR:
    const int32_t mem_real_size =
        (dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;
    for (int addr = 0; addr < mem_real_size; addr++) {
        reduce_word_t words[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = words complete dim = 0
        for (int pe = 0; pe < PE_NUM; ++pe) {
#pragma HLS UNROLL
            words[pe] = prop_mem[pe][addr];
        }

    LOOP_DRAIN_PACK_IDX:
        for (int pack_idx = 0; pack_idx < DISTANCES_PER_REDUCE_WORD;
             pack_idx++) {
#pragma HLS PIPELINE II = 1
            int key = real_addr + pack_idx;
            if (key >= num_keys_per_pe) {
                break;
            }
            const int large_key_base = key << LOG_PE_NUM;

        LOOP_DRAIN_PES:
            for (int pe = 0; pe < PE_NUM; pe++) {
#pragma HLS UNROLL
                ap_fixed_pod_t dist_pod = get_raw_val(words[pe], pack_idx);
                data_pack.data[pe].node_id = large_key_base | pe;
                data_pack.data[pe].prop = dist_pod;
                // printf("[BIG] Outputting node_id %d with distance %f\n",
                // data_pack.data[pe].node_id,
                // (float)*reinterpret_cast<distance_t*>(&dist_pod));
                // fflush(NULL);
            }
            const int remaining_nodes = dst_num - large_key_base;
            data_pack.end_pos =
                (remaining_nodes < PE_NUM) ? remaining_nodes : PE_NUM;
            o_0.write(data_pack);
        }
        real_addr += DISTANCES_PER_REDUCE_WORD;
    }

    data_pack.end_flag = true;
    data_pack.end_pos = 0;
    o_0.write(data_pack);
}

static void Scatt_234(hls::stream<struct_sbu_7_t> &i_0,
                      hls::stream<struct_abu_9_t> &o_0,
                      hls::stream<struct_nbu_11_t> &o_1,
                      hls::stream<struct_abu_9_t> &o_2) {
    struct_sbu_7_t in_batch_i_0;
    struct_abu_9_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    struct_abu_9_t out_batch_o_2;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_43:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_FOR_42:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].ele_0;
            out_batch_o_1.data[i] = in_batch_i_0.data[i].ele_1;
            out_batch_o_2.data[i] = in_batch_i_0.data[i].ele_2;
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        out_batch_o_2.end_flag = end_flag;
        out_batch_o_2.end_pos = end_pos;
        o_2.write(out_batch_o_2);
        if (end_flag) {
            break;
        }
    }
}

static void Memor_231(hls::stream<struct_ibu_14_t> &o_0_node_id,
                      hls::stream<struct_nbu_11_t> &i_0_node_id) {
    struct_nbu_11_t in_batch_i_0_node_id;
    struct_ibu_14_t out_batch_o_0_node_id;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_45:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_node_id = i_0_node_id.read();
    LOOP_FOR_44:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0_node_id.data[i] = in_batch_i_0_node_id.data[i];
        }
        end_flag = in_batch_i_0_node_id.end_flag;
        end_pos = in_batch_i_0_node_id.end_pos;
        out_batch_o_0_node_id.end_flag = end_flag;
        out_batch_o_0_node_id.end_pos = end_pos;
        o_0_node_id.write(out_batch_o_0_node_id);
        if (end_flag) {
            break;
        }
    }
}

static void CopyC_247(hls::stream<struct_nbu_11_t> &i_0,
                      hls::stream<struct_nbu_11_t> &o_0,
                      hls::stream<struct_nbu_11_t> &o_1) {
    struct_nbu_11_t in_batch_i_0;
    struct_nbu_11_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_47:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_FOR_46:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i];
            out_batch_o_1.data[i] = in_batch_i_0.data[i];
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        if (end_flag) {
            break;
        }
    }
}

static void fused_op_269(hls::stream<struct_abu_9_t> &i_0,
                         hls::stream<struct_nbu_11_t> &i_1,
                         hls::stream<struct_abu_9_t> &i_2,
                         hls::stream<struct_sbu_7_t> &o_0) {
    struct_abu_9_t in_batch_i_0;
    struct_nbu_11_t in_batch_i_1;
    struct_abu_9_t in_batch_i_2;
    struct_sbu_7_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_49:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
        in_batch_i_2 = i_2.read();
    LOOP_FOR_48:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // -- Inlining FusedOp fused_op_269 --
            // Inlining Gathe_262
            out_batch_o_0.data[i].ele_0 = in_batch_i_0.data[i];
            out_batch_o_0.data[i].ele_1 = in_batch_i_1.data[i];
            out_batch_o_0.data[i].ele_2 = in_batch_i_2.data[i];
            // -- End Inlining FusedOp fused_op_269 --
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        if (end_flag) {
            break;
        }
    }
}

static void Memor_274(hls::stream<edge_batch_t> &i_0_edge_id,
                      hls::stream<struct_abu_9_t> &o_0_edge_src_distance,
                      hls::stream<struct_nbu_11_t> &o_0_edge_dst,
                      hls::stream<struct_abu_9_t> &o_0_edge_weight) {
    edge_batch_t in_batch_i_0_edge_id;
    struct_abu_9_t out_batch_o_0_edge_src_distance;
    struct_nbu_11_t out_batch_o_0_edge_dst;
    struct_abu_9_t out_batch_o_0_edge_weight;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_51:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0_edge_id = i_0_edge_id.read();
    LOOP_FOR_50:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0_edge_src_distance.data[i] =
                in_batch_i_0_edge_id.src_distances[i];
            out_batch_o_0_edge_dst.data[i] = in_batch_i_0_edge_id.dsts[i];
            out_batch_o_0_edge_weight.data[i] = in_batch_i_0_edge_id.weights[i];
        }
        end_flag = in_batch_i_0_edge_id.end_flag;
        end_pos = in_batch_i_0_edge_id.end_pos;
        out_batch_o_0_edge_src_distance.end_flag = end_flag;
        out_batch_o_0_edge_src_distance.end_pos = end_pos;
        o_0_edge_src_distance.write(out_batch_o_0_edge_src_distance);
        out_batch_o_0_edge_dst.end_flag = end_flag;
        out_batch_o_0_edge_dst.end_pos = end_pos;
        o_0_edge_dst.write(out_batch_o_0_edge_dst);
        out_batch_o_0_edge_weight.end_flag = end_flag;
        out_batch_o_0_edge_weight.end_pos = end_pos;
        o_0_edge_weight.write(out_batch_o_0_edge_weight);
        if (end_flag) {
            break;
        }
    }
}

static void Memor_299(hls::stream<node_dist_batch_t> &i_all_node_distances,
                      hls::stream<struct_abu_9_t> &o_0_node_distance,
                      int32_t dst_num) {
    // Efficiently filters a stream of all node distances against a stream of
    // requested node IDs.
    node_dist_batch_t in_dist_batch;
    struct_abu_9_t out_dist_batch;
#pragma HLS ARRAY_PARTITION variable = in_dist_batch.data complete dim = 0
#pragma HLS ARRAY_PARTITION variable = out_dist_batch.data complete dim = 0
    out_dist_batch.end_flag = false;
LOOP_WHILE_52:
    for (uint32_t in_node_base_id = 0; in_node_base_id < dst_num;
         in_node_base_id += PE_NUM) {
#pragma HLS PIPELINE II = 1
        in_dist_batch = i_all_node_distances.read();
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_dist_batch.data[i] = in_dist_batch.data[i]; // Direct copy
            // printf("[BIG] Reading node_id %d with distance %f\n",
            // in_node_base_id + i,
            // (float)*reinterpret_cast<distance_t*>(&out_dist_batch.data[i]));
            // fflush(NULL);
        }
        const int remaining_nodes = dst_num - in_node_base_id;
        out_dist_batch.end_pos =
            (remaining_nodes < PE_NUM) ? remaining_nodes : PE_NUM;

        o_0_node_distance.write(out_dist_batch);
    }
    // Send the final (empty) output batch with the end flag
    struct_abu_9_t final_batch;
    final_batch.end_flag = true;
    final_batch.end_pos = 0;
    o_0_node_distance.write(final_batch);
// Drain any remaining batches from the all_distances stream to prevent deadlock
LOOP_WHILE_53:
    while ((!in_dist_batch.end_flag)) {
        in_dist_batch = i_all_node_distances.read();
    }
}

// static void Memor_299(hls::stream<node_dist_batch_t> &i_all_node_distances,
//                       hls::stream<struct_abu_9_t> &o_0_node_distance,
//                       hls::stream<struct_nbu_11_t> &i_0_node_id) {
//     // Efficiently filters a stream of all node distances against a stream of
//     // requested node IDs.
//     struct_nbu_11_t in_node_id_batch;
//     node_dist_batch_t in_dist_batch;
//     struct_abu_9_t out_dist_batch;
// #pragma HLS ARRAY_PARTITION variable = in_node_id_batch.data complete dim = 0
// #pragma HLS ARRAY_PARTITION variable = in_dist_batch.data complete dim = 0
// #pragma HLS ARRAY_PARTITION variable = out_dist_batch.data complete dim = 0
//     out_dist_batch.end_flag = false;
//     // Initial reads to prime the pipeline
//     in_node_id_batch = i_0_node_id.read();
//     in_dist_batch = i_all_node_distances.read();
//     uint32_t in_node_base_id = 0;
//     uint32_t id_idx = 0;
//     uint32_t in_node_end_id;
//     in_node_end_id = in_dist_batch.end_pos;
// #pragma HLS BIND_STORAGE variable = in_node_end_id type = register impl = srl
// LOOP_WHILE_52:
//     while (true) {
// #pragma HLS PIPELINE II = 1
// #pragma HLS expression_balance
//         uint32_t current_batch_len = in_node_id_batch.end_pos;
//         if (((current_batch_len == 0) | (id_idx >= current_batch_len))) {
//             if ((id_idx > 0)) {
//                 out_dist_batch.end_pos = id_idx;
//                 o_0_node_distance.write(out_dist_batch);
//             }
//             if (in_node_id_batch.end_flag) {
//                 break;
//             }
//             in_node_id_batch = i_0_node_id.read();
//             id_idx = 0;
//             continue;
//         }
//         node_id_t target_node_id = in_node_id_batch.data[id_idx];
//         if ((target_node_id >= in_node_end_id)) {
//             // Target is in a future batch, load the next distance batch
//             in_node_base_id = in_node_end_id;
//             in_dist_batch = i_all_node_distances.read();
//             uint32_t batch_len = in_dist_batch.end_pos;
//             in_node_end_id = (in_node_base_id + batch_len);
// #pragma HLS BIND_OP variable = in_node_end_id op = add impl = fabric latency
// = 0
//             continue;
//         }
//         // Target found, calculate index and copy distance
//         out_dist_batch.data[id_idx] =
//             in_dist_batch.data[target_node_id - in_node_base_id];
//         id_idx = (id_idx + 1);
//     }
//     // Send the final (empty) output batch with the end flag
//     struct_abu_9_t final_batch;
//     final_batch.end_flag = true;
//     final_batch.end_pos = 0;
//     o_0_node_distance.write(final_batch);
// // Drain any remaining batches from the all_distances stream to prevent
// deadlock LOOP_WHILE_53:
//     while ((!in_dist_batch.end_flag)) {
//         in_dist_batch = i_all_node_distances.read();
//     }
// }

static void Scatt_302(hls::stream<internal_end_data_batch_t> &i_0,
                      hls::stream<struct_abu_9_t> &o_0,
                      hls::stream<struct_nbu_11_t> &o_1) {
    internal_end_data_batch_t in_batch_i_0;
    struct_abu_9_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_55:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_FOR_54:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i].prop;
            out_batch_o_1.data[i] = in_batch_i_0.data[i].node_id;
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        if (end_flag) {
            break;
        }
    }
}

static void CopyC_306(hls::stream<struct_nbu_11_t> &i_0,
                      hls::stream<struct_nbu_11_t> &o_0,
                      hls::stream<struct_nbu_11_t> &o_1) {
    struct_nbu_11_t in_batch_i_0;
    struct_nbu_11_t out_batch_o_0;
    struct_nbu_11_t out_batch_o_1;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_57:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
    LOOP_FOR_56:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_batch_o_0.data[i] = in_batch_i_0.data[i];
            out_batch_o_1.data[i] = in_batch_i_0.data[i];
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        out_batch_o_1.end_flag = end_flag;
        out_batch_o_1.end_pos = end_pos;
        o_1.write(out_batch_o_1);
        if (end_flag) {
            break;
        }
    }
}

static void fused_op_294(hls::stream<internal_end_data_batch_t> &i_0,
                         hls::stream<node_dist_batch_t> &i_1,
                         hls::stream<internal_end_data_batch_t> &o_0) {
    internal_end_data_batch_t in_batch_i_0;
    node_dist_batch_t in_batch_i_1;
    internal_end_data_batch_t out_batch_o_0;
    bool end_flag;
    uint8_t end_pos;
LOOP_WHILE_59:
    while (true) {
#pragma HLS PIPELINE
        in_batch_i_0 = i_0.read();
        in_batch_i_1 = i_1.read();
    LOOP_FOR_58:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            // -- Inlining FusedOp fused_op_294 --
            // Inlining BinOp_128
            ap_fixed_pod_t fused_temp_BinOp_128_o_0;
            distance_t lhs_128 =
                *reinterpret_cast<distance_t *>(&in_batch_i_0.data[i].prop);
            distance_t rhs_128 =
                *reinterpret_cast<distance_t *>(&in_batch_i_1.data[i]);
            distance_t temp_BinOp_128_o_0_ap_result;
            temp_BinOp_128_o_0_ap_result =
                (((lhs_128) < (rhs_128) ? lhs_128 : rhs_128));
            fused_temp_BinOp_128_o_0 = *reinterpret_cast<ap_fixed_pod_t *>(
                &temp_BinOp_128_o_0_ap_result);
            // Inlining Gathe_288
            out_batch_o_0.data[i].prop = fused_temp_BinOp_128_o_0;
            out_batch_o_0.data[i].node_id = in_batch_i_0.data[i].node_id;
            // -- End Inlining FusedOp fused_op_294 --
        }
        end_flag = in_batch_i_0.end_flag;
        end_pos = in_batch_i_0.end_pos;
        out_batch_o_0.end_flag = end_flag;
        out_batch_o_0.end_pos = end_pos;
        o_0.write(out_batch_o_0);
        if (end_flag) {
            break;
        }
    }
}

static void graphyflow_big_dataflow(
    hls::stream<edge_batch_t> &response_to_318,
    hls::stream<node_dist_batch_t> &all_node_distances_to_343,
    hls::stream<internal_end_data_batch_t> &internal_end_stream,
    int32_t dst_num) {
#pragma HLS DATAFLOW
    hls::stream<struct_kbu_50_t> reduce_105_z2d_pair;
#pragma HLS STREAM variable = reduce_105_z2d_pair depth = 4
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_d2o_pair[8];
#pragma HLS STREAM variable = reduce_105_d2o_pair depth = 4
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_o2u_pair[8];
#pragma HLS STREAM variable = reduce_105_o2u_pair depth = 4
    //     hls::stream<struct_ibu_14_t> intermediate_key;
    // #pragma HLS STREAM variable = intermediate_key depth = 4
    //     hls::stream<internal_end_data_batch_t> intermediate_transform;
    // #pragma HLS STREAM variable = intermediate_transform depth = 4
    //     hls::stream<struct_sbu_7_t> stream_o_0_273;
    // #pragma HLS STREAM variable = stream_o_0_273 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_0_236;
    // #pragma HLS STREAM variable = stream_o_0_236 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_1_237;
    // #pragma HLS STREAM variable = stream_o_1_237 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_2_238;
    // #pragma HLS STREAM variable = stream_o_2_238 depth = 4
    //     hls::stream<struct_ibu_14_t> stream_o_0_node_id_232;
    // #pragma HLS STREAM variable = stream_o_0_node_id_232 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_1_250;
    // #pragma HLS STREAM variable = stream_o_1_250 depth = 4
    hls::stream<internal_end_data_batch_t> stream_o_0_107;
#pragma HLS STREAM variable = stream_o_0_107 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_0_249;
    // #pragma HLS STREAM variable = stream_o_0_249 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_0_edge_src_distance_275;
    // #pragma HLS STREAM variable = stream_o_0_edge_src_distance_275 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_0_edge_dst_277;
    // #pragma HLS STREAM variable = stream_o_0_edge_dst_277 depth = 4
    //     hls::stream<struct_abu_9_t> stream_o_0_edge_weight_278;
    // #pragma HLS STREAM variable = stream_o_0_edge_weight_278 depth = 4
    hls::stream<struct_abu_9_t> stream_o_0_node_distance_300;
#pragma HLS STREAM variable = stream_o_0_node_distance_300 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_1_309;
    // #pragma HLS STREAM variable = stream_o_1_309 depth = 4
    hls::stream<struct_abu_9_t> stream_o_0_304;
#pragma HLS STREAM variable = stream_o_0_304 depth = 4
    hls::stream<struct_nbu_11_t> stream_o_1_305;
#pragma HLS STREAM variable = stream_o_1_305 depth = 4
    //     hls::stream<struct_nbu_11_t> stream_o_0_308;
    // #pragma HLS STREAM variable = stream_o_0_308 depth = 4
    // --- Function Calls (in topological order) ---
    // Memor_274(response_to_318, stream_o_0_edge_src_distance_275,
    //           stream_o_0_edge_dst_277, stream_o_0_edge_weight_278);
    // fused_op_269(stream_o_0_edge_src_distance_275, stream_o_0_edge_dst_277,
    //              stream_o_0_edge_weight_278, stream_o_0_273);
    // Scatt_234(stream_o_0_273, stream_o_0_236, stream_o_1_237,
    // stream_o_2_238); CopyC_247(stream_o_1_237, stream_o_0_249,
    // stream_o_1_250); Memor_231(stream_o_0_node_id_232, stream_o_1_250);
    // --- Start of Reduce Super-Block for Reduc_105 ---
    Reduc_105_pre_process(response_to_318, reduce_105_z2d_pair);
    // stream_zipper_0(intermediate_key, intermediate_transform,
    //                 reduce_105_z2d_pair);
    demux_1(reduce_105_z2d_pair, reduce_105_d2o_pair);
    omega_switch_2(reduce_105_d2o_pair, reduce_105_o2u_pair);
    Reduc_105_unit_reduce(reduce_105_o2u_pair, stream_o_0_107, dst_num);
    // --- End of Reduce Super-Block for Reduc_105 ---
    // Scatt_302(stream_o_0_107, stream_o_0_304, stream_o_1_305);
    // CopyC_306(stream_o_1_305, stream_o_0_308, stream_o_1_309);
    // Memor_299(all_node_distances_to_343, stream_o_0_node_distance_300,
    // dst_num);
    fused_op_294(stream_o_0_107, all_node_distances_to_343,
                 internal_end_stream);
}

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void graphyflow_big(const bus_word_t *edge_props,
                               const bus_word_t *node_props,
                               //    bus_word_t *output,
                               int32_t num_nodes, int32_t num_edges,
                               int32_t dst_num,
                               hls::stream<node_dist_pkt_t> &node_dist_stream,
                               hls::stream<write_burst_pkt_t> &output_stream) {
#pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem1
// #pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = edge_props
#pragma HLS INTERFACE s_axilite port = node_props
// #pragma HLS INTERFACE s_axilite port = output
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = dst_num
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW

    // Streams for the new COO-style property loading
    hls::stream<node_id_burst_t> stream_src_ids;
#pragma HLS STREAM variable = stream_src_ids depth = 16
    //     hls::stream<node_id_burst_t> stream_src_ids_2;
    // #pragma HLS STREAM variable = stream_src_ids_2 depth = 16
    hls::stream<distance_req_pack_t> stream_dist_req;
#pragma HLS STREAM variable = stream_dist_req depth = 16
    hls::stream<cacheline_req_t> stream_cache_req;
#pragma HLS STREAM variable = stream_cache_req depth = 16
    hls::stream<cacheline_resp_t> stream_cache_resp;
#pragma HLS STREAM variable = stream_cache_resp depth = 16
    hls::stream<bus_word_t> stream_cachelines[PE_NUM];
#pragma HLS STREAM variable = stream_cachelines depth = 16

    // Existing streams
    hls::stream<node_distance_burst_t> node_distance_burst_stream_1;
#pragma HLS STREAM variable = node_distance_burst_stream_1 depth = 16
    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 16
    hls::stream<edge_batch_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 16
    hls::stream<node_dist_batch_t> stream_node_dist_data;
#pragma HLS STREAM variable = stream_node_dist_data depth = 16
    hls::stream<internal_end_data_batch_t> stream_result_data;
#pragma HLS STREAM variable = stream_result_data depth = 16

    // --- Data Loading ---
    // src_id_loader(src_ids, stream_src_ids_1, stream_src_ids_2, num_edges);
    edge_descriptor_loader(edge_props, stream_src_ids, edge_stream, num_edges);

    // --- New COO-style Source Property Loading Pipeline ---
    dist_req_packer(stream_src_ids, stream_dist_req, num_edges);
    cacheline_req_sender(stream_dist_req, stream_cache_req);
    node_property_loader(node_props, stream_cache_req, stream_cache_resp,
                         num_nodes);
    node_prop_resp_receiver(stream_cache_resp, stream_cachelines);
    merge_node_props(stream_cachelines, edge_stream, stream_edge_data,
                     num_edges);

    // --- Node Property Responder for Reduce Operation ---
    node_property_responder(node_dist_stream, num_nodes, stream_node_dist_data);

    // --- Main Dataflow Processing ---
    graphyflow_big_dataflow(stream_edge_data, stream_node_dist_data,
                            stream_result_data, dst_num);

    // --- Final Writeback ---
    // final_writeback(stream_result_data, dst_num, output);
    //     hls::stream<bus_word_t> bus_word_stream;
    // #pragma HLS STREAM variable = bus_word_stream depth = 4
    pack_distances_to_bus_words(stream_result_data, output_stream);
    // write_bus_words_to_ddr(bus_word_stream, output, dst_num);
}
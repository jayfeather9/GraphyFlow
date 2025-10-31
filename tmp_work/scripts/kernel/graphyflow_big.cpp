#include "graphyflow_big.h"

static void
edge_descriptor_loader(const bus_word_t *edge_props_ddr,
                       hls::stream<node_id_burst_t> &stream_src_ids,
                       hls::stream<edge_descriptor_batch_t> &edge_stream,
                       int32_t num_edges) {
    const int bits_per_edge = NODE_ID_BITWIDTH + NODE_ID_BITWIDTH;
    const int edges_per_word = AXI_BUS_WIDTH / bits_per_edge;
    const int num_wide_reads =
        (num_edges + edges_per_word - 1) / edges_per_word;

    int edges_read = 0;
    edge_descriptor_batch_t edge_batch;
#pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
    edge_batch.end_pos = 0;

    node_id_burst_t src_id_burst;
#pragma HLS ARRAY_PARTITION variable = src_id_burst.data complete dim = 0

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
#pragma HLS ARRAY_PARTITION variable = node_id_burst.data complete dim = 0
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cache_idx[pe_idx] = node_id_burst.data[pe_idx] >> LOG_DIST_PER_WORD;
            // printf("PE %d requests node ID %d (cache idx %d)\n", pe_idx,
            // (int)node_id_burst.data[pe_idx], (int)cache_idx[pe_idx]);
            // fflush(NULL);
        }

        ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cache_idx_diffs[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cache_idx_diffs complete dim = 0
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
#pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
            req_pack.offset = num_unread;
            req_pack.end_flag = false;

            for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
                req_pack.idx[pe_idx] = cache_idx[pe_idx];
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

static void cacheline_req_sender(
    hls::stream<distance_req_pack_t> &distance_req_pack_stream,
    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream) {

    cacheline_request_pkt_t cache_req;
    cache_req.last = false;
    cache_req.data = 0;
    cache_req.dest = 0;
    cacheline_req_stream.write(cache_req);

    ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cacheline_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cacheline_idx complete dim = 0

LOOP_SEND_CACHE_REQ:
    while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = cacheline_idx inter false
        distance_req_pack_t req_pack = distance_req_pack_stream.read();
#pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cacheline_idx[pe_idx] = req_pack.idx[pe_idx];
        }

        {
        LOOP_SEND_CACHE_REQ_INNER:
            for (ap_uint<4> i = req_pack.offset; i < PE_NUM; i++) {
#pragma HLS PIPELINE II = 1 rewind
#pragma HLS unroll factor = 1
                cache_req.data = cacheline_idx[i];
                cache_req.dest = i;
                cache_req.last = req_pack.end_flag;
                cacheline_req_stream.write(cache_req);
                // printf("Sent cacheline req for idx %d to PE %d\n",
                // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
            }
        }

        if (req_pack.end_flag) {
            break;
        }
    }
    cache_req.last = true;
    cacheline_req_stream.write(cache_req);
}

static void node_prop_resp_receiver(
    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
    hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM]) {

    cacheline_response_pkt_t cache_resp = cacheline_resp_stream.read();
    bus_word_t first_line = cache_resp.data;
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        cacheline_streams[pe_idx].write(first_line);
    }

LOOP_RECEIVE_CACHE_RESP:
    while (true) {
#pragma HLS PIPELINE II = 1
        if (cacheline_resp_stream.read_nb(cache_resp)) {
            if (cache_resp.last) {
                break;
            }
            bus_word_t resp_line = cache_resp.data;
            ap_uint<8> target_pe = cache_resp.dest;
            cacheline_streams[target_pe].write(resp_line);
        }
    }
}

// ap_fixed_pod_t get_val_from_bus(const bus_word_t bus, int offset) {
// #pragma HLS INLINE
//     switch (offset) {
//     case 0:
//         return bus.range(31, 0);
//     case 1:
//         return bus.range(63, 32);
//     case 2:
//         return bus.range(95, 64);
//     case 3:
//         return bus.range(127, 96);
//     case 4:
//         return bus.range(159, 128);
//     case 5:
//         return bus.range(191, 160);
//     case 6:
//         return bus.range(223, 192);
//     case 7:
//         return bus.range(255, 224);
//     case 8:
//         return bus.range(287, 256);
//     case 9:
//         return bus.range(319, 288);
//     case 10:
//         return bus.range(351, 320);
//     case 11:
//         return bus.range(383, 352);
//     case 12:
//         return bus.range(415, 384);
//     case 13:
//         return bus.range(447, 416);
//     case 14:
//         return bus.range(479, 448);
//     case 15:
//         return bus.range(511, 480);
//     default:
//         return 0;
//     }
// }

static void
merge_node_props(hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM],
                 hls::stream<edge_descriptor_batch_t> &edge_stream,
                 //  hls::stream<node_id_burst_t> &src_id_burst_stream,
                 hls::stream<update_tuple_t> &edge_batch_stream,
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
    distance_t real_edge_weight =
        1.0; // All edge weights are 1.0 in unweighted graph
    const ap_fixed_pod_t edge_weight = (*reinterpret_cast<ap_fixed_pod_t *>(
        &real_edge_weight)); // All edge weights are 1.0 in unweighted graph
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
        update_tuple_t out_batch;
#pragma HLS ARRAY_PARTITION variable = out_batch.node_id complete dim = 0
#pragma HLS ARRAY_PARTITION variable = out_batch.prop complete dim = 0
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
                ap_fixed_pod_t prop = cacheline.range(
                    DISTANCE_BITWIDTH - 1 + (offset << LOG_DIST_BITWIDTH),
                    offset << LOG_DIST_BITWIDTH);

                out_batch.node_id[pe_idx] = edge_batch.edges[pe_idx].dst_id;
                out_batch.prop[pe_idx] = (prop + edge_weight);

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
    update_tuple_t end_batch;
    end_batch.end_flag = true;
    end_batch.end_pos = 0;
    edge_batch_stream.write(end_batch);
}

// --- 2. Utility Network Functions ---

static void
demux_1(hls::stream<update_tuple_t> &in_batch_stream,
        hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8]) {
    update_tuple_t in_batch;
#pragma HLS ARRAY_PARTITION variable = in_batch.node_id complete dim = 0
#pragma HLS ARRAY_PARTITION variable = in_batch.prop complete dim = 0
LOOP_WHILE_22:
    while (true) {
#pragma HLS PIPELINE
        in_batch = in_batch_stream.read();
        net_wrapper_kt_pair_105_t_t wrapper_data;
#pragma HLS ARRAY_PARTITION variable = wrapper_data.node_id complete dim = 0
#pragma HLS ARRAY_PARTITION variable = wrapper_data.prop complete dim = 0
    LOOP_FOR_20:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            if ((i < in_batch.end_pos)) {
                wrapper_data.node_id = in_batch.node_id[i];
                wrapper_data.prop = in_batch.prop[i];
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
                if (((data1.node_id >> i) & 1)) {
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
                if (((data2.node_id >> i) & 1)) {
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
#pragma HLS ARRAY_PARTITION variable = stream_stage_0 complete dim = 0
    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_1[8];
#pragma HLS STREAM variable = stream_stage_1 depth = 2
#pragma HLS ARRAY_PARTITION variable = stream_stage_1 complete dim = 0
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

inline ap_fixed_pod_t get_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> bits;
    switch (idx) {
    case 0:
        bits = word.range(DISTANCE_BITWIDTH - 1, 0);
        break;
    case 1:
        bits = word.range(2 * DISTANCE_BITWIDTH - 1, DISTANCE_BITWIDTH);
        break;
    case 2:
        bits = word.range(3 * DISTANCE_BITWIDTH - 1, 2 * DISTANCE_BITWIDTH);
        break;
    case 3:
        bits = word.range(4 * DISTANCE_BITWIDTH - 1, 3 * DISTANCE_BITWIDTH);
        break;
    case 4:
        bits = word.range(5 * DISTANCE_BITWIDTH - 1, 4 * DISTANCE_BITWIDTH);
        break;
    case 5:
        bits = word.range(6 * DISTANCE_BITWIDTH - 1, 5 * DISTANCE_BITWIDTH);
        break;
    case 6:
        bits = word.range(7 * DISTANCE_BITWIDTH - 1, 6 * DISTANCE_BITWIDTH);
        break;
    case 7:
        bits = word.range(8 * DISTANCE_BITWIDTH - 1, 7 * DISTANCE_BITWIDTH);
        break;
    default:
        bits = 0;
        break;
    }
    return bits;
}

inline void set_val(reduce_word_t &word, int idx, ap_fixed_pod_t val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits =
        *reinterpret_cast<ap_uint<DISTANCE_BITWIDTH> *>(&val);
    switch (idx) {
    case 0:
        word.range(DISTANCE_BITWIDTH - 1, 0) = val_bits;
        break;
    case 1:
        word.range(2 * DISTANCE_BITWIDTH - 1, DISTANCE_BITWIDTH) = val_bits;
        break;
    case 2:
        word.range(3 * DISTANCE_BITWIDTH - 1, 2 * DISTANCE_BITWIDTH) = val_bits;
        break;
    case 3:
        word.range(4 * DISTANCE_BITWIDTH - 1, 3 * DISTANCE_BITWIDTH) = val_bits;
        break;
    case 4:
        word.range(5 * DISTANCE_BITWIDTH - 1, 4 * DISTANCE_BITWIDTH) = val_bits;
        break;
    case 5:
        word.range(6 * DISTANCE_BITWIDTH - 1, 5 * DISTANCE_BITWIDTH) = val_bits;
        break;
    case 6:
        word.range(7 * DISTANCE_BITWIDTH - 1, 6 * DISTANCE_BITWIDTH) = val_bits;
        break;
    case 7:
        word.range(8 * DISTANCE_BITWIDTH - 1, 7 * DISTANCE_BITWIDTH) = val_bits;
        break;
    default:
        break;
    }
}

// Single-PE aggregation function
// Handles initialization and aggregation for one PE
static void Reduc_105_unit_reduce_single_pe(
    hls::stream<net_wrapper_kt_pair_105_t_t> &kt_wrap_item_single,
    hls::stream<reduce_word_t> &pe_mem_out, int32_t pe_id, int32_t dst_num) {

    // --- Phase 1: Memory Declaration ---
    const int MEM_SIZE = (MAX_NUM >> LOG_PE_NUM) / DISTANCES_PER_REDUCE_WORD;
    reduce_word_t prop_mem[MEM_SIZE];
#pragma HLS BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM
#pragma HLS dependence variable = prop_mem inter false

    // Latency-hiding cache for recently accessed URAM words
    reduce_word_t cache_data_buffer[L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
    int32_t cache_addr_buffer[L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0

    const int32_t num_words =
        (dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;
    const int32_t num_word_per_pe = (num_words + PE_NUM - 1) / PE_NUM;

#ifdef EMULATION
    memset(prop_mem, 0, sizeof(reduce_word_t) * MEM_SIZE);
#endif

LOOP_INIT_CACHE_ADDR:
    for (int i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
        cache_addr_buffer[i] = -1; // Invalidate cache
    }

    // --- Phase 3: Aggregation Loop ---
    bool end_flag = false;

LOOP_AGGREGATE:
    while (true) {
#pragma HLS PIPELINE II = 1
        net_wrapper_kt_pair_105_t_t kt_elem;
        kt_elem = kt_wrap_item_single.read();
        if (kt_elem.end_flag) {
            break;
        }
        int32_t key = kt_elem.node_id >> LOG_PE_NUM;
        ap_fixed_pod_t incoming_dist_pod = kt_elem.prop;

        int32_t word_addr =
            (key >> LOG_DISTANCES_PER_REDUCE_WORD); // Divide by 8 since 8
                                                    // distances per word
        int32_t pack_idx =
            (key & (DISTANCES_PER_REDUCE_WORD - 1)); // Get lower 3 bits (0-7)

        if ((key & 0x40000000) == 0) {
            reduce_word_t current_word = prop_mem[word_addr];

            // Check cache first
            for (int i = L; i >= 0; --i) {
#pragma HLS UNROLL
                if (cache_addr_buffer[i] == word_addr) {
                    current_word = cache_data_buffer[i];
                    break;
                }
            }

            // Shift cache
            for (int i = 0; i < L; i++) {
#pragma HLS UNROLL
                cache_addr_buffer[i] = cache_addr_buffer[i + 1];
                cache_data_buffer[i] = cache_data_buffer[i + 1];
            }

            ap_fixed_pod_t old_dist_pod = get_val(current_word, pack_idx);
            ap_fixed_pod_t new_dist_pod =
                (old_dist_pod < incoming_dist_pod && old_dist_pod != 0x0)
                    ? old_dist_pod
                    : incoming_dist_pod;

            set_val(current_word, pack_idx, new_dist_pod);

            // Write back to URAM and update cache
            prop_mem[word_addr] = current_word;
            cache_addr_buffer[L] = word_addr;
            cache_data_buffer[L] = current_word;
        }
    }

    // --- Phase 4: Stream out aggregated memory ---
LOOP_STREAM_OUT:
    for (int i = 0; i < num_word_per_pe; i++) {
#pragma HLS UNROLL factor = 1
        pe_mem_out.write(prop_mem[i]);
        prop_mem[i] = 0;
    }
}

// float ap_fixed_to_float2(ap_fixed_pod_t val) {
//     return (float)*reinterpret_cast<distance_t *>(&val);
// }

// Multi-PE drain function
// Collects aggregated data from all PEs and outputs final results
// Each PE holds disjoint destinations (striped by PE index)
// Need to interleave bytes to get sequential output order
static void
Reduc_105_drain_multi_pe(hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM],
                         hls::stream<write_burst_pkt_t> &kernel_out_stream,
                         int32_t dst_num) {

    write_burst_pkt_t one_write_burst;
    one_write_burst.last = 0;

    // Each iteration: read 1 reduce_word from each PE (8 PEs × 8 bytes = 64
    // bytes) This fills exactly 1 bus word (512 bits = 64 bytes)
LOOP_DRAIN_ADDR:
    for (int32_t base_addr = 0; base_addr < dst_num;
         base_addr += (PE_NUM << LOG_DISTANCES_PER_REDUCE_WORD)) {
#pragma HLS PIPELINE II = 1
        // Read one word from each PE
        reduce_word_t words[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = words complete dim = 0

        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            words[pe_idx] = pe_mem_in[pe_idx].read();
        }

        // Interleave bytes: PE0[0], PE1[0], ..., PE7[0], PE0[1], PE1[1], ...
        for (int byte_idx = 0; byte_idx < DISTANCES_PER_REDUCE_WORD;
             byte_idx++) {
#pragma HLS UNROLL
            for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
                int output_bit_pos =
                    (byte_idx * PE_NUM + pe_idx) * DISTANCE_BITWIDTH;
                one_write_burst.data.range(
                    output_bit_pos + DISTANCE_BITWIDTH - 1, output_bit_pos) =
                    words[pe_idx].range((byte_idx + 1) * DISTANCE_BITWIDTH - 1,
                                        byte_idx * DISTANCE_BITWIDTH);
            }
        }

        kernel_out_stream.write(one_write_burst);
    }
}

static void
graphyflow_big_dataflow(hls::stream<update_tuple_t> &input_to_demux,
                        hls::stream<write_burst_pkt_t> &kernel_out_stream,
                        int32_t dst_num) {
#pragma HLS DATAFLOW
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_d2o_pair[8];
#pragma HLS STREAM variable = reduce_105_d2o_pair depth = 16
#pragma HLS ARRAY_PARTITION variable = reduce_105_d2o_pair complete dim = 0
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_o2u_pair[8];
#pragma HLS STREAM variable = reduce_105_o2u_pair depth = 2
#pragma HLS ARRAY_PARTITION variable = reduce_105_o2u_pair complete dim = 0

    demux_1(input_to_demux, reduce_105_d2o_pair);
    omega_switch_2(reduce_105_d2o_pair, reduce_105_o2u_pair);

    hls::stream<reduce_word_t> pe_mem_out_streams[PE_NUM];
#pragma HLS STREAM variable = pe_mem_out_streams depth = 4
LOOP_FOR_60:
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        Reduc_105_unit_reduce_single_pe(reduce_105_o2u_pair[pe_idx],
                                        pe_mem_out_streams[pe_idx], pe_idx,
                                        dst_num);
    }
    Reduc_105_drain_multi_pe(pe_mem_out_streams, kernel_out_stream, dst_num);
}

// --- 5. Top-level AXI Kernel Wrapper ---
extern "C" void
graphyflow_big(const bus_word_t *edge_props,
               //    const bus_word_t *node_props,
               //    bus_word_t *output,
               int32_t num_nodes, int32_t num_edges, int32_t dst_num,
               hls::stream<cacheline_request_pkt_t> &cacheline_req_stream,
               hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
               hls::stream<write_burst_pkt_t> &kernel_out_stream) {
#pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem0
// #pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem1
// #pragma HLS INTERFACE m_axi port = output offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = edge_props
// #pragma HLS INTERFACE s_axilite port = node_props
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
#pragma HLS STREAM variable = stream_dist_req depth = 32
    hls::stream<bus_word_t> stream_cachelines[PE_NUM];
#pragma HLS STREAM variable = stream_cachelines depth = 32

    // Existing streams
    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 32
    hls::stream<update_tuple_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 16

    // --- Data Loading ---
    edge_descriptor_loader(edge_props, stream_src_ids, edge_stream, num_edges);

    // --- New COO-style Source Property Loading Pipeline ---
    dist_req_packer(stream_src_ids, stream_dist_req, num_edges);
    cacheline_req_sender(stream_dist_req, cacheline_req_stream);
    node_prop_resp_receiver(cacheline_resp_stream, stream_cachelines);
    merge_node_props(stream_cachelines, edge_stream, stream_edge_data,
                     num_edges);

    // --- Main Dataflow Processing ---
    graphyflow_big_dataflow(stream_edge_data, kernel_out_stream, dst_num);
}

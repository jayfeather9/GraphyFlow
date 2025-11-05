#include "graphyflow_big.h"

template <typename T1, typename T2>
void stream2axistream(
                    hls::stream<T1> &stream,
                    hls::stream<T2> &axi_stream
                    ){
    
    stream2axistream:
    while (true){

        T1 tmp_t1 = stream.read();
        
        T2 tmp_t2;
        tmp_t2.data = tmp_t1.idx;
        tmp_t2.dest = tmp_t1.dst;
        tmp_t2.last = tmp_t1.end_flag;

        axi_stream.write(tmp_t2);

        if(tmp_t1.end_flag) break;
    }
}



template <typename T1, typename T2>
void axistream2stream(
                    hls::stream<T1> &axi_stream,
                    hls::stream<T2> &stream
                    ){
    
    axistream2stream:
    while (true){

        T1 tmp_t1 = axi_stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.data;
        tmp_t2.dst = tmp_t1.dest;
        tmp_t2.end_flag = tmp_t1.last;
        
        stream.write(tmp_t2);
        if(tmp_t2.end_flag) break;
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
    ap_uint<26> last_idx_max = 0;

LOOP_DRP_SEND_REQ:
    for (int32_t node_burst_idx = 0; node_burst_idx < max_node_burst_idx;
         node_burst_idx += 1) {
#pragma HLS PIPELINE II = 1
        node_id_burst_t node_id_burst = src_id_burst_stream.read();

        ap_uint<26> cache_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cache_idx complete dim = 0
        
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cache_idx[pe_idx] = node_id_burst.data[pe_idx].range(30, 0) >> LOG_DIST_PER_WORD;
        }

        ap_uint<26> cache_idx_diffs[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cache_idx_diffs complete dim = 0
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cache_idx_diffs[pe_idx] = cache_idx[pe_idx] - last_idx_max;
        }

        // if not all diffs are zero, send a req_pack
        if (cache_idx_diffs[PE_NUM - 1]) {
            ap_uint<PE_NUM> valid_mask;
            for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
                if (cache_idx_diffs[pe_idx] == 0) {
                    valid_mask.range(pe_idx, pe_idx) = 1;
                } else {
                    valid_mask.range(pe_idx, pe_idx) = 0;
                }
            }

            ap_uint<4> num_unread = count_end_ones(valid_mask);
            // printf("Packing req for %d unread PEs\n", (int)num_unread);
            // fflush(NULL);

            distance_req_pack_t req_pack;
// #pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
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
    {
        distance_req_pack_t end_req_pack;
        end_req_pack.end_flag = true;
        end_req_pack.offset = 7;
        distance_req_pack_stream.write(end_req_pack);
    }
}

static void cacheline_req_sender(
    hls::stream<distance_req_pack_t> &distance_req_pack_stream,
    hls::stream<cacheline_req_t> &cacheline_req_stream) {

    {  
        cacheline_req_t cache_req;
        cache_req.end_flag = false;
        cache_req.idx = 0;
        cacheline_req_stream.write(cache_req);
    }

    ap_uint<26> cacheline_idx[PE_NUM];
#pragma HLS ARRAY_PARTITION variable = cacheline_idx complete dim = 0

LOOP_SEND_CACHE_REQ:
    while (true) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = cacheline_idx inter false

        distance_req_pack_t req_pack = distance_req_pack_stream.read();
// #pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            cacheline_idx[pe_idx] = req_pack.idx[pe_idx];
        }

        {
        LOOP_SEND_CACHE_REQ_INNER:
            for (ap_uint<4> i = req_pack.offset; i < PE_NUM; i++) {
#pragma HLS PIPELINE II = 1 rewind
#pragma HLS unroll factor = 1
                cacheline_req_t cache_req;
                cache_req.idx = cacheline_idx[i];
                cache_req.dst = i;
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
    // cache_req.last = true;
    // cacheline_req_stream.write(cache_req);
}

static void node_prop_resp_receiver(
    hls::stream<cacheline_resp_t> &cacheline_resp_stream,
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
        if (cacheline_resp_stream.read_nb(cache_resp)) {
            if (cache_resp.end_flag) {
                break;
            }
            bus_word_t resp_line = cache_resp.data;
            ap_uint<8> target_pe = cache_resp.dst;
            cacheline_streams[target_pe].write(resp_line);
        }
    }
}

static void
merge_node_props(hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM],
                 hls::stream<edge_descriptor_batch_t> &edge_stream,
                 //  hls::stream<node_id_burst_t> &src_id_burst_stream,
                 hls::stream<update_tuple_t> &edge_batch_stream,
                 uint32_t edge_num) {

    bus_word_t last_cacheline[PE_NUM] = {0};
#pragma HLS ARRAY_PARTITION variable = last_cacheline complete dim = 0
    uint32_t last_cache_idx[PE_NUM] = {0};
#pragma HLS ARRAY_PARTITION variable = last_cache_idx complete dim = 0

// Init first cacheline for each PE
LOOP_INIT_CACHELINE:
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        last_cacheline[pe_idx] = cacheline_streams[pe_idx].read();
        last_cache_idx[pe_idx] = 0x0;
    }

    const uint32_t scatter_size = (edge_num + PE_NUM - 1) / PE_NUM;
    distance_t real_edge_weight =
        1.0; // All edge weights are 1.0 in unweighted graph
    const ap_fixed_pod_t edge_weight = (*reinterpret_cast<ap_fixed_pod_t *>(
        &real_edge_weight)); 

LOOP_SCATTER_EDGES:
    for (int32_t edge_batch_idx = 0; edge_batch_idx < scatter_size;
         edge_batch_idx++) {
#pragma HLS PIPELINE II = 1
        edge_descriptor_batch_t edge_batch;
        edge_batch = edge_stream.read();

        update_tuple_t out_batch;

        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            uint32_t cacheline_idx =
                (edge_batch.edges[pe_idx].src_id.range(30, 0) >> LOG_DIST_PER_WORD);
            uint32_t offset =
                (edge_batch.edges[pe_idx].src_id.range(30, 0) & (DIST_PER_WORD - 1));
            bus_word_t cacheline;
            if (cacheline_idx == last_cache_idx[pe_idx]) {
                cacheline = last_cacheline[pe_idx];
            } else {
                cacheline = cacheline_streams[pe_idx].read();
            }

            ap_fixed_pod_t prop = cacheline.range(31 + (offset << 5), offset << 5);

            out_batch.data[pe_idx].node_id = edge_batch.edges[pe_idx].dst_id;
            out_batch.data[pe_idx].prop = (prop + edge_weight);

            if (pe_idx == PE_NUM - 1) {
                last_cacheline[pe_idx] = cacheline;
                last_cache_idx[pe_idx] = cacheline_idx;
            }
        }
        edge_batch_stream.write(out_batch);

        for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            last_cacheline[pe_idx] = last_cacheline[PE_NUM - 1];
            last_cache_idx[pe_idx] = last_cache_idx[PE_NUM - 1];
        }
    }
}

// --- 2. Utility Network Functions ---

static void
demux_1(hls::stream<update_tuple_t> &in_batch_stream,
        hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8],
        uint32_t edge_num) {
    const uint32_t scatter_size = (edge_num + PE_NUM - 1) / PE_NUM;

LOOP_WHILE_22:
    for (uint32_t batch_idx = 0; batch_idx < scatter_size; batch_idx++) {
#pragma HLS PIPELINE II = 1
        update_tuple_t in_batch;
        in_batch = in_batch_stream.read();

    LOOP_FOR_20:
        for (uint32_t i = 0; i < PE_NUM; i++) {
#pragma HLS UNROLL
            out_streams[i].write(in_batch.data[i]);
        }
    }
    // Propagate end_flag to all output streams
LOOP_FOR_21:
    for (uint32_t i = 0; i < 8; i++) {
#pragma HLS UNROLL
        net_wrapper_kt_pair_105_t_t end_wrapper;
        end_wrapper.end_flag = true;
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

// Single-PE aggregation function
// Handles initialization and aggregation for one PE
static void Reduc_105_unit_reduce_single_pe(
    hls::stream<net_wrapper_kt_pair_105_t_t> &kt_wrap_item_single,
    hls::stream<reduce_word_t> &pe_mem_out, int32_t dst_num) {

    // --- Phase 1: Memory Declaration ---
    const int MEM_SIZE = (MAX_NUM >> LOG_PE_NUM) / DISTANCES_PER_REDUCE_WORD;
    reduce_word_t prop_mem[MEM_SIZE];
#pragma HLS BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM
#pragma HLS dependence variable = prop_mem inter false

    // Latency-hiding cache for recently accessed URAM words
    reduce_word_t cache_data_buffer[L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
    uint32_t cache_addr_buffer[L + 1];
#pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0

    const uint32_t num_words =
        (dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;
    const uint32_t num_word_per_pe = (num_words + PE_NUM - 1) / PE_NUM;

#ifdef EMULATION
    memset(prop_mem, 0, sizeof(reduce_word_t) * MEM_SIZE);
#endif

LOOP_INIT_CACHE_ADDR:
    for (int i = 0; i < L + 1; i++) {
#pragma HLS UNROLL
        cache_addr_buffer[i] = 0x7FFFFFFF; // Invalidate cache
    }

    // --- Phase 3: Aggregation Loop ---
LOOP_AGGREGATE:
    while (true) {
#pragma HLS PIPELINE II = 1
        net_wrapper_kt_pair_105_t_t kt_elem;
        kt_elem = kt_wrap_item_single.read();
        if (kt_elem.end_flag) {
            break;
        }
        uint32_t key = kt_elem.node_id >> LOG_PE_NUM;
        ap_fixed_pod_t incoming_dist_pod = kt_elem.prop;

        uint32_t word_addr = (key >> 1);
        // uint32_t pack_idx = (key & 1);

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

        reduce_word_t tmp_cur_word = current_word;

        ap_fixed_pod_t msb = tmp_cur_word.range(63, 32);
        ap_fixed_pod_t lsb = tmp_cur_word.range(31, 0);

        ap_fixed_pod_t msb_out = 
            (msb < incoming_dist_pod && msb != 0x0) ? msb : incoming_dist_pod;
        ap_fixed_pod_t lsb_out = 
            (lsb < incoming_dist_pod && lsb != 0x0) ? lsb : incoming_dist_pod;

        reduce_word_t accumulated_msb;
        reduce_word_t accumulated_lsb;

        accumulated_msb.range(63, 32) = msb_out;
        accumulated_msb.range(31, 0) = tmp_cur_word.range(31, 0);

        accumulated_lsb.range(63, 32) = tmp_cur_word.range(63, 32);
        accumulated_lsb.range(31, 0) = lsb_out;

        if (key & 0x01) {
            prop_mem[word_addr] = accumulated_msb;
            cache_data_buffer[L] = accumulated_msb;
        } else {
            prop_mem[word_addr] = accumulated_lsb;
            cache_data_buffer[L] = accumulated_lsb;
        }
        cache_addr_buffer[L] = word_addr;
    }

    // --- Phase 4: Stream out aggregated memory ---
LOOP_STREAM_OUT:
    for (int i = 0; i < num_word_per_pe; i++) {
#pragma HLS UNROLL factor = 1
        reduce_word_t tmp_word = prop_mem[i];
        prop_mem[i] = 0;
        pe_mem_out.write(tmp_word);
    }
}

// Multi-PE drain function
// Collects aggregated data from all PEs and outputs final results
static void
Reduc_105_drain_multi_pe(hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM],
                         hls::stream<write_burst_pkt_t> &kernel_out_stream,
                         int32_t dst_num) {

    // --- Phase 2: High-Performance Drain Loop ---

LOOP_DRAIN_ADDR:
    for (int32_t base_addr = 0; base_addr < dst_num;
         base_addr += (PE_NUM << 1)) {
#pragma HLS PIPELINE II = 1
        write_burst_pkt_t one_write_burst;
        reduce_word_t tmp_data[PE_NUM];
    LOOP_FOR_57:
        for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
            tmp_data[pe_idx] = pe_mem_in[pe_idx].read();

            one_write_burst.data.range(31 + (pe_idx << 5), (pe_idx << 5)) =
                tmp_data[pe_idx].range(31, 0);
            one_write_burst.data.range(31 + (pe_idx << 5) + 256,
                                       (pe_idx << 5) + 256) =
                tmp_data[pe_idx].range(63, 32);
        }
        kernel_out_stream.write(one_write_burst);
    }
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
#pragma HLS INTERFACE s_axilite port = edge_props
#pragma HLS INTERFACE s_axilite port = num_nodes
#pragma HLS INTERFACE s_axilite port = num_edges
#pragma HLS INTERFACE s_axilite port = dst_num
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS DATAFLOW

    hls::stream<node_id_burst_t> stream_src_ids;
#pragma HLS STREAM variable = stream_src_ids depth = 16
    hls::stream<distance_req_pack_t> stream_dist_req;
#pragma HLS STREAM variable = stream_dist_req depth = 32
    hls::stream<bus_word_t> stream_cachelines[PE_NUM];
#pragma HLS STREAM variable = stream_cachelines depth = 32

    hls::stream<edge_descriptor_batch_t> edge_stream;
#pragma HLS STREAM variable = edge_stream depth = 32
    hls::stream<update_tuple_t> stream_edge_data;
#pragma HLS STREAM variable = stream_edge_data depth = 16

    hls::stream<cacheline_req_t> cacheline_req;
#pragma HLS STREAM variable = cacheline_req depth = 32
    hls::stream<cacheline_resp_t> cacheline_resp;
#pragma HLS STREAM variable = cacheline_resp depth = 32

    // --- Data Loading ---
    const int edges_per_word = AXI_BUS_WIDTH / (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH);
    const int num_wide_reads = (num_edges + edges_per_word - 1) / edges_per_word;

    int edges_read = 0;

LOOP_EDL_READ:
    for (int i = 0; i < num_wide_reads; i++) {
#pragma HLS PIPELINE II = 1
        bus_word_t wide_word = edge_props[i];
        edge_descriptor_batch_t edge_batch;
        node_id_burst_t src_id_burst;
    LOOP_EDL_UNPACK:
        for (int j = 0; j < edges_per_word; j++) {
#pragma HLS UNROLL
            if (edges_read + j < num_edges) {
                ap_uint<64> packed_edge = wide_word.range(63 + (j << 6), (j << 6));
                edge_t edge;
                node_id_t src_id;
                edge.dst_id = packed_edge.range(19, 0);
                edge.src_id = packed_edge.range(63, 32);
                src_id = edge.src_id;

                edge_batch.edges[j] = edge;
                src_id_burst.data[j] = src_id;
            }
        }
        stream_src_ids.write(src_id_burst);
        edges_read += edges_per_word;
        edge_batch.end_pos = (edges_read <= num_edges)
                                 ? edges_per_word
                                 : (num_edges & (edges_per_word - 1));
        edge_stream.write(edge_batch);
    }

    // --- New COO-style Source Property Loading Pipeline ---
    dist_req_packer(stream_src_ids, stream_dist_req, num_edges);
    cacheline_req_sender(stream_dist_req, cacheline_req);
    stream2axistream(cacheline_req, cacheline_req_stream);
    axistream2stream(cacheline_resp_stream, cacheline_resp);
    node_prop_resp_receiver(cacheline_resp, stream_cachelines);
    merge_node_props(stream_cachelines, edge_stream, stream_edge_data,
                     num_edges);

    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_d2o_pair[8];
#pragma HLS STREAM variable = reduce_105_d2o_pair depth = 16
#pragma HLS ARRAY_PARTITION variable = reduce_105_d2o_pair complete dim = 0
    hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_o2u_pair[8];
#pragma HLS STREAM variable = reduce_105_o2u_pair depth = 2
#pragma HLS ARRAY_PARTITION variable = reduce_105_o2u_pair complete dim = 0

    demux_1(stream_edge_data, reduce_105_d2o_pair, num_edges);
    

    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_0[8];
#pragma HLS STREAM variable = stream_stage_0 depth = 2
#pragma HLS ARRAY_PARTITION variable = stream_stage_0 complete dim = 0
    hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_1[8];
#pragma HLS STREAM variable = stream_stage_1 depth = 2
#pragma HLS ARRAY_PARTITION variable = stream_stage_1 complete dim = 0
    switch2x2_2(2, reduce_105_d2o_pair[0], reduce_105_d2o_pair[1], stream_stage_0[0],
                stream_stage_0[1]);
    switch2x2_2(2, reduce_105_d2o_pair[2], reduce_105_d2o_pair[3], stream_stage_0[2],
                stream_stage_0[3]);
    switch2x2_2(2, reduce_105_d2o_pair[4], reduce_105_d2o_pair[5], stream_stage_0[4],
                stream_stage_0[5]);
    switch2x2_2(2, reduce_105_d2o_pair[6], reduce_105_d2o_pair[7], stream_stage_0[6],
                stream_stage_0[7]);
    switch2x2_2(1, stream_stage_0[0], stream_stage_0[4], stream_stage_1[0],
                stream_stage_1[1]);
    switch2x2_2(1, stream_stage_0[1], stream_stage_0[5], stream_stage_1[2],
                stream_stage_1[3]);
    switch2x2_2(1, stream_stage_0[2], stream_stage_0[6], stream_stage_1[4],
                stream_stage_1[5]);
    switch2x2_2(1, stream_stage_0[3], stream_stage_0[7], stream_stage_1[6],
                stream_stage_1[7]);
    switch2x2_2(0, stream_stage_1[0], stream_stage_1[4], reduce_105_o2u_pair[0],
                reduce_105_o2u_pair[1]);
    switch2x2_2(0, stream_stage_1[1], stream_stage_1[5], reduce_105_o2u_pair[2],
                reduce_105_o2u_pair[3]);
    switch2x2_2(0, stream_stage_1[2], stream_stage_1[6], reduce_105_o2u_pair[4],
                reduce_105_o2u_pair[5]);
    switch2x2_2(0, stream_stage_1[3], stream_stage_1[7], reduce_105_o2u_pair[6],
                reduce_105_o2u_pair[7]);
    // Reduc_105_unit_reduce(reduce_105_o2u_pair, stream_o_0_107, dst_num);
    hls::stream<reduce_word_t> pe_mem_out_streams[PE_NUM];
#pragma HLS STREAM variable = pe_mem_out_streams depth = 4
LOOP_FOR_60:
    for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
#pragma HLS UNROLL
        Reduc_105_unit_reduce_single_pe(reduce_105_o2u_pair[pe_idx],
                                        pe_mem_out_streams[pe_idx], dst_num);
    }
    Reduc_105_drain_multi_pe(pe_mem_out_streams, kernel_out_stream, dst_num);

    // --- Final Writeback ---
    // final_writeback(stream_result_data, dst_num, output);
    //     hls::stream<bus_word_t> bus_word_stream;
    // #pragma HLS STREAM variable = bus_word_stream depth = 4
    // pack_distances_to_bus_words(stream_result_data, kernel_out_stream);
    // write_bus_words_to_ddr(bus_word_stream, output, dst_num);
}

#include "shared_kernel_params.h"

{{MERGE_FUNCTION}}

{{GRAPHYFLOW_APPLY_FUNC}}

extern "C" void
apply_kernel({{APPLY_PARAMS}}) {
{{APPLY_INTERFACE_PRAGMAS}}
#pragma HLS DATAFLOW

    hls::stream<in_write_burst_w_dst_pkt_t> write_burst_stream;
#pragma HLS STREAM variable = write_burst_stream depth = 16

    {{MERGE_CALL}}
    apply_func(node_props, write_burst_stream, kernel_out_stream);
}
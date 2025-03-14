#include <hls_stream.h>
#include <string.h>
#include "graph_fpga.h"

#include "fpga_global_mem.h"
#include "fpga_slice.h"
#include "fpga_gather.h"
#include "fpga_filter.h"
#include "fpga_process_edge.h"
#include "fpga_cache.h"
#include "fpga_edge_prop.h"




extern "C" {
#pragma THUNDERGP MSLR_FUNCTION
    void  readEdgesCU#%d#(
        uint16          *edgesHeadArray,
        uint16          *vertexPushinProp,
        uint16          *edgesTailArray,
        uint16          *tmpVertexProp,
#if HAVE_EDGE_PROP
        uint16          *edgeProp,
#endif
        int             edge_end,
        int             sink_offset,
        int             sink_end
    )
    {
#include "fpga_gs_top.h"
    }

} // extern C

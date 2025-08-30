import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.backend_defines import HLSBasicType, HLSType, HLSVar, HLSFunction
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.backend_manager import BackendManager
from graphyflow.lambda_func import lambda_min


def strip_whitespace(text):
    """A helper function to remove all whitespace and newlines for comparison."""
    return "".join(text.split())


def render_hls_function(func: HLSFunction) -> str:
    """A helper to render a complete C++ function from an HLSFunction object."""
    params_str_list = []
    for p in func.params:
        if p.type.type == HLSBasicType.POINTER:
            params_str_list.append(p.type.get_upper_decl(p.name))
        else:
            params_str_list.append(p.type.get_upper_param(p.name, True))

    params_str = ", ".join(params_str_list)
    body_str = "".join([line.gen_code(1) for line in func.codes])

    return f"static void {func.name}({params_str}) {{\n{body_str}}}\n"


def test_stream_to_mem_generation():
    print("--- Testing Task 1.5: stream_to_mem Generation ---")

    # 1. Define a graph with a clear output type
    g = GlobalGraph(
        properties={
            "node": {"distance": dfir.FloatType()},
            "edge": {"weight": dfir.FloatType()},
        }
    )
    edges = g.add_graph_input("edge")
    pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
    min_dist = pdu.reduce_by(
        reduce_key=lambda src_dist, dst, edge_w: dst.id,
        reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
        reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
    )

    # 2. Run frontend and passes
    dfirs = g.to_dfir()
    comp_col = delete_placeholder_components_pass(dfirs[0])

    # 3. Instantiate BackendManager and run prerequisite methods
    bkd_mng = BackendManager()
    bkd_mng.comp_col_store = comp_col
    bkd_mng.global_graph_store = g
    bkd_mng.axi_output_ports = comp_col.outputs
    bkd_mng._analyze_and_map_types(comp_col)

    # 4. Call the target function
    s2m_func = bkd_mng._generate_stream_to_mem_func()

    # 5. Render generated code to string
    generated_code = render_hls_function(s2m_func)

    # 6. Define the expected "golden" code
    output_port = bkd_mng.axi_output_ports[0]
    internal_batch_type = bkd_mng.batch_type_map[bkd_mng.type_map[output_port.data_type]]

    expected_code = f"""
    static void stream_to_mem_func(hls::stream<{internal_batch_type.name}> &in_{output_port.unique_name}_stream, KernelOutputBatch* out_{output_port.unique_name}) {{
        int32_t i = 0;
        while (true) {{
            #pragma HLS PIPELINE
            {internal_batch_type.name} internal_batch;
            internal_batch = in_{output_port.unique_name}_stream.read();
            KernelOutputBatch output_batch;
            for (uint32_t k = 0; k < PE_NUM; k++) {{
                #pragma HLS UNROLL
                output_batch.data[k].distance = (float)internal_batch.data[k].ele_0;
                output_batch.data[k].id = internal_batch.data[k].ele_1.id;
            }}
            output_batch.end_flag = internal_batch.end_flag;
            output_batch.end_pos = internal_batch.end_pos;
            out_{output_port.unique_name}[i] = output_batch;
            if (out_{output_port.unique_name}[i].end_flag) {{
                break;
            }}
            i = (i + 1);
        }}
    }}
    """

    # 7. Compare and report results
    if strip_whitespace(generated_code) == strip_whitespace(expected_code):
        print("stream_to_mem Generation Test: SUCCESS")
        return True
    else:
        print("stream_to_mem Generation Test: FAILED")
        print("\n--- EXPECTED ---")
        print(expected_code.strip())
        print("\n--- GENERATED ---")
        print(generated_code.strip())
        return False


if __name__ == "__main__":
    test_dir = project_root / "implement_tests" / "task_1_5"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 1.5. Test script location: {test_dir}")

    success = test_stream_to_mem_generation()

    if success:
        print("\nAll tests for Task 1.5 PASSED.")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 1.5 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 1.5 FAILED.")
        sys.exit(1)

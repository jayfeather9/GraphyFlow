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


def strip_whitespace(text):
    """A helper function to remove all whitespace and newlines for comparison."""
    return "".join(text.split())


def render_hls_function(func: HLSFunction) -> str:
    """A helper to render a complete C++ function from an HLSFunction object."""
    params_str_list = []
    for p in func.params:
        if p.type.type == HLSBasicType.POINTER:
            params_str_list.append(p.type.get_upper_decl(p.name))
        elif "uint16_t" in p.type.name:
            params_str_list.append(f"uint16_t {p.name}")
        else:
            params_str_list.append(p.type.get_upper_param(p.name, True))

    params_str = ", ".join(params_str_list)
    body_str = "".join([line.gen_code(1) for line in func.codes])

    return f"static void {func.name}({params_str}) {{\n{body_str}}}\n"


def test_mem_to_stream_generation():
    print("--- Testing Task 1.4: mem_to_stream Generation ---")

    # 1. Define a simple graph to provide context for types and I/O
    g = GlobalGraph(
        properties={
            "node": {"distance": dfir.FloatType()},
            "edge": {"weight": dfir.FloatType()},
        }
    )
    edges = g.add_graph_input("edge")
    _ = edges.map_(map_func=lambda edge: edge.weight)

    # 2. Run frontend and passes
    dfirs = g.to_dfir()
    comp_col = delete_placeholder_components_pass(dfirs[0])

    # 3. Instantiate BackendManager and run prerequisite methods
    bkd_mng = BackendManager()
    bkd_mng.comp_col_store = comp_col
    bkd_mng.global_graph_store = g

    # --- *** 关键修正：使用正确的逻辑来查找顶层输入端口 *** ---
    top_level_inputs = []
    for comp in comp_col.components:
        if isinstance(comp, dfir.IOComponent) and comp.io_type == dfir.IOComponent.IOType.INPUT:
            if comp.get_port("o_0").connected:
                top_level_inputs.append(comp.get_port("o_0").connection)

    bkd_mng.axi_input_ports = top_level_inputs
    # -----------------------------------------------------------

    bkd_mng.axi_output_ports = comp_col.outputs
    bkd_mng._analyze_and_map_types(comp_col)

    # 4. Call the target function to generate the mem_to_stream logic
    m2s_func = bkd_mng._generate_mem_to_stream_func()

    # 5. Render the generated code to a string
    generated_code = render_hls_function(m2s_func)

    # 6. Define the expected "golden" code
    input_port = bkd_mng.axi_input_ports[0]
    batch_type = bkd_mng.batch_type_map[bkd_mng.type_map[input_port.data_type]]

    expected_code = f"""
    static void mem_to_stream_func(const {batch_type.name}* in_{input_port.unique_name}, hls::stream<{batch_type.name}> &out_{input_port.unique_name}_stream, uint16_t num_batches) {{
        for (uint32_t i = 0; i < num_batches; i++) {{
            #pragma HLS PIPELINE
            out_{input_port.unique_name}_stream.write(in_{input_port.unique_name}[i]);
        }}
    }}
    """

    # 7. Compare and report results
    if strip_whitespace(generated_code) == strip_whitespace(expected_code):
        print("mem_to_stream Generation Test: SUCCESS")
        return True
    else:
        print("mem_to_stream Generation Test: FAILED")
        print("\n--- EXPECTED ---")
        print(expected_code.strip())
        print("\n--- GENERATED ---")
        print(generated_code.strip())
        return False


if __name__ == "__main__":
    test_dir = project_root / "implement_tests" / "task_1_4"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 1.4. Test script location: {test_dir}")

    success = test_mem_to_stream_generation()

    if success:
        print("\nAll tests for Task 1.4 PASSED.")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 1.4 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 1.4 FAILED.")
        sys.exit(1)

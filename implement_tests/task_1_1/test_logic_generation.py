# tests/task_1_1/test_logic_generation.py

import os
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径中，以便导入 GraphyFlow 模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import graphyflow.dataflow_ir as dfir
import graphyflow.dataflow_ir_datatype as dftype
from graphyflow.dataflow_ir import UnaryOp, BinOp
from graphyflow.backend_defines import HLSBasicType, HLSType, HLSFunction, HLSVar, CodePragma, CodeIf
from graphyflow.backend_manager import BackendManager
from graphyflow.backend_utils import generate_demux


def strip_whitespace(text):
    """一个辅助函数，用于移除代码中的所有空格和换行符，以便进行比较。"""
    lines = text.strip().split("\n")
    clean_lines = [line for line in lines if not line.strip().startswith(("#", "//"))]
    return "".join("".join(line.split()) for line in clean_lines)


def test_demux_generation():
    print("--- Testing Demux Generation ---")

    # 1. 定义 demux 需要的类型
    PE_NUM = 8
    base_type = HLSType(HLSBasicType.INT)
    batch_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[base_type], array_dims=[PE_NUM])
    batch_type = HLSType(
        HLSBasicType.STRUCT,
        [batch_array_type, HLSType(HLSBasicType.BOOL), HLSType(HLSBasicType.UINT8)],
        struct_prop_names=["data", "end_flag", "end_pos"],
    )
    wrapper_type = HLSType(
        HLSBasicType.STRUCT,
        [base_type, HLSType(HLSBasicType.BOOL)],
        struct_prop_names=["data", "end_flag"],
    )

    # 2. 调用 demux 生成器
    demux_func = generate_demux(PE_NUM, batch_type, wrapper_type)
    generated_code = "".join([code.gen_code() for code in demux_func.codes])

    # *** 修正2：动态构建期望字符串，使其不受生成名称变化的影响 ***
    # 3. 定义期望生成的 C++ "黄金"代码
    expected_code = f"""
    {batch_type.name} in_batch;
    while (true) {{
        in_batch = in_batch_stream.read();
        {wrapper_type.name} wrapper_data;
        for (uint32_t i = 0; i < PE_NUM; i++) {{
            #pragma HLS UNROLL
            if ((i < in_batch.end_pos)) {{
                wrapper_data.data = in_batch.data[i];
                wrapper_data.end_flag = false;
                out_streams[i].write(wrapper_data);
            }}
        }}
        if (in_batch.end_flag) {{
            break;
        }}
    }}
    // Propagate end_flag to all output streams
    {wrapper_type.name} end_wrapper;
    end_wrapper.end_flag = true;
    for (uint32_t i = 0; i < 8; i++) {{
        #pragma HLS UNROLL
        out_streams[i].write(end_wrapper);
    }}
    """

    # 4. 比较并报告结果 (移除 pragma 和注释后比较)
    if strip_whitespace(generated_code) == strip_whitespace(expected_code):
        print("Demux Generation Test: SUCCESS")
        return True
    else:
        print("Demux Generation Test: FAILED")
        print("\n--- EXPECTED (logic only) ---")
        print(strip_whitespace(expected_code))
        print("\n--- GENERATED (logic only) ---")
        print(strip_whitespace(generated_code))
        return False


def test_collect_generation():
    print("\n--- Testing CollectComponent Generation ---")

    # 1. 设置测试环境
    bkd_mng = BackendManager()

    int_type = dftype.IntType()
    opt_int_type = dftype.OptionalType(int_type)
    array_opt_int_type = dftype.ArrayType(opt_int_type)

    collect_comp = dfir.CollectComponent(input_type=array_opt_int_type)

    comp_col = dfir.ComponentCollection([collect_comp], collect_comp.in_ports, collect_comp.out_ports)

    hls_func = HLSFunction("Colle_Test", collect_comp)
    bkd_mng.hls_functions[collect_comp.readable_id] = hls_func

    bkd_mng._analyze_and_map_types(comp_col)

    in_hls_type = bkd_mng.type_map[opt_int_type]  # Corrected: base type for batching
    out_hls_type = bkd_mng.type_map[collect_comp.output_type.type_]  # Corrected: base type for batching
    in_batch_type = bkd_mng.batch_type_map[in_hls_type]
    out_batch_type = bkd_mng.batch_type_map[out_hls_type]

    hls_func.params = [
        HLSVar("i_0", HLSType(HLSBasicType.STREAM, [in_batch_type])),
        HLSVar("o_0", HLSType(HLSBasicType.STREAM, [out_batch_type])),
    ]

    # 2. 调用 _translate_collect_op
    generated_codes = bkd_mng._translate_collect_op(hls_func)
    generated_code = "".join([code.gen_code() for code in generated_codes])

    # 3. 定义期望生成的 C++ "黄金"代码
    expected_code = f"""
    {in_batch_type.name} in_batch_i_0;
    {out_batch_type.name} out_batch_o_0;
    while (true) {{
        in_batch_i_0 = i_0.read();
        uint8_t out_idx = 0;
        for (uint32_t i = 0; i < PE_NUM; i++) {{
            if ((in_batch_i_0.data[i].valid & (i < in_batch_i_0.end_pos))) {{
                out_batch_o_0.data[out_idx] = in_batch_i_0.data[i].data;
                out_idx = (out_idx + 1);
            }}
        }}
        out_batch_o_0.end_pos = out_idx;
        out_batch_o_0.end_flag = in_batch_i_0.end_flag;
        o_0.write(out_batch_o_0);
        if (in_batch_i_0.end_flag) {{
            break;
        }}
    }}
    """

    # 4. 比较并报告结果
    if strip_whitespace(generated_code) == strip_whitespace(expected_code):
        print("Collect Generation Test: SUCCESS")
        return True
    else:
        print("Collect Generation Test: FAILED")
        print("\n--- EXPECTED (logic only) ---")
        print(strip_whitespace(expected_code))
        print("\n--- GENERATED (logic only) ---")
        print(strip_whitespace(generated_code))
        return False


if __name__ == "__main__":
    test_dir = project_root / "implement_tests" / "task_1_1"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 1.1. Test script location: {test_dir}")

    demux_ok = test_demux_generation()
    collect_ok = test_collect_generation()

    if demux_ok and collect_ok:
        print("\nAll tests for Task 1.1 PASSED.")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 1.1 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 1.1 FAILED.")
        sys.exit(1)

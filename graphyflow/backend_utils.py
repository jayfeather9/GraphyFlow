# hls_network_generators.py
from __future__ import annotations
import math
from typing import List

# 从您现有的后端和IR文件中导入必要的类
import graphyflow.backend_manager as hls
import graphyflow.dataflow_ir as dfir

# 全局计数器，确保每次生成的函数名都唯一
_generator_id_counter = 0


def _get_unique_id() -> int:
    """获取一个唯一的ID用于函数命名。"""
    global _generator_id_counter
    id = _generator_id_counter
    _generator_id_counter += 1
    return id


def create_non_blocking_read(
    stream_var: hls.HLSVar, body_if_not_empty: List[hls.HLSCodeLine]
) -> hls.CodeIf:
    """
    一个辅助函数，用于快速生成非阻塞读数据流的代码块。
    它利用了您在 backend.py 中对 HLSExpr 和 CodeIf 的更新。

    Args:
        stream_var: 要检查的流变量 (HLSVar)。
        body_if_not_empty: 如果流不为空，要执行的代码行列表。

    Returns:
        一个配置好的 CodeIf 实例。
    """
    # 构造条件表达式: !stream.empty()
    empty_expr = hls.HLSExpr(
        hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, stream_var)]
    )
    not_empty_expr = hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [empty_expr])

    # 返回一个只有 'if' 分支的 CodeIf 块
    return hls.CodeIf(expr=not_empty_expr, if_codes=body_if_not_empty)


def generate_omega_network(n: int, data_type_name: str) -> List[hls.HLSFunction]:
    """
    生成一个 N x N Omega 网络的完整HLS C++代码。

    Args:
        n (int): Omega网络的端口数，必须是2的幂。
        data_type_name (str): 网络中流动的数据类型名称，
                              需保证该类型有 .dst 和 .end_flag 成员。

    Returns:
        str: 一个包含所有必要函数和定义的完整C++代码字符串。
    """
    # --- 1. 参数校验和初始化 ---
    if not (n > 0 and (n & (n - 1) == 0)):
        raise ValueError("网络大小 'n' 必须是2的正整数次幂。")

    log_n = int(math.log2(n))
    switches_per_stage = n // 2
    gen_id = _get_unique_id()

    # --- 2. HLS类型定义 ---
    # 定义一个代表流动数据的HLSType，即使是占位符，它对于构造流类型也是必需的
    # 我们假设它是一个结构体，但内部细节未知，这对生成代码来说足够了。
    data_tuple_type = hls.HLSType(
        hls.HLSBasicType.STRUCT,
        sub_types=[hls.HLSType(hls.HLSBasicType.UINT), hls.HLSType(hls.HLSBasicType.BOOL)],
        struct_name=data_type_name,
        struct_prop_names=["dst", "end_flag"],
    )

    # 定义流类型 hls::stream<your_data_type>
    stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[data_tuple_type])
    bool_type = hls.HLSType(hls.HLSBasicType.BOOL)
    int_type = hls.HLSType(hls.HLSBasicType.INT)

    # --- 3. 生成 'sender' 函数 ---
    sender_func_name = f"sender_{gen_id}"

    # 定义sender的参数
    sender_params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("update_set_stm_in1", stream_type),
        hls.HLSVar("update_set_stm_in2", stream_type),
        hls.HLSVar("update_set_stm_out1", stream_type),
        hls.HLSVar("update_set_stm_out2", stream_type),
        hls.HLSVar("update_set_stm_out3", stream_type),
        hls.HLSVar("update_set_stm_out4", stream_type),
    ]

    # 构建sender的函数体
    in1_end_flag_var = hls.HLSVar("in1_end_flag", bool_type)
    in2_end_flag_var = hls.HLSVar("in2_end_flag", bool_type)
    data1_var = hls.HLSVar("data1", data_tuple_type)
    data2_var = hls.HLSVar("data2", data_tuple_type)
    i_var_expr = hls.HLSExpr(hls.HLSExprT.VAR, hls.HLSVar("i", int_type))

    # 路由逻辑表达式: (data.dst >> i) & 0x1
    def create_routing_expr(data_var: hls.HLSVar) -> hls.HLSExpr:
        dst_expr = hls.HLSExpr(
            hls.HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "dst"),
            [hls.HLSExpr(hls.HLSExprT.VAR, data_var)],
        )
        shifted_expr = hls.HLSExpr(hls.HLSExprT.BINOP, dfir.BinOp.SR, [dst_expr, i_var_expr])
        return hls.HLSExpr(
            hls.HLSExprT.BINOP, dfir.BinOp.AND, [shifted_expr, hls.HLSExpr(hls.HLSExprT.CONST, 1)]
        )

    # in1的处理逻辑
    in1_routing_expr = create_routing_expr(data1_var)
    in1_end_flag_check_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, data1_var)],
    )
    in1_if_not_end_block = hls.CodeIf(
        in1_routing_expr,
        if_codes=[hls.CodeWriteStream(sender_params[4], data1_var)],  # out2
        else_codes=[hls.CodeWriteStream(sender_params[3], data1_var)],  # out1
    )
    in1_read_block = hls.CodeIf(
        hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [in1_end_flag_check_expr]),
        if_codes=[in1_if_not_end_block],
        else_codes=[hls.CodeAssign(in1_end_flag_var, in1_end_flag_check_expr)],
    )
    in1_non_blocking_read = create_non_blocking_read(
        sender_params[1],
        [
            hls.CodeVarDecl(data1_var.name, data1_var.type),
            hls.CodeAssign(
                data1_var,
                hls.HLSExpr(
                    hls.HLSExprT.STREAM_READ,
                    None,
                    [hls.HLSExpr(hls.HLSExprT.VAR, sender_params[1])],
                ),
            ),
            in1_read_block,
        ],
    )

    # in2的处理逻辑
    in2_routing_expr = create_routing_expr(data2_var)
    in2_end_flag_check_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, data2_var)],
    )
    in2_if_not_end_block = hls.CodeIf(
        in2_routing_expr,
        if_codes=[hls.CodeWriteStream(sender_params[6], data2_var)],  # out4
        else_codes=[hls.CodeWriteStream(sender_params[5], data2_var)],  # out3
    )
    in2_read_block = hls.CodeIf(
        hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [in2_end_flag_check_expr]),
        if_codes=[in2_if_not_end_block],
        else_codes=[hls.CodeAssign(in2_end_flag_var, in2_end_flag_check_expr)],
    )
    in2_non_blocking_read = create_non_blocking_read(
        sender_params[2],
        [
            hls.CodeVarDecl(data2_var.name, data2_var.type),
            hls.CodeAssign(
                data2_var,
                hls.HLSExpr(
                    hls.HLSExprT.STREAM_READ,
                    None,
                    [hls.HLSExpr(hls.HLSExprT.VAR, sender_params[2])],
                ),
            ),
            in2_read_block,
        ],
    )

    # 最终退出逻辑
    end_data_var = hls.HLSVar("data", data_tuple_type)
    exit_cond = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, in1_end_flag_var),
            hls.HLSExpr(hls.HLSExprT.VAR, in2_end_flag_var),
        ],
    )
    exit_block = hls.CodeIf(
        exit_cond,
        [
            hls.CodeVarDecl(end_data_var.name, end_data_var.type),
            hls.CodeAssign(
                hls.HLSVar(f"{end_data_var.name}.end_flag", bool_type),
                hls.HLSExpr(hls.HLSExprT.CONST, True),
            ),
            hls.CodeWriteStream(sender_params[3], end_data_var),
            hls.CodeWriteStream(sender_params[4], end_data_var),
            hls.CodeWriteStream(sender_params[5], end_data_var),
            hls.CodeWriteStream(sender_params[6], end_data_var),
            hls.CodeAssign(in1_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeAssign(in2_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeBreak(),
        ],
    )

    sender_body = [
        hls.CodePragma(f"function_instantiate variable=i"),
        hls.CodeVarDecl(in1_end_flag_var.name, in1_end_flag_var.type),
        hls.CodeVarDecl(in2_end_flag_var.name, in2_end_flag_var.type),
        hls.CodeWhile(
            iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True),
            codes=[
                hls.CodePragma("PIPELINE II=1"),
                in1_non_blocking_read,
                in2_non_blocking_read,
                exit_block,
            ],
        ),
    ]

    sender_function = hls.HLSFunction(name=sender_func_name, comp=None)
    sender_function.params = sender_params
    sender_function.codes = sender_body

    # --- 4. 生成 'receiver' 函数 ---
    receiver_func_name = f"receiver_{gen_id}"

    # 定义receiver的参数
    receiver_params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("update_set_stm_out1", stream_type),
        hls.HLSVar("update_set_stm_out2", stream_type),
        hls.HLSVar("update_set_stm_in1", stream_type),
        hls.HLSVar("update_set_stm_in2", stream_type),
        hls.HLSVar("update_set_stm_in3", stream_type),
        hls.HLSVar("update_set_stm_in4", stream_type),
    ]

    # 构建receiver的函数体
    r_end_flags = [hls.HLSVar(f"in{i+1}_end_flag", bool_type) for i in range(4)]

    def create_receiver_read_block(in_stream_var, out_stream_var, flag_var):
        data_var = hls.HLSVar("data", data_tuple_type)
        end_check_expr = hls.HLSExpr(
            hls.HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "end_flag"),
            [hls.HLSExpr(hls.HLSExprT.VAR, data_var)],
        )
        if_not_end_block = hls.CodeIf(
            hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check_expr]),
            if_codes=[hls.CodeWriteStream(out_stream_var, data_var)],
            else_codes=[hls.CodeAssign(flag_var, end_check_expr)],
        )
        return create_non_blocking_read(
            in_stream_var,
            [
                hls.CodeVarDecl(data_var.name, data_var.type),
                hls.CodeAssign(
                    data_var,
                    hls.HLSExpr(
                        hls.HLSExprT.STREAM_READ,
                        None,
                        [hls.HLSExpr(hls.HLSExprT.VAR, in_stream_var)],
                    ),
                ),
                if_not_end_block,
            ],
        )

    # 构建合并逻辑
    merge_logic_1 = create_receiver_read_block(
        receiver_params[3], receiver_params[1], r_end_flags[0]
    )
    merge_logic_1.elifs.append(
        (
            hls.HLSExpr(
                hls.HLSExprT.STREAM_EMPTY,
                None,
                [hls.HLSExpr(hls.HLSExprT.VAR, receiver_params[5])],
            ),  # if in3 is not empty
            create_receiver_read_block(
                receiver_params[5], receiver_params[1], r_end_flags[2]
            ).if_codes,
        )
    )
    # a bit of hack since the `create_receiver_read_block` creates a whole if block, we only need the body.
    # The empty check is incorrect in the above elif, it should be `!empty`.
    # Let's rebuild it manually.

    in1_cond = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [
            hls.HLSExpr(
                hls.HLSExprT.STREAM_EMPTY,
                None,
                [hls.HLSExpr(hls.HLSExprT.VAR, receiver_params[3])],
            )
        ],
    )
    in3_cond = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [
            hls.HLSExpr(
                hls.HLSExprT.STREAM_EMPTY,
                None,
                [hls.HLSExpr(hls.HLSExprT.VAR, receiver_params[5])],
            )
        ],
    )
    in2_cond = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [
            hls.HLSExpr(
                hls.HLSExprT.STREAM_EMPTY,
                None,
                [hls.HLSExpr(hls.HLSExprT.VAR, receiver_params[4])],
            )
        ],
    )
    in4_cond = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [
            hls.HLSExpr(
                hls.HLSExprT.STREAM_EMPTY,
                None,
                [hls.HLSExpr(hls.HLSExprT.VAR, receiver_params[6])],
            )
        ],
    )

    merge_block_1_body = create_receiver_read_block(
        receiver_params[3], receiver_params[1], r_end_flags[0]
    ).if_codes
    merge_block_3_body = create_receiver_read_block(
        receiver_params[5], receiver_params[1], r_end_flags[2]
    ).if_codes
    merge13 = hls.CodeIf(in1_cond, merge_block_1_body, elifs=[(in3_cond, merge_block_3_body)])

    merge_block_2_body = create_receiver_read_block(
        receiver_params[4], receiver_params[2], r_end_flags[1]
    ).if_codes
    merge_block_4_body = create_receiver_read_block(
        receiver_params[6], receiver_params[2], r_end_flags[3]
    ).if_codes
    merge24 = hls.CodeIf(in2_cond, merge_block_2_body, elifs=[(in4_cond, merge_block_4_body)])

    # 退出逻辑
    r_exit_cond = hls.HLSExpr(hls.HLSExprT.VAR, r_end_flags[0])
    for i in range(1, 4):
        r_exit_cond = hls.HLSExpr(
            hls.HLSExprT.BINOP,
            dfir.BinOp.AND,
            [r_exit_cond, hls.HLSExpr(hls.HLSExprT.VAR, r_end_flags[i])],
        )

    r_exit_block = hls.CodeIf(
        r_exit_cond,
        [
            hls.CodeVarDecl(end_data_var.name, end_data_var.type),
            hls.CodeAssign(
                hls.HLSVar(f"{end_data_var.name}.end_flag", bool_type),
                hls.HLSExpr(hls.HLSExprT.CONST, True),
            ),
            hls.CodeWriteStream(receiver_params[1], end_data_var),
            hls.CodeWriteStream(receiver_params[2], end_data_var),
            hls.CodeBreak(),
        ],
    )

    receiver_body = [
        hls.CodePragma(f"function_instantiate variable=i"),
    ]
    receiver_body.extend([hls.CodeVarDecl(v.name, v.type) for v in r_end_flags])
    receiver_body.append(
        hls.CodeWhile(
            iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True),
            codes=[hls.CodePragma("PIPELINE II=1"), merge13, merge24, r_exit_block],
        )
    )

    receiver_function = hls.HLSFunction(name=receiver_func_name, comp=None)
    receiver_function.params = receiver_params
    receiver_function.codes = receiver_body

    # --- 5. 生成 'switch2x2' 函数 ---
    switch_func_name = f"switch2x2_{gen_id}"

    switch_params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("update_set_stm_in1", stream_type),
        hls.HLSVar("update_set_stm_in2", stream_type),
        hls.HLSVar("update_set_stm_out1", stream_type),
        hls.HLSVar("update_set_stm_out2", stream_type),
    ]

    local_streams = [hls.HLSVar(f"l1_{i+1}", stream_type) for i in range(4)]

    switch_body = [hls.CodePragma("DATAFLOW")]
    for stream_var in local_streams:
        switch_body.append(hls.CodeVarDecl(stream_var.name, stream_var.type))
        switch_body.append(hls.CodePragma(f"STREAM variable={stream_var.name} depth=2"))

    # 调用 sender 和 receiver
    sender_call = hls.CodeCall(
        sender_function,
        [
            switch_params[0],  # i
            switch_params[1],  # in1
            switch_params[2],  # in2
            local_streams[0],  # l1_1 for sender's out1
            local_streams[1],  # l1_2 for sender's out2
            local_streams[2],  # l1_3 for sender's out3
            local_streams[3],  # l1_4 for sender's out4
        ],
    )
    receiver_call = hls.CodeCall(
        receiver_function,
        [
            switch_params[0],  # i
            switch_params[3],  # out1
            switch_params[4],  # out2
            local_streams[0],  # l1_1 for receiver's in1
            local_streams[1],  # l1_2 for receiver's in2
            local_streams[2],  # l1_3 for receiver's in3
            local_streams[3],  # l1_4 for receiver's in4
        ],
    )
    switch_body.extend([sender_call, receiver_call])

    switch2x2_function = hls.HLSFunction(name=switch_func_name, comp=None)
    switch2x2_function.params = switch_params
    switch2x2_function.codes = switch_body

    # --- 6. 生成 'omega_switch' 顶层函数 ---
    top_func_name = f"omega_switch_{gen_id}"

    top_ins = [hls.HLSVar(f"in_{i}", stream_type) for i in range(n)]
    top_outs = [hls.HLSVar(f"out_{i}", stream_type) for i in range(n)]

    top_body = [hls.CodePragma("DATAFLOW")]

    # 定义所有中间流
    # streams[stage][line]
    intermediate_streams = []
    for s in range(log_n - 1):
        stage_streams = [hls.HLSVar(f"stream_{s}_{k}", stream_type) for k in range(n)]
        intermediate_streams.append(stage_streams)
        for var in stage_streams:
            top_body.append(hls.CodeVarDecl(var.name, var.type))
            top_body.append(hls.CodePragma(f"STREAM variable={var.name} depth=2"))

    # 辅助函数：完美反混洗 (Right-shift)
    def unshuffle(p: int, num_bits: int) -> int:
        mask = (1 << num_bits) - 1
        return ((p & 1) << (num_bits - 1)) | (p >> 1)

    # 循环生成所有 switch2x2 调用
    for s in range(log_n):
        for j in range(switches_per_stage):
            in1_idx = 2 * j
            in2_idx = 2 * j + 1

            # 确定输入流
            if s == 0:
                in1_var = top_ins[in1_idx]
                in2_var = top_ins[in2_idx]
            else:
                unshuffled_idx1 = unshuffle(in1_idx, log_n)
                unshuffled_idx2 = unshuffle(in2_idx, log_n)
                in1_var = intermediate_streams[s - 1][unshuffled_idx1]
                in2_var = intermediate_streams[s - 1][unshuffled_idx2]

            # 确定输出流
            if s == log_n - 1:
                out1_var = top_outs[in1_idx]
                out2_var = top_outs[in2_idx]
            else:
                out1_var = intermediate_streams[s][in1_idx]
                out2_var = intermediate_streams[s][in2_idx]

            # 生成调用
            call = hls.CodeCall(
                switch2x2_function,
                [
                    hls.HLSExpr(hls.HLSExprT.CONST, s),  # stage number 'i'
                    in1_var,
                    in2_var,
                    out1_var,
                    out2_var,
                ],
            )
            top_body.append(call)

    omega_switch_function = hls.HLSFunction(name=top_func_name, comp=None)
    omega_switch_function.params = top_ins + top_outs
    omega_switch_function.codes = top_body

    # --- 7. 组装最终的C++代码 ---
    all_functions = [sender_function, receiver_function, switch2x2_function, omega_switch_function]
    return all_functions

    # 生成文件头
    # final_code = (
    #     f"// Omega Network (N={n}) - Generated by hls_network_generators.py\n"
    #     f"// Generation ID: {gen_id}\n\n"
    #     "#include <hls_stream.h>\n"
    #     "#include <stdint.h>\n\n"
    #     f"// User must ensure this data type is defined, for example:\n"
    #     f"// typedef struct {{\n"
    #     f"//     uint32_t dst;\n"
    #     f"//     bool end_flag;\n"
    #     f"//     // ... other members\n"
    #     f"// }} {data_type_name};\n\n"
    # )
    # final_code = ""

    # # 生成所有函数定义
    # for func in all_functions:
    #     params_str = ", ".join([f"{p.type.name}& {p.name}" for p in func.params])
    #     # 对于顶层函数，参数可能不都是引用
    #     if func == omega_switch_function:
    #         params_str = ", ".join([f"{p.type.name} {p.name}" for p in func.params])

    #     func_def = f"void {func.name}({params_str}) {{\n"
    #     func_body_code = "".join([line.gen_code(indent_lvl=1) for line in func.codes])
    #     func_def += func_body_code
    #     func_def += "}\n\n"
    #     final_code += func_def

    # return final_code


# if __name__ == "__main__":
#     # --- 使用示例 ---
#     # 定义网络大小和数据类型名称
#     N = 16
#     DATA_TYPE = "update_tuple_dt"

#     print(f"--- Generating HLS code for an {N}x{N} Omega Network ---")

#     try:
#         generated_cpp_code = generate_omega_network(n=N, data_type_name=DATA_TYPE)

#         # 将生成的代码写入文件
#         # output_filename = f"omega_network_{N}x{N}.cpp"
#         # with open(output_filename, "w") as f:
#         #     f.write(generated_cpp_code)

#         # print(f"\nSuccessfully generated HLS code in '{output_filename}'")

#         # 也可以直接打印到控制台
#         print("\n--- Generated Code ---")
#         print(generated_cpp_code)

#     except ValueError as e:
#         print(f"Error: {e}")

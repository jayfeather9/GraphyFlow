# backend_utils.py
from __future__ import annotations
import math
from typing import List

# 从您现有的后端和IR文件中导入必要的类
# 注意：我们现在从 backend_defines 导入基础类
import graphyflow.backend_defines as hls
import graphyflow.dataflow_ir as dfir

# 全局计数器，确保每次生成的函数名都唯一
_generator_id_counter = 0


def _get_unique_id() -> int:
    """获取一个唯一的ID用于函数命名。"""
    global _generator_id_counter
    id = _generator_id_counter
    _generator_id_counter += 1
    return id


def create_non_blocking_read(stream_var: hls.HLSVar, body_if_not_empty: List[hls.HLSCodeLine]) -> hls.CodeIf:
    """
    一个辅助函数，用于快速生成非阻塞读数据流的代码块。
    """
    empty_expr = hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, stream_var)])
    not_empty_expr = hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [empty_expr])
    # 修复了构造函数调用，确保默认参数安全
    return hls.CodeIf(expr=not_empty_expr, if_codes=body_if_not_empty)


# ======================================================================== #
#                      第二步：新增函数生成器                                #
# ======================================================================== #


def generate_merge_stream_2x1(data_type: hls.HLSType) -> hls.HLSFunction:
    """
    生成一个2合1数据流合并单元 (mergeStream2x1) 的 HLSFunction 对象。
    这个函数是构成归约树的基础。
    """
    gen_id = _get_unique_id()
    func_name = f"mergeStream2x1_{gen_id}"

    stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[data_type])
    bool_type = hls.HLSType(hls.HLSBasicType.BOOL)
    int_type = hls.HLSType(hls.HLSBasicType.INT)

    # 定义函数参数
    params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("in1", stream_type),
        hls.HLSVar("in2", stream_type),
        hls.HLSVar("out", stream_type),
    ]

    # 定义函数体内的变量
    in1_end_flag_var = hls.HLSVar("in1_end_flag", bool_type)
    in2_end_flag_var = hls.HLSVar("in2_end_flag", bool_type)
    data1_var = hls.HLSVar("data1", data_type)
    data2_var = hls.HLSVar("data2", data_type)

    # 为了模拟 read_from_stream_nb，我们需要一个辅助函数来生成读取和检查的逻辑
    def create_nb_read_logic(in_stream_var, data_var, process_flag_var, end_flag_var):
        end_check_expr = hls.HLSExpr(
            hls.HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "end_flag"),
            [hls.HLSExpr(hls.HLSExprT.VAR, data_var)],
        )

        # if (data.end_flag) inX_end_flag = 1;
        set_end_flag = hls.CodeIf(
            end_check_expr, [hls.CodeAssign(end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))]
        )

        read_body = [
            hls.CodeAssign(process_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True)),
            hls.CodeVarDecl(data_var.name, data_var.type),
            hls.CodeAssign(
                data_var,
                hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, in_stream_var)]),
            ),
            set_end_flag,
        ]

        return create_non_blocking_read(in_stream_var, read_body)

    # 构建主循环体
    p1_flag = hls.HLSVar("in1_process_flag", bool_type)
    p2_flag = hls.HLSVar("in2_process_flag", bool_type)

    # if(in1_process_flag && (!in1_end_flag)) write_to_stream(out, data1);
    write_cond1 = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, p1_flag),
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                dfir.UnaryOp.NOT,
                [hls.HLSExpr(hls.HLSExprT.VAR, in1_end_flag_var)],
            ),
        ],
    )
    write_block1 = hls.CodeIf(write_cond1, [hls.CodeWriteStream(params[3], data1_var)])

    write_cond2 = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, p2_flag),
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                dfir.UnaryOp.NOT,
                [hls.HLSExpr(hls.HLSExprT.VAR, in2_end_flag_var)],
            ),
        ],
    )
    write_block2 = hls.CodeIf(write_cond2, [hls.CodeWriteStream(params[3], data2_var)])

    # 退出逻辑
    exit_cond = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, in1_end_flag_var),
            hls.HLSExpr(hls.HLSExprT.VAR, in2_end_flag_var),
        ],
    )
    end_data_var = hls.HLSVar("data", data_type)
    exit_block = hls.CodeIf(
        exit_cond,
        [
            hls.CodeVarDecl(end_data_var.name, end_data_var.type),
            hls.CodeAssign(
                hls.HLSVar(f"{end_data_var.name}.end_flag", bool_type),
                hls.HLSExpr(hls.HLSExprT.CONST, True),
            ),
            hls.CodeWriteStream(params[3], end_data_var),
            hls.CodeAssign(in1_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeAssign(in2_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeBreak(),
        ],
    )

    while_body = [
        hls.CodePragma("PIPELINE II=2"),
        # 初始化 process flags
        hls.CodeAssign(p1_flag, hls.HLSExpr(hls.HLSExprT.CONST, False)),
        hls.CodeAssign(p2_flag, hls.HLSExpr(hls.HLSExprT.CONST, False)),
        # 读取
        create_nb_read_logic(params[1], data1_var, p1_flag, in1_end_flag_var),
        create_nb_read_logic(params[2], data2_var, p2_flag, in2_end_flag_var),
        # 写入
        write_block1,
        write_block2,
        # 退出
        exit_block,
    ]

    func_body = [
        hls.CodePragma("function_instantiate variable=i"),
        hls.CodeVarDecl(in1_end_flag_var.name, in1_end_flag_var.type),
        hls.CodeVarDecl(in2_end_flag_var.name, in2_end_flag_var.type),
        hls.CodeVarDecl(p1_flag.name, p1_flag.type),
        hls.CodeVarDecl(p2_flag.name, p2_flag.type),
        hls.CodeWhile(codes=while_body, iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True)),
    ]

    merge_func = hls.HLSFunction(name=func_name, comp=None)
    merge_func.params = params
    merge_func.codes = func_body
    return merge_func


def generate_reduction_tree(n: int, data_type: hls.HLSType, merge_func: hls.HLSFunction) -> hls.HLSFunction:
    """
    生成一个 N->1 的归约树。
    """
    if not (n > 0 and (n & (n - 1) == 0)):
        raise ValueError("归约树的输入数量 'n' 必须是2的幂。")

    gen_id = _get_unique_id()
    func_name = f"reductionTree_{gen_id}"
    log_n = int(math.log2(n))

    # 定义函数参数
    stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[data_type])
    stream_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[n])

    params = [
        hls.HLSVar("i", hls.HLSType(hls.HLSBasicType.INT)),
        # *** 接口统一：使用流数组 ***
        hls.HLSVar("update_set_stm", stream_array_type),
        hls.HLSVar("reduced_update_tuple_stm", stream_type),
    ]

    # 定义函数体
    body = [hls.CodePragma("DATAFLOW")]

    # 声明所有中间流
    streams_by_level = {}
    num_streams = n
    for level in range(log_n - 1):  # Only need log_n - 1 levels of intermediate streams
        num_streams //= 2
        if num_streams == 0:
            break
        level_stream_type = hls.HLSType(
            hls.HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[num_streams]
        )
        level_stream_var = hls.HLSVar(f"l{level+1}_update_tuples", level_stream_type)
        streams_by_level[level + 1] = level_stream_var
        body.append(hls.CodeVarDecl(level_stream_var.name, level_stream_var.type))
        body.append(hls.CodePragma(f"STREAM variable={level_stream_var.name} depth=2"))

    # 生成 mergeStream2x1 调用
    num_mergers_at_level = n // 2
    input_streams = params[1]  # The initial array of streams
    for level in range(log_n):
        output_streams = streams_by_level.get(level + 1, params[2])  # Final output is the function param
        for i in range(num_mergers_at_level):
            in1_var = hls.HLSVar(f"{input_streams.name}[{i*2}]", stream_type)
            in2_var = hls.HLSVar(f"{input_streams.name}[{i*2 + 1}]", stream_type)

            # 如果是最后一级，输出是单个流，否则是数组中的一个元素
            if level == log_n - 1:
                out_var = output_streams
            else:
                out_var = hls.HLSVar(f"{output_streams.name}[{i}]", stream_type)

            call = hls.CodeCall(
                merge_func,
                [
                    hls.HLSExpr(hls.HLSExprT.CONST, i),  # 'i' for instantiation
                    in1_var,
                    in2_var,
                    out_var,
                ],
            )
            body.append(call)

        input_streams = output_streams
        num_mergers_at_level //= 2

    tree_func = hls.HLSFunction(name=func_name, comp=None)
    tree_func.params = params
    tree_func.codes = body
    return tree_func


def generate_demux(n: int, batch_type: hls.HLSType, wrapper_type: hls.HLSType) -> hls.HLSFunction:
    """
    生成一个 1->N 的数据流拆分器 (Demux/Unbatcher)。
    它接收批处理流，输出N个独立流。
    """
    gen_id = _get_unique_id()
    func_name = f"demux_{gen_id}"

    in_stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[batch_type])
    out_stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[wrapper_type])
    out_stream_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[out_stream_type], array_dims=[n])

    params = [
        hls.HLSVar("in_batch_stream", in_stream_type),
        hls.HLSVar("out_streams", out_stream_array_type),
    ]

    in_batch_var = hls.HLSVar("in_batch", batch_type)
    wrapper_var = hls.HLSVar("wrapper_data", wrapper_type)

    # --- 修正后的内部循环逻辑 ---
    # 仅当 `i < in_batch.end_pos` 时才分发数据
    inner_loop_body_if = [
        hls.CodeAssign(
            hls.HLSVar(f"{wrapper_var.name}.data", wrapper_type.sub_types[0]),
            hls.HLSExpr(
                hls.HLSExprT.VAR, hls.HLSVar(f"{in_batch_var.name}.data[i]", wrapper_type.sub_types[0])
            ),
        ),
        hls.CodeAssign(
            hls.HLSVar(f"{wrapper_var.name}.end_flag", hls.HLSType(hls.HLSBasicType.BOOL)),
            hls.HLSExpr(hls.HLSExprT.CONST, False),
        ),
        hls.CodeWriteStream(hls.HLSVar(f"out_streams[i]", out_stream_type), wrapper_var),
    ]

    cond_expr = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.LT,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, hls.HLSVar("i", hls.HLSType(hls.HLSBasicType.UINT))),
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "end_pos"),
                [hls.HLSExpr(hls.HLSExprT.VAR, in_batch_var)],
            ),
        ],
    )

    inner_for_loop = hls.CodeFor(
        [hls.CodePragma("UNROLL"), hls.CodeIf(cond_expr, inner_loop_body_if)],
        iter_limit="PE_NUM",
        iter_name="i",
    )

    # While 循环体
    while_body = [
        hls.CodePragma("PIPELINE"),
        hls.CodeAssign(
            in_batch_var,
            hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[0])]),
        ),
        hls.CodeVarDecl(wrapper_var.name, wrapper_type),
        inner_for_loop,
        hls.CodeIf(
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "end_flag"),
                [hls.HLSExpr(hls.HLSExprT.VAR, in_batch_var)],
            ),
            [hls.CodeBreak()],
        ),
    ]

    # 结束标志传播逻辑
    end_wrapper_var = hls.HLSVar("end_wrapper", wrapper_type)
    final_loop = hls.CodeFor(
        [
            hls.CodePragma("UNROLL"),
            hls.CodeWriteStream(hls.HLSVar(f"out_streams[i]", out_stream_type), end_wrapper_var),
        ],
        iter_limit=n,
        iter_name="i",
    )

    body = [
        hls.CodeVarDecl(in_batch_var.name, in_batch_var.type),
        hls.CodeWhile(codes=while_body, iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True)),
        hls.CodeComment("Propagate end_flag to all output streams"),
        hls.CodeVarDecl(end_wrapper_var.name, end_wrapper_var.type),
        hls.CodeAssign(
            hls.HLSVar(f"{end_wrapper_var.name}.end_flag", hls.HLSType(hls.HLSBasicType.BOOL)),
            hls.HLSExpr(hls.HLSExprT.CONST, True),
        ),
        final_loop,
    ]

    demux_func = hls.HLSFunction(name=func_name, comp=None)
    demux_func.params = params
    demux_func.codes = body
    return demux_func


# ======================================================================== #
#                      OMEGA NETWORK GENERATOR (FULL)                      #
# ======================================================================== #


def generate_omega_network(n: int, wrapper_type, routing_key_member: str) -> List[hls.HLSFunction]:
    """
    生成一个 N x N Omega 网络的完整HLS C++代码，包括所有子模块。
    """
    if not (n > 0 and (n & (n - 1) == 0)):
        raise ValueError("网络大小 'n' 必须是2的正整数次幂。")

    log_n = int(math.log2(n))
    switches_per_stage = n // 2
    gen_id = _get_unique_id()

    # --- HLS类型定义 ---

    # b. 其他类型现在基于 wrapper_type
    stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[wrapper_type])
    bool_type = hls.HLSType(hls.HLSBasicType.BOOL)
    int_type = hls.HLSType(hls.HLSBasicType.INT)
    stream_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[n])

    # --- 调用子模块生成器 ---
    sender_function = generate_omega_sender(
        gen_id, wrapper_type, stream_type, int_type, bool_type, routing_key_member
    )
    receiver_function = generate_omega_receiver(gen_id, wrapper_type, stream_type, int_type, bool_type)
    switch2x2_function = generate_omega_switch2x2(
        gen_id, stream_type, int_type, sender_function, receiver_function
    )
    omega_switch_function = generate_omega_top(
        gen_id, n, log_n, n // 2, stream_array_type, stream_type, switch2x2_function
    )

    return [sender_function, receiver_function, switch2x2_function, omega_switch_function]


# Helper functions to keep generate_omega_network clean
def generate_omega_sender(gen_id, data_tuple_type, stream_type, int_type, bool_type, routing_key_member: str):
    func_name = f"sender_{gen_id}"
    params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("in1", stream_type),
        hls.HLSVar("in2", stream_type),
        hls.HLSVar("out1", stream_type),
        hls.HLSVar("out2", stream_type),
        hls.HLSVar("out3", stream_type),
        hls.HLSVar("out4", stream_type),
    ]
    in1_end_flag_var = hls.HLSVar("in1_end_flag", bool_type)
    in2_end_flag_var = hls.HLSVar("in2_end_flag", bool_type)
    data1_var, data2_var = hls.HLSVar("data1", data_tuple_type), hls.HLSVar("data2", data_tuple_type)

    def create_routing_expr(data_var):
        inner_data_expr = hls.HLSExpr(
            hls.HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "data"),
            [hls.HLSExpr(hls.HLSExprT.VAR, data_var)],
        )
        key_expr = hls.HLSExpr(
            hls.HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, routing_key_member), [inner_data_expr]
        )

        i_var_expr = hls.HLSExpr(hls.HLSExprT.VAR, hls.HLSVar("i", int_type))
        shifted_expr = hls.HLSExpr(hls.HLSExprT.BINOP, dfir.BinOp.SR, [key_expr, i_var_expr])
        return hls.HLSExpr(
            hls.HLSExprT.BINOP, dfir.BinOp.AND, [shifted_expr, hls.HLSExpr(hls.HLSExprT.CONST, 1)]
        )

    # Logic for in1
    end_check1 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, data1_var)],
    )
    route_if1 = hls.CodeIf(
        create_routing_expr(data1_var),
        if_codes=[hls.CodeWriteStream(params[4], data1_var)],
        else_codes=[hls.CodeWriteStream(params[3], data1_var)],
    )
    process_if1 = hls.CodeIf(
        hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check1]),
        if_codes=[route_if1],
        else_codes=[hls.CodeAssign(in1_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))],
    )
    nb_read1 = create_non_blocking_read(
        params[1],
        [
            hls.CodeVarDecl(data1_var.name, data1_var.type),
            hls.CodeAssign(
                data1_var,
                hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[1])]),
            ),
            process_if1,
        ],
    )

    # Logic for in2
    end_check2 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, data2_var)],
    )
    route_if2 = hls.CodeIf(
        create_routing_expr(data2_var),
        if_codes=[hls.CodeWriteStream(params[6], data2_var)],
        else_codes=[hls.CodeWriteStream(params[5], data2_var)],
    )
    process_if2 = hls.CodeIf(
        hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check2]),
        if_codes=[route_if2],
        else_codes=[hls.CodeAssign(in2_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))],
    )
    nb_read2 = create_non_blocking_read(
        params[2],
        [
            hls.CodeVarDecl(data2_var.name, data2_var.type),
            hls.CodeAssign(
                data2_var,
                hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[2])]),
            ),
            process_if2,
        ],
    )

    # Exit logic
    exit_cond = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, in1_end_flag_var),
            hls.HLSExpr(hls.HLSExprT.VAR, in2_end_flag_var),
        ],
    )
    end_data_var = hls.HLSVar("data", data_tuple_type)
    exit_block = hls.CodeIf(
        exit_cond,
        [
            hls.CodeVarDecl(end_data_var.name, end_data_var.type),
            hls.CodeAssign(
                hls.HLSVar(f"{end_data_var.name}.end_flag", bool_type),
                hls.HLSExpr(hls.HLSExprT.CONST, True),
            ),
            hls.CodeWriteStream(params[3], end_data_var),
            hls.CodeWriteStream(params[4], end_data_var),
            hls.CodeWriteStream(params[5], end_data_var),
            hls.CodeWriteStream(params[6], end_data_var),
            hls.CodeAssign(in1_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeAssign(in2_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeBreak(),
        ],
    )

    body = [
        hls.CodePragma("function_instantiate variable=i"),
        hls.CodeVarDecl(in1_end_flag_var.name, in1_end_flag_var.type, init_val="false"),
        hls.CodeVarDecl(in2_end_flag_var.name, in2_end_flag_var.type, init_val="false"),
        hls.CodeWhile(
            [hls.CodePragma("PIPELINE II=1"), nb_read1, nb_read2, exit_block],
            hls.HLSExpr(hls.HLSExprT.CONST, True),
        ),
    ]
    func = hls.HLSFunction(name=func_name, comp=None)
    func.params, func.codes = params, body
    return func


def generate_omega_receiver(gen_id, data_tuple_type, stream_type, int_type, bool_type):
    func_name = f"receiver_{gen_id}"
    params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("out1", stream_type),
        hls.HLSVar("out2", stream_type),
        hls.HLSVar("in1", stream_type),
        hls.HLSVar("in2", stream_type),
        hls.HLSVar("in3", stream_type),
        hls.HLSVar("in4", stream_type),
    ]
    end_flags = [hls.HLSVar(f"in{i+1}_end_flag", bool_type) for i in range(4)]

    def get_read_body(in_stream, out_stream, flag_var):
        data_var = hls.HLSVar("data", data_tuple_type)
        end_check = hls.HLSExpr(
            hls.HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "end_flag"),
            [hls.HLSExpr(hls.HLSExprT.VAR, data_var)],
        )
        process_if = hls.CodeIf(
            hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check]),
            if_codes=[hls.CodeWriteStream(out_stream, data_var)],
            else_codes=[hls.CodeAssign(flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))],
        )
        return [
            hls.CodeVarDecl(data_var.name, data_var.type),
            hls.CodeAssign(
                data_var,
                hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, in_stream)]),
            ),
            process_if,
        ]

    # Merge logic
    cond1 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[3])])],
    )
    cond3 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[5])])],
    )
    merge13 = hls.CodeIf(
        cond1,
        get_read_body(params[3], params[1], end_flags[0]),
        elifs=[(cond3, get_read_body(params[5], params[1], end_flags[2]))],
    )

    cond2 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[4])])],
    )
    cond4 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[6])])],
    )
    merge24 = hls.CodeIf(
        cond2,
        get_read_body(params[4], params[2], end_flags[1]),
        elifs=[(cond4, get_read_body(params[6], params[2], end_flags[3]))],
    )

    # Exit logic
    exit_cond = hls.HLSExpr(hls.HLSExprT.VAR, end_flags[0])
    for i in range(1, 4):
        exit_cond = hls.HLSExpr(
            hls.HLSExprT.BINOP,
            dfir.BinOp.AND,
            [exit_cond, hls.HLSExpr(hls.HLSExprT.VAR, end_flags[i])],
        )
    end_data_var = hls.HLSVar("data", data_tuple_type)
    exit_block = hls.CodeIf(
        exit_cond,
        [
            hls.CodeVarDecl(end_data_var.name, end_data_var.type),
            hls.CodeAssign(
                hls.HLSVar(f"{end_data_var.name}.end_flag", bool_type),
                hls.HLSExpr(hls.HLSExprT.CONST, True),
            ),
            hls.CodeWriteStream(params[1], end_data_var),
            hls.CodeWriteStream(params[2], end_data_var),
            hls.CodeBreak(),
        ],
    )

    body = [hls.CodePragma("function_instantiate variable=i")]
    body.extend([hls.CodeVarDecl(v.name, v.type, init_val="false") for v in end_flags])
    body.append(
        hls.CodeWhile(
            iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True),
            codes=[hls.CodePragma("PIPELINE II=1"), merge13, merge24, exit_block],
        )
    )

    func = hls.HLSFunction(name=func_name, comp=None)
    func.params, func.codes = params, body
    return func


def generate_omega_switch2x2(gen_id, stream_type, int_type, sender_func, receiver_func):
    func_name = f"switch2x2_{gen_id}"
    params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("in1", stream_type),
        hls.HLSVar("in2", stream_type),
        hls.HLSVar("out1", stream_type),
        hls.HLSVar("out2", stream_type),
    ]
    local_streams = [hls.HLSVar(f"l1_{i+1}", stream_type) for i in range(4)]
    body = [hls.CodePragma("DATAFLOW")]
    for stream_var in local_streams:
        body.append(hls.CodeVarDecl(stream_var.name, stream_var.type))
        body.append(hls.CodePragma(f"STREAM variable={stream_var.name} depth=2"))

    sender_call = hls.CodeCall(
        sender_func,
        [
            params[0],
            params[1],
            params[2],
            local_streams[0],
            local_streams[1],
            local_streams[2],
            local_streams[3],
        ],
    )
    receiver_call = hls.CodeCall(
        receiver_func,
        [
            params[0],
            params[3],
            params[4],
            local_streams[0],
            local_streams[1],
            local_streams[2],
            local_streams[3],
        ],
    )
    body.extend([sender_call, receiver_call])

    func = hls.HLSFunction(name=func_name, comp=None)
    func.params, func.codes = params, body
    return func


def generate_omega_top(gen_id, n, log_n, switches_per_stage, stream_array_type, stream_type, switch2x2_func):
    func_name = f"omega_switch_{gen_id}"
    params = [
        hls.HLSVar("in_streams", stream_array_type),
        hls.HLSVar("out_streams", stream_array_type),
    ]
    body = [hls.CodePragma("DATAFLOW")]

    intermediate_streams = []
    for s in range(log_n - 1):
        stage_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[n])
        stage_var = hls.HLSVar(f"stream_stage_{s}", stage_array_type)
        intermediate_streams.append(stage_var)
        body.append(hls.CodeVarDecl(stage_var.name, stage_var.type))
        body.append(hls.CodePragma(f"STREAM variable={stage_var.name} depth=2"))

    def unshuffle(p, num_bits):
        return ((p & 1) << (num_bits - 1)) | (p >> 1)

    for s in range(log_n):
        for j in range(switches_per_stage):
            idx1, idx2 = 2 * j, 2 * j + 1
            if s == 0:
                in1_var, in2_var = hls.HLSVar(f"in_streams[{idx1}]", stream_type), hls.HLSVar(
                    f"in_streams[{idx2}]", stream_type
                )
            else:
                uidx1, uidx2 = unshuffle(idx1, log_n), unshuffle(idx2, log_n)
                in_array = intermediate_streams[s - 1]
                in1_var, in2_var = hls.HLSVar(f"{in_array.name}[{uidx1}]", stream_type), hls.HLSVar(
                    f"{in_array.name}[{uidx2}]", stream_type
                )

            if s == log_n - 1:
                out1_var, out2_var = hls.HLSVar(f"out_streams[{idx1}]", stream_type), hls.HLSVar(
                    f"out_streams[{idx2}]", stream_type
                )
            else:
                out_array = intermediate_streams[s]
                out1_var, out2_var = hls.HLSVar(f"{out_array.name}[{idx1}]", stream_type), hls.HLSVar(
                    f"{out_array.name}[{idx2}]", stream_type
                )

            call = hls.CodeCall(
                switch2x2_func,
                [
                    hls.HLSExpr(hls.HLSExprT.CONST, log_n - s - 1),
                    in1_var,
                    in2_var,
                    out1_var,
                    out2_var,
                ],
            )
            body.append(call)

    func = hls.HLSFunction(name=func_name, comp=None)
    func.params, func.codes = params, body
    return func


def generate_stream_zipper(
    key_batch_type: hls.HLSType,
    transform_batch_type: hls.HLSType,
    out_kt_pair_batch_type: hls.HLSType,
) -> hls.HLSFunction:
    """
    生成一个流合并器 (Stream Zipper)。
    输入: 两个批处理流 (key, transform)
    输出: 一个合并了key和transform的批处理流 (kt_pair)
    """
    gen_id = _get_unique_id()
    func_name = f"stream_zipper_{gen_id}"

    params = [
        hls.HLSVar("in_key_batch_stream", hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[key_batch_type])),
        hls.HLSVar(
            "in_transform_batch_stream",
            hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[transform_batch_type]),
        ),
        hls.HLSVar(
            "out_pair_batch_stream",
            hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[out_kt_pair_batch_type]),
        ),
    ]

    key_batch_var = hls.HLSVar("key_batch", key_batch_type)
    transform_batch_var = hls.HLSVar("transform_batch", transform_batch_type)
    out_batch_var = hls.HLSVar("out_batch", out_kt_pair_batch_type)

    # 内部循环: out.data[i].key = key.data[i]; out.data[i].transform = transform.data[i];
    kt_pair_type = out_kt_pair_batch_type.sub_types[0].sub_types[0]
    assign_key = hls.CodeAssign(
        hls.HLSVar(f"{out_batch_var.name}.data[i].key", kt_pair_type.sub_types[0]),
        hls.HLSVar(f"{key_batch_var.name}.data[i]", key_batch_type.sub_types[0].sub_types[0]),
    )
    assign_transform = hls.CodeAssign(
        hls.HLSVar(f"{out_batch_var.name}.data[i].transform", kt_pair_type.sub_types[1]),
        hls.HLSVar(f"{transform_batch_var.name}.data[i]", transform_batch_type.sub_types[0].sub_types[0]),
    )
    for_loop = hls.CodeFor(
        [hls.CodePragma("UNROLL"), assign_key, assign_transform],
        iter_limit="PE_NUM",
        iter_name="i",
    )

    # while循环体
    end_flag_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, key_batch_var)],
    )
    end_pos_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_pos"),
        [hls.HLSExpr(hls.HLSExprT.VAR, key_batch_var)],
    )

    while_body = [
        hls.CodePragma("PIPELINE"),
        hls.CodeAssign(
            key_batch_var,
            hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[0])]),
        ),
        hls.CodeAssign(
            transform_batch_var,
            hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[1])]),
        ),
        for_loop,
        hls.CodeAssign(
            hls.HLSVar(f"{out_batch_var.name}.end_flag", hls.HLSType(hls.HLSBasicType.BOOL)),
            end_flag_expr,
        ),
        hls.CodeAssign(
            hls.HLSVar(f"{out_batch_var.name}.end_pos", hls.HLSType(hls.HLSBasicType.UINT8)),
            end_pos_expr,
        ),
        hls.CodeWriteStream(params[2], out_batch_var),
        hls.CodeIf(end_flag_expr, [hls.CodeBreak()]),
    ]

    body = [
        hls.CodeVarDecl(key_batch_var.name, key_batch_var.type),
        hls.CodeVarDecl(transform_batch_var.name, transform_batch_var.type),
        hls.CodeVarDecl(out_batch_var.name, out_batch_var.type),
        hls.CodeWhile(iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True), codes=while_body),
    ]

    zipper_func = hls.HLSFunction(name=func_name, comp=None)
    zipper_func.params = params
    zipper_func.codes = body
    return zipper_func


def generate_stream_unzipper(
    n: int,
    wrapped_kt_pair_type: hls.HLSType,
    out_key_stream_array_type: hls.HLSType,
    out_transform_stream_array_type: hls.HLSType,
) -> hls.HLSFunction:
    """
    生成一个流拆分器 (Stream Unzipper)。
    输入: 一个流数组 (承载 wrapper<kt_pair>)
    输出: 两个流数组 (key, transform)
    """
    gen_id = _get_unique_id()
    func_name = f"stream_unzipper_{gen_id}"

    in_stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[wrapped_kt_pair_type])
    in_stream_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[in_stream_type], array_dims=[n])

    params = [
        hls.HLSVar("in_streams", in_stream_array_type),
        hls.HLSVar("out_key_streams", out_key_stream_array_type),
        hls.HLSVar("out_transform_streams", out_transform_stream_array_type),
    ]

    wrapper_var = hls.HLSVar("wrapper", wrapped_kt_pair_type)
    end_flag_var = hls.HLSVar(
        "end_flag_local",
        hls.HLSType(hls.HLSBasicType.BOOL),
    )

    # 内部循环体
    end_check_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, wrapper_var)],
    )

    # 表达式: wrapper.data.key
    key_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "key"),
        [
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "data"),
                [hls.HLSExpr(hls.HLSExprT.VAR, wrapper_var)],
            )
        ],
    )
    # 表达式: wrapper.data.transform
    transform_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "transform"),
        [
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "data"),
                [hls.HLSExpr(hls.HLSExprT.VAR, wrapper_var)],
            )
        ],
    )

    write_key = hls.CodeWriteStream(
        hls.HLSVar(f"out_key_streams[i]", out_key_stream_array_type.sub_types[0]), key_expr
    )
    write_transform = hls.CodeWriteStream(
        hls.HLSVar(f"out_transform_streams[i]", out_transform_stream_array_type.sub_types[0]),
        transform_expr,
    )

    unzip_if_block = hls.CodeIf(
        hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check_expr]),
        if_codes=[write_key, write_transform],
        else_codes=[hls.CodeAssign(end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))],
    )

    # For循环
    for_loop_body = [
        hls.CodePragma("UNROLL"),
        hls.CodeAssign(
            wrapper_var,
            hls.HLSExpr(
                hls.HLSExprT.STREAM_READ,
                None,
                [hls.HLSExpr(hls.HLSExprT.VAR, hls.HLSVar(f"in_streams[i]", in_stream_type))],
            ),
        ),
        unzip_if_block,
    ]
    for_loop = hls.CodeFor(for_loop_body, iter_limit=n, iter_name="i")

    # While循环
    while_body = [
        hls.CodePragma("PIPELINE"),
        hls.CodeAssign(end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
        hls.CodeVarDecl(wrapper_var.name, wrapper_var.type),
        for_loop,
        hls.CodeIf(end_flag_var, [hls.CodeBreak()]),
    ]

    body = [
        hls.CodeVarDecl(end_flag_var.name, end_flag_var.type, init_val="false"),
        hls.CodeWhile(iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True), codes=while_body),
    ]

    unzipper_func = hls.HLSFunction(name=func_name, comp=None)
    unzipper_func.params = params
    unzipper_func.codes = body
    return unzipper_func


# ======================================================================== #
#                      测试代码入口                                        #
# ======================================================================== #
if __name__ == "__main__":

    def print_hls_function(func: hls.HLSFunction):
        """辅助函数，用于打印单个HLSFunction对象的C++代码。"""
        # 修正函数签名的生成，以正确处理流数组
        params_list = []
        for p in func.params:
            if p.type.type == hls.HLSBasicType.ARRAY and p.type.sub_types[0].type == hls.HLSBasicType.STREAM:
                # C++ 数组参数语法: type name[]
                base_type_str = p.type.sub_types[0].name
                params_list.append(f"{base_type_str} {p.name}[]")
            else:
                # 普通参数语法: type& name
                params_list.append(f"{p.type.name}& {p.name}")

        params_str = ", ".join(params_list)

        print(f"// --- Function: {func.name} ---")
        print(f"void {func.name}({params_str}) {{")
        code_body = "".join([line.gen_code(indent_lvl=1) for line in func.codes])
        print(code_body, end="")
        print("}\n")

    print("=" * 50)
    print("      Running backend_utils.py Test Suite")
    print("=" * 50)

    # --- 1. 定义测试用的数据类型 ---
    N = 8
    DATA_TYPE_NAME = "update_tuple_dt"

    print(f"\n[INFO] Using N={N} and data_type='{DATA_TYPE_NAME}' for tests.\n")

    # a. 基础数据类型
    data_type = hls.HLSType(
        hls.HLSBasicType.STRUCT,
        sub_types=[hls.HLSType(hls.HLSBasicType.UINT), hls.HLSType(hls.HLSBasicType.BOOL)],
        struct_name=DATA_TYPE_NAME,
        struct_prop_names=["dst", "end_flag"],
    )
    # b. 批处理类型 (用于 Demux 测试)
    batch_data_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[data_type], array_dims=[N])
    batch_type = hls.HLSType(
        hls.HLSBasicType.STRUCT,
        sub_types=[
            batch_data_array_type,
            hls.HLSType(hls.HLSBasicType.BOOL),
            hls.HLSType(hls.HLSBasicType.UINT8),
        ],
        struct_prop_names=["data", "end_flag", "end_pos"],
    )

    # --- 2. 测试 generate_merge_stream_2x1 ---
    print("=" * 20, "Test: generate_merge_stream_2x1", "=" * 20)
    merge_func = generate_merge_stream_2x1(data_type)
    print_hls_function(merge_func)

    # --- 3. 测试 generate_reduction_tree ---
    print("=" * 20, "Test: generate_reduction_tree", "=" * 22)
    tree_func = generate_reduction_tree(n=N, data_type=data_type, merge_func=merge_func)
    print_hls_function(tree_func)

    # --- 4. 测试 generate_demux ---
    print("=" * 20, "Test: generate_demux", "=" * 28)
    demux_func = generate_demux(n=N, batch_type=batch_type)
    print_hls_function(demux_func)

    # --- 5. 测试 generate_omega_network ---
    print("=" * 20, "Test: generate_omega_network", "=" * 22)
    omega_functions = generate_omega_network(n=N, data_type_name=DATA_TYPE_NAME)
    for func in omega_functions:
        print_hls_function(func)

    print("=" * 50)
    print("            Test Suite Finished")
    print("=" * 50)

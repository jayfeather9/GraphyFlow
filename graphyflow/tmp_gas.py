from graphyflow.structures import *

global_cnt = 0


def new_random_name():
    global global_cnt
    global_cnt += 1
    return f"tmp_{global_cnt}"


def translate_graph(g: GlobalGraph):
    nodes = g.topo_sort_nodes()
    print(nodes)
    inputs = []
    while isinstance(nodes[0], Inputer) or isinstance(nodes[0], GetLength):
        inputs.append(nodes[0])
        nodes.pop(0)
    if isinstance(nodes[1], GetLength):
        inputs.append(nodes[1])
        nodes.pop(1)
    assert (
        isinstance(nodes[0], Map_)
        and isinstance(nodes[1], ReduceBy)
        and isinstance(nodes[2], Map_)
        and isinstance(nodes[3], Updater)
    )
    scatter = nodes[0]
    gather = nodes[1]
    apply = nodes[2]
    update = nodes[3]
    # 一个node的node.lambdas形如下：
    """ lambdas = [ { 'edges': [ (18, 20),
               (19, 20),
               (15, 18),
               (17, 18),
               (13, 15),
               (14, 15),
               (11, 17),
               (16, 17),
               (12, 19)],
    'input_ids': [11, 12],
    'nodes': { 11: {'name': 'arg0', 'type': 'input'},
               12: {'name': 'arg1', 'type': 'input'},
               13: { 'pseudo_element': <graphyflow.structures.PseudoElement object at 0x7f4ffde29590>,
                     'type': 'pseudo'},
               14: {'type': 'constant', 'value': 0.15},
               15: {'operator': '/', 'type': 'operation'},
               16: {'type': 'constant', 'value': 0.85},
               17: {'operator': '*', 'type': 'operation'},
               18: {'operator': '+', 'type': 'operation'},
               19: {'attr': 'out_degree', 'type': 'attr'},
               20: {'operator': '/', 'type': 'operation'}},
    'output_ids': [12, 20]}"""
    # 首先检查 scatter，先确定它必须是只有一个输入，两个输出，第二个输出必须是dst
    scatter_lambda = scatter.lambdas[0]
    assert len(scatter_lambda["input_ids"]) == 1
    assert len(scatter_lambda["output_ids"]) == 2
    assert scatter_lambda["nodes"][scatter_lambda["output_ids"][1]]["attr"] == "dst"
    # 然后检查 gather，gather是一个reduceby，两个lambdas分别是reduce的key和reduce方式，
    # key必须是俩输入一个输出，且输出必须是第二个，因为就是dst，方式则只要满足两个输入两个输出即可
    gather_lambda_key = gather.lambdas[0]
    assert len(gather_lambda_key["output_ids"]) == 1
    assert gather_lambda_key["output_ids"][0] == gather_lambda_key["input_ids"][1]
    gather_lambda_reduce = gather.lambdas[1]
    assert len(gather_lambda_reduce["input_ids"]) == 2
    assert len(gather_lambda_reduce["output_ids"]) == 2
    # 然后检查 apply，apply是一个map，两个输入两个输出即可
    apply_lambda = apply.lambdas[0]
    assert len(apply_lambda["output_ids"]) == 2
    assert len(apply_lambda["input_ids"]) == 2

    def find_ref_line(end_node_id, edges):
        visited = set()
        results = []
        to_visit = [end_node_id]
        while to_visit:
            node = to_visit.pop()
            if node in visited:
                continue
            visited.add(node)
            results.append(node)
            to_visit.extend([src for src, dst in edges if dst == node])
        results.reverse()
        return results

    # parse scatter, 先从第一个输出倒序遍历找到所有依赖的节点并保证该序列是拓扑序，然后根据拓扑序生成C++代码
    scatter_code = ""
    scatter_calc_nodes = find_ref_line(
        scatter_lambda["output_ids"][1], scatter_lambda["edges"]
    )
    """we generate something like:
    inline prop_t scatterFunc(prop_t srcProp, prop_t edgeProp)
    {
        return (srcProp);
    } but scatter_code first can only contain the lines in the function body"""
    # scatter code generation
    for node_id in scatter_calc_nodes:
        node_var = scatter_lambda["nodes"][node_id]
        if node_var["type"] == "input":
            pass
        elif node_var["type"] == "attr":
            node_var["var_name"] = new_random_name()
            prev_node = [src for src, dst in scatter_lambda["edges"] if dst == node_id]
            assert len(prev_node) == 1
            from_name = (
                f"srcProp"
                if prev_node[0] in scatter_lambda["input_ids"]
                else f"edgeProp.{node_var['attr']}"
            )
            scatter_code += f"\tprop_t {node_var['var_name']} = {from_name};\n"
        if node_id == scatter_calc_nodes[-1]:
            scatter_code += f"\treturn {node_var['var_name']};\n"
    scatter_code = (
        "inline prop_t scatterFunc(prop_t srcProp, prop_t edgeProp) {\n"
        + scatter_code
        + "}\n"
    )
    print(scatter_code)

    # gather code generation
    gather_code = ""
    gather_calc_nodes = find_ref_line(
        gather_lambda_reduce["output_ids"][0], gather_lambda_reduce["edges"]
    )
    for node_id in gather_calc_nodes:
        node_var = gather_lambda_reduce["nodes"][node_id]
        if node_var["type"] == "input":
            pass
        elif node_var["type"] == "idx":
            node_var["var_name"] = new_random_name()
            assert node_var["attr"] == 0
            prev_node = [
                src for src, dst in gather_lambda_reduce["edges"] if dst == node_id
            ]
            assert len(prev_node) == 1
            from_name = (
                "ori"
                if prev_node[0] == gather_lambda_reduce["input_ids"][0]
                else "update"
            )
            gather_code += f"\tprop_t {node_var['var_name']} = {from_name};\n"
        elif node_var["type"] == "operation":
            node_var["var_name"] = new_random_name()
            prev_node = [
                src for src, dst in gather_lambda_reduce["edges"] if dst == node_id
            ]
            assert len(prev_node) == 2
            op_node_1 = gather_lambda_reduce["nodes"][prev_node[0]]
            op_node_2 = gather_lambda_reduce["nodes"][prev_node[1]]
            gather_code += f"\tprop_t {node_var['var_name']} = {op_node_1['var_name']} {node_var['operator']} {op_node_2['var_name']};\n"
        if node_id == gather_calc_nodes[-1]:
            gather_code += f"\treturn {node_var['var_name']};\n"
    gather_code = (
        "inline prop_t gatherFunc(prop_t ori, prop_t update) {\n" + gather_code + "}\n"
    )
    print(gather_code)

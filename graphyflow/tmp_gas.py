import yaml
import os
from graphyflow.structures import *

target_path = "./output"
yaml_path = "./thundergp_conpatible_config.yaml"
global_cnt = 0


def new_random_name():
    global global_cnt
    global_cnt += 1
    return f"tmp_{global_cnt}"


def read_yaml_config(yaml_file):
    """Read and parse the YAML configuration file."""
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def create_apply_kernel_mk(config, output_file):
    """Create apply_kernel.mk (empty file)."""
    with open(output_file, "w") as file:
        pass  # Write an empty file


def create_build_mk(config, output_file):
    """Create build.mk from the YAML configuration."""
    build_config = config.get("build", {})

    with open(output_file, "w") as file:
        # Scatter-gather kernel section
        sg_kernel = build_config.get("scatter_gather_kernel", {})
        file.write(f"#scatter-gather kernel\n")
        file.write(
            f"HAVE_EDGE_PROP={str(sg_kernel.get('have_edge_prop', False)).lower()}\n"
        )
        file.write(
            f"HAVE_UNSIGNED_PROP={str(sg_kernel.get('have_unsigned_prop', False)).lower()}\n\n"
        )

        # Apply kernel section
        apply_kernel = build_config.get("apply_kernel", {})
        file.write(f"#apply kernel\n")
        file.write(f"HAVE_APPLY={str(apply_kernel.get('have_apply', True)).lower()}\n")
        file.write(
            f"CUSTOMIZE_APPLY={str(apply_kernel.get('customize_apply', False)).lower()}\n"
        )
        file.write(
            f"HAVE_APPLY_OUTDEG={str(apply_kernel.get('have_apply_outdeg', True)).lower()}\n\n"
        )

        # Scheduler section
        file.write(f"#scheduler\n")
        file.write(
            f"SCHEDULER={build_config.get('scheduler', 'secondOrderEstimator')}\n\n"
        )

        # Entry section
        entry_config = build_config.get("entry", {})
        file.write(f"#entry\n")
        file.write(
            f"DEFAULT_ENTRY={str(entry_config.get('default_entry', True)).lower()}\n"
        )


def create_config_mk(config, output_file):
    """Create config.mk from the YAML configuration."""
    config_data = config.get("config", {})

    with open(output_file, "w") as file:
        file.write(f"FREQ={config_data.get('freq', 280)}\n\n")

        queue_size = config_data.get("queue_size", {})
        file.write(f"QUEUE_SIZE_FILTER={queue_size.get('filter', 16)}\n")
        file.write(f"QUEUE_SIZE_MEMORY={queue_size.get('memory', 512)}\n\n")

        file.write(
            f"LOG_SCATTER_CACHE_BURST_SIZE={config_data.get('log_scatter_cache_burst_size', 6)}\n\n"
        )

        file.write(
            f"APPLY_REF_ARRAY_SIZE={config_data.get('apply_ref_array_size', 1)}\n"
        )


def generate_mk_files_from_yaml(yaml_config_file="config.yaml", output_dir="."):
    """Generate .mk files from a YAML configuration file."""
    # Check if YAML config file exists
    if not os.path.exists(yaml_config_file):
        print(f"Error: {yaml_config_file} not found.")
        return

    # Read YAML configuration
    config = read_yaml_config(yaml_config_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the .mk files
    create_apply_kernel_mk(config, os.path.join(output_dir, "apply_kernel.mk"))
    create_build_mk(config, os.path.join(output_dir, "build.mk"))
    create_config_mk(config, os.path.join(output_dir, "config.mk"))

    print(f"Generated .mk files successfully in {output_dir} from YAML configuration.")


def translate_graph(g: GlobalGraph):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    nodes = g.topo_sort_nodes()
    use_out_deg = False
    # print(nodes)
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
        """find the nodes in the reverse topological order"""
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
    # print(scatter_code)

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
    # print(gather_code)

    apply_code = ""
    apply_calc_nodes = find_ref_line(
        apply_lambda["output_ids"][0], apply_lambda["edges"]
    )
    for node_id in apply_calc_nodes:
        node_var = apply_lambda["nodes"][node_id]
        if node_var["type"] == "input":
            if node_id == apply_lambda["input_ids"][0]:
                node_var["var_name"] = "tProp"
        elif node_var["type"] == "constant":
            node_var["var_name"] = str(node_var["value"])
        elif node_var["type"] == "attr":
            assert node_var["attr"] == "out_degree"
            node_var["var_name"] = "outDeg"
            use_out_deg = True
        elif node_var["type"] == "pseudo":
            node_var["var_name"] = "arg"
        elif node_var["type"] == "operation":
            node_var["var_name"] = new_random_name()
            prev_node = [src for src, dst in apply_lambda["edges"] if dst == node_id]
            assert len(prev_node) == 2
            op_node_1 = apply_lambda["nodes"][prev_node[0]]
            op_node_2 = apply_lambda["nodes"][prev_node[1]]
            if node_var["operator"] == "/":
                apply_code += f"\tprop_t {node_var['var_name']} = {op_node_2['var_name']} == 0 ? 0 : {op_node_1['var_name']} {node_var['operator']} {op_node_2['var_name']};\n"
            else:
                apply_code += f"\tprop_t {node_var['var_name']} = {op_node_1['var_name']} {node_var['operator']} {op_node_2['var_name']};\n"
        else:
            assert False
        if node_id == apply_calc_nodes[-1]:
            apply_code += f"\treturn {node_var['var_name']};\n"
    apply_code = (
        "inline prop_t applyFunc( prop_t tProp, prop_t source, prop_t outDeg, unsigned int (&extra)[APPLY_REF_ARRAY_SIZE], unsigned int arg ) {\n"
        + apply_code
        + "}\n"
    )
    # print(apply_code)

    l2_code = (
        "#ifndef __L2_H__\n#define __L2_H__\n\ninline prop_t preprocessProperty(prop_t srcProp) {	return (srcProp); }\n\n"
        + scatter_code
        + gather_code
        + apply_code
        + "\n\n#endif"
    )

    data_prep_arg_code = "\n"
    if use_out_deg:
        data_prep_arg_code += "unsigned int dataPrepareGetArg(graphInfo *info) {\n\treturn info->vertexNum;\n}\n\n"

    data_prep_prop_code = ""
    for prop, prop_info in g.node_properties.items():
        if prop_info == "out_degree":
            pass
    for prop, prop_info in g.edge_properties.items():
        prop_type, prop_init = prop_info
        prop_init = prop_init.replace("node_num", "info->vertexNum")
        prop_init = prop_init.replace("edge.src.out_degree", "outDeg[i]")
        if prop_type == float:
            prop_init = f"float2int({prop_init})"
        data_prep_prop_code += "\tfor (int i = 0; i < vertexNum; i++) {\n"
        data_prep_prop_code += (
            "\t\tif outDeg[i] != 0 {\n"
            + f"\t\t\tvertexPushinProp[i] = {prop_init};\n"
            + "\t\t}\n"
            + "\t}\n"
        )
    data_prep_prop_code = (
        "int dataPrepareProperty(graphInfo *info) {\n\tint *outDeg = (int*)get_host_mem_pointer(MEM_ID_OUT_DEG_ORIGIN);\n\tprop_t *vertexPushinProp = (prop_t*)get_host_mem_pointer(MEM_ID_PUSHIN_PROP);\n"
        + "\tint alignedVertexNum = get_he_mem(MEM_ID_PUSHIN_PROP)->size/sizeof(int);\n"
        + data_prep_prop_code
        + "\tfor (int i = vertexNum; i < alignedVertexNum; i++) {\n\t\tvertexPushinProp[i]  = 0;\n\t}\n"
        + "\n\treturn 0;\n}"
    )
    # print(l2_code)
    # print(data_prep_arg_code)
    # print(data_prep_prop_code)
    data_prep_code = (
        """#include "host_graph_api.h"
#include "fpga_application.h"

#define INT2FLOAT                   (pow(2,30))

int float2int(float a) {
    return (int)(a * INT2FLOAT);
}

float int2float(int a) {
    return ((float)a / INT2FLOAT);
}
"""
        + data_prep_arg_code
        + data_prep_prop_code
    )

    # print(data_prep_code)

    with open(os.path.join(target_path, "l2.h"), "w") as f:
        f.write(l2_code)
    with open(os.path.join(target_path, "dataPrepare.cpp"), "w") as f:
        f.write(data_prep_code)
    generate_mk_files_from_yaml(yaml_path, target_path)

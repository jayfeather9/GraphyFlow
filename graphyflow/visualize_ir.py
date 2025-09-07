import re
from graphviz import Digraph


def parse_components(text):
    # print(text)
    components = []
    port_to_component = {}

    # 首先提取组件列表部分
    components_match = re.search(r"components:\s*\[(.*?)\],\s*inputs:", text, re.DOTALL)
    if not components_match:
        return [], {}, [], []

    components_text = components_match.group(1)

    # 基于括号匹配来拆分组件
    def split_components_by_brackets(text):
        components = []
        current = ""
        bracket_level = 0
        in_component = False
        comp_st = 0

        for i, char in enumerate(text):
            if not in_component and "Component(" in text[max(0, i - 15) : i + 1]:
                in_component = True
                comp_st = i - 1
                while comp_st >= 0 and (text[comp_st] == "_" or text[comp_st].isalpha()):
                    comp_st -= 1
                comp_st += 1

            if in_component:
                if char == "(":
                    bracket_level += 1
                elif char == ")":
                    bracket_level -= 1
                    if bracket_level == 0:
                        components.append(text[comp_st : i + 1])
                        in_component = False

        return components

    component_texts = split_components_by_brackets(components_text)
    # print(("="*100 + "\n").join(component_texts))
    component_matches = [
        (re.match(r"(\w+)Component", comp).group(1), comp)
        for comp in component_texts
        if re.match(r"(\w+)Component", comp)
    ]

    for idx, (comp_type, comp_content) in enumerate(component_matches):
        # print(comp_type, comp_content)
        comp_id = f"comp_{idx}"

        component = {
            "id": comp_id,
            "type": comp_type + "Component",
            "label": comp_type,
            "ports": {},
            "connections": [],
        }

        # Extract value for ConstantComponent
        if comp_type == "Constant":
            value_match = re.search(r"value: (\d+)", comp_content)
            if value_match:
                component["label"] += f"\nValue: {value_match.group(1)}"

        # Extract BinOp type
        if comp_type == "BinOp" or comp_type == "UnaryOp":
            op_match = re.search(r"op: (\w+)\.(\w+)", comp_content)
            if op_match:
                component["label"] += f"\nOp: {op_match.group(2)}"

        # Parse ports
        ports = re.findall(
            r"Port\[(\d+)\] (\w+) \((.*?)\)(?: => | <= )?\[?(\d+)?\]? (\w+)?",
            comp_content,
        )
        for port in ports:
            port_num, port_name, port_type, target_port, target_name = port
            if port_num in port_to_component:
                continue
            # print(port_num, port_name)
            component["ports"][port_num] = {
                "name": port_name,
                "type": port_type,
                "direction": "in" if port_name.startswith("i_") else "out",
            }
            port_to_component[port_num] = comp_id

            if target_port and port_name.startswith("o_"):
                component["connections"].append((port_num, target_port))

        unconnected_ports = re.findall(r"Port\[(\d+)\] (\w+) \((.*?)\)", comp_content)
        for port in unconnected_ports:
            port_num, port_name, port_type = port
            if port_num in port_to_component:
                continue
            print(port_num, port_name)
            component["ports"][port_num] = {
                "name": port_name,
                "type": port_type,
                "direction": "in" if port_name.startswith("i_") else "out",
            }
            port_to_component[port_num] = comp_id
        components.append(component)

    # 解析 inputs 和 outputs
    inputs = []
    outputs = []

    inputs_match = re.search(r"inputs:\s*\[(.*)\]", text)
    if inputs_match:
        inputs_text = inputs_match.group(1)
        inputs = re.findall(r"Port\[(\d+)\] (\w+) \((.*?)\)", inputs_text)

    outputs_match = re.search(r"outputs:\s*\[(.*)\]", text)
    if outputs_match:
        outputs_text = outputs_match.group(1)
        outputs = re.findall(r"Port\[(\d+)\] (\w+) \((.*?)\)", outputs_text)

    return components, port_to_component, inputs, outputs


def visualize_components(text):
    components, port_to_component, inputs, outputs = parse_components(text)
    # print(inputs, outputs)
    dot = Digraph(comment="Component Visualization")

    # 添加输入节点
    if inputs:
        dot.node(
            "input_node",
            "Input",
            shape="ellipse",
            style="filled",
            fillcolor="lightblue",
        )

    # 添加输出节点
    if outputs:
        dot.node(
            "output_node",
            "Output",
            shape="ellipse",
            style="filled",
            fillcolor="lightgreen",
        )

    # Add nodes
    for comp in components:
        shape = "ellipse"
        if "Constant" in comp["type"]:
            shape = "box3d"
        elif "BinOp" in comp["type"]:
            shape = "diamond"
        elif "Scatter" in comp["type"]:
            shape = "trapezium"
        dot.node(comp["id"], comp["label"], shape=shape)

    # Add edges
    connections = set()
    for comp in components:
        for src_port, tgt_port in comp["connections"]:
            if src_port in port_to_component and tgt_port in port_to_component:
                src_comp = port_to_component[src_port]
                tgt_comp = port_to_component[tgt_port]
                if (src_comp, tgt_comp) not in connections:
                    label = f'{comp["ports"][src_port]["name"]}: {comp["ports"][src_port]["type"]}'
                    dot.edge(src_comp, tgt_comp, label)
                    connections.add((src_comp, tgt_comp))

    for port_num, port_name, port_type in inputs:
        # print(port_num, port_name, port_type)
        if port_num in port_to_component:
            comp_id = port_to_component[port_num]
            dot.edge(
                "input_node",
                comp_id,
                label=f"{port_type}",
                color="blue",
                penwidth="2.0",
            )

    for port_num, port_name, port_type in outputs:
        # print(port_num, port_name, port_type)
        for comp in components:
            for p_num, p_info in comp["ports"].items():
                if p_num == port_num and p_info["direction"] == "out":
                    dot.edge(
                        comp["id"],
                        "output_node",
                        label=f"{port_type}",
                        color="green",
                        penwidth="2.0",
                    )

    return dot


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--component_def", type=str)
    parser.add_argument("--file_path", type=str)
    args = parser.parse_args()
    if args.component_def:
        dot = visualize_components(args.component_def)
        dot.render("component_graph", view=False, format="png")
    elif args.file_path:
        with open(args.file_path, "r") as f:
            text = f.read()
        dot = visualize_components(text)
        dot.render("component_graph", view=False, format="png")
    else:
        print("Please provide either a component definition or a file path.")

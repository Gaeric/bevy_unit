import bpy
import json
import os


default_nodes_cache = {}


def serialize_value(val):
    """
    Converts Blender-specific types (Vectors, Eulers, etc.) into
    JSON-compatible lists.
    """
    if isinstance(val, str):
        return val

    # Check for Vector or Euler types by looking for x, y attributes
    if hasattr(val, "x") and hasattr(val, "y"):
        return list(val)
    # Convert any other iterable (like colors or arrays) to a list
    return list(val) if hasattr(val, "__iter__") else val


def mat_tree(temp_group):
    all_materials_data = []
    # Iterate through all materials in the current Blender file
    for mat in bpy.data.materials:
        # Skip materials that don't use nodes (e.g., legacy viewport colors)
        if not mat.node_tree:
            continue

        mat_data = {"mat": mat.name, "nodes": {}, "links": []}

        for node in mat.node_tree.nodes:
            node_type = node.bl_idname
            if node_type not in default_nodes_cache:
                default_nodes_cache[node_type] = temp_group.nodes.new(
                    node_type)

            default_node = default_nodes_cache[node_type]
            n_data = {
                # The internal Blender class name (e.g., ShaderNodeMixRGB)
                "type": node.bl_idname,
                "props": {
                    k: serialize_value(getattr(node, k))
                    for k in ['operation', 'blend_type', 'interpolation']
                    if hasattr(node, k)
                },
                "in": {}
            }

            for p_name in ['opration', 'blend_type', 'interpolation']:
                if hasattr(node, p_name):
                    val = getattr(node, p_name)
                    default_val = getattr(default_node, p_name)
                    if val != default_val:
                        n_data['props'][p_name] = serialize_value

            for i in node.inputs:
                if i.is_linked:
                    n_data['in'][i.name] = 'LINKED'
                elif hasattr(i, 'default_value'):
                    current_val = serialize_value(i.default_value)
                    factory_val = serialize_value(
                        default_node.inputs[i.name].default_value)

                    if current_val != factory_val:
                        n_data['in'][i.name] = current_val

                mat_data["nodes"][node.name] = n_data

        # --- Process Connections (Links) ---
        for link in mat.node_tree.links:
            # Create a human-readable string representing the data flow
            mat_data["links"].append(
                f"{link.from_node.name}[{link.from_socket.name}] -> "
                f"{link.to_node.name}[{link.to_socket.name}]"
            )

        all_materials_data.append(mat_data)
    return all_materials_data


def export_minified_nodes():
    """
    Serializes the active material's node tree into a structured JSON format.
    Designed to help LLMs (like GPT-4 or Claude) accurately convert
    Blender shader logic into WGSL or other shading languages.
    """

    temp_group = bpy.data.node_groups.new("TempDefault", 'ShaderNodeTree')
    all_materials_data = mat_tree(temp_group)

    # Save the result to a temporary directory
    file_path = os.path.join("/tmp", "minified_nodes.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_materials_data, f, indent=4, ensure_ascii=False)

    print(f"Export complete: {file_path}")

    for group in bpy.data.node_groups:
        if group.name == "TempDefault":
            bpy.data.node_groups.remove(group)


if __name__ == "__main__":
    export_minified_nodes()

import bpy
import json
import os


def serialize_value(val):
    """
    Converts Blender-specific types (Vectors, Eulers, etc.) into
    JSON-compatible lists.
    """
    # Check for Vector or Euler types by looking for x, y attributes
    if hasattr(val, "x") and hasattr(val, "y"):
        return list(val)
    # Convert any other iterable (like colors or arrays) to a list
    return list(val) if hasattr(val, "__iter__") else val


def export_minified_nodes():
    """
    Serializes the active material's node tree into a structured JSON format.
    Designed to help LLMs (like GPT-4 or Claude) accurately convert
    Blender shader logic into WGSL or other shading languages.
    """
    all_materials_data = []

    # Iterate through all materials in the current Blender file
    for mat in bpy.data.materials:
        # Skip materials that don't use nodes (e.g., legacy viewport colors)
        if not mat.node_tree:
            continue

        mat_data = {"mat": mat.name, "nodes": {}, "links": []}

        # --- Process individual Nodes ---
        for node in mat.node_tree.nodes:
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

            # Capture input socket values
            for i in node.inputs:
                # Only include data if the socket is connected to another node
                # or has a non-default (non-zero) value to keep the JSON small
                if i.is_linked or (hasattr(i, 'default_value') and
                                   serialize_value(i.default_value) not in
                                   [0, 0.0, [0, 0, 0], [0, 0, 0, 1]]):
                    n_data["in"][i.name] = serialize_value(
                        getattr(i, 'default_value', None))

            mat_data["nodes"][node.name] = n_data

        # --- Process Connections (Links) ---
        for link in mat.node_tree.links:
            # Create a human-readable string representing the data flow
            mat_data["links"].append(
                f"{link.from_node.name}[{link.from_socket.name}] -> "
                f"{link.to_node.name}[{link.to_socket.name}]"
            )

        all_materials_data.append(mat_data)

    # Save the result to a temporary directory
    file_path = os.path.join("/tmp", "minified_nodes.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_materials_data, f, indent=4, ensure_ascii=False)

    print(f"Export complete: {file_path}")


if __name__ == "__main__":
    export_minified_nodes()

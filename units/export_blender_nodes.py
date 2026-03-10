#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: export_blender_nodes.py
# Datetime: Tue Mar 10 22:54:45 2026
# Software: Emacs
# Author:   Gaeric

import bpy
import json
import os


def serialize_value(val):
    if hasattr(val, "x") and hasattr(val, "y"):
        return list(val)  # Vector/Euler
    return list(val) if hasattr(val, "__iter__") else val


def export_minified_nodes():
    all_materials_data = []
    for mat in bpy.data.materials:
        if not mat.node_tree:
            continue

        mat_data = {"mat": mat.name, "nodes": {}, "links": []}

        for node in mat.node_tree.nodes:
            n_data = {
                "type": node.bl_idname,
                "props": {k: serialize_value(getattr(node, k))
                          for k in ['operation', 'blend_type', 'interpolation']
                          if hasattr(node, k)},
                "in": {}
            }
            # 只記錄有連線或有非零默認值的輸入
            for i in node.inputs:
                if i.is_linked or (hasattr(i, 'default_value') and
                                   serialize_value(i.default_value) not in
                                   [0, 0.0, [0, 0, 0], [0, 0, 0, 1]]):
                    n_data["in"][i.name] = serialize_value(
                        getattr(i, 'default_value', None))

            mat_data["nodes"][node.name] = n_data

        for link in mat.node_tree.links:
            mat_data["links"].append(
                f"{link.from_node.name}[{link.from_socket.name}] -> \
{link.to_node.name}[{link.to_socket.name}]")

        all_materials_data.append(mat_data)

    file_path = os.path.join("/tmp", "minified_nodes.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_materials_data, f, indent=4, ensure_ascii=False)
    print(f"complete: {file_path}")


if __name__ == "__main__":
    export_minified_nodes()

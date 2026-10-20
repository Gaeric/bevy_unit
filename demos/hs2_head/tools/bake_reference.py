"""Render per-channel ground truth for one HS2 material, headless.

The bake compute shaders evaluate the material in *texture space* (one texel
per output pixel, UV is the pixel coordinate). To compare them with Blender we
render a 1x1 plane whose UV covers 0..1 with an orthographic camera, so every
render pixel corresponds to the same UV as the bake pixel.

Channels are rendered through an Emission shader (no lighting, no BSDF), with
the `Raw` view transform, so the PNG bytes are the raw values the node graph
produces. This matches `save_img()` in `bake/mod.rs`, which also writes raw
texels.

Usage:

    blender -b prefab_materials_meshexporter.blend \
        --python demos/hs2_head/tools/bake_reference.py -- \
        --material Head --texture-dir assets/materials/head \
        --prefix cf_m_skin_head_01 --out demos/hs2_head/output/reference/head \
        --size 1024

Note: Blender's UV origin is bottom-left, while the bake writes row 0 at the
top of the image. Comparison code has to flip one of them.
"""

import argparse
import os
import sys

import bpy

# Image texture nodes of the prefab and the file each one receives from
# `importer.py::load_textures`. `srgb=False` mirrors `csp='Non-Color'`.
TEXTURES = {
    "MainTex": ("MainTex.png", True, "NONE"),
    "DetailMainTex": ("DetailMainTex.png", False, None),
    "DetailGlossMap": ("DetailGlossMap.png", False, None),
    "Eyebrow": ("Texture3.png", False, None),
    "BumpMap": ("BumpMap_converted.png", False, None),
    "BumpMap2": ("BumpMap2_converted.png", False, None),
    "Subsurface": ("SubsurfaceAlbedo.png", False, None),
}

# Channel name -> (group node, socket) inside the "Skin shader" node group.
CHANNELS = {
    "base_color": ("Mix.001", "Result"),
    "roughness": ("Math.001", "Value"),
}


def assign_textures(material, texture_dir, prefix):
    for node in material.node_tree.nodes:
        if node.bl_idname != "ShaderNodeTexImage":
            continue
        entry = TEXTURES.get(node.name)
        if entry is None:
            continue
        suffix, is_srgb, alpha_mode = entry
        path = os.path.join(texture_dir, f"{prefix}_{suffix}")
        if not os.path.exists(path):
            print(f"WARNING: missing texture for node {node.name}: {path}")
            continue
        image = bpy.data.images.load(path, check_existing=True)
        image.colorspace_settings.name = "sRGB" if is_srgb else "Non-Color"
        if alpha_mode:
            image.alpha_mode = alpha_mode
        node.image = image
        print(f"texture {node.name} <- {os.path.basename(path)}")


def skin_group(material):
    """Resolve material -> Head texture shader -> Skin shader."""
    outer = material.node_tree.nodes["Shader"].node_tree
    for node in outer.nodes:
        if node.bl_idname == "ShaderNodeGroup" and node.node_tree.name == "Skin shader":
            return node.node_tree
    raise RuntimeError("Skin shader group not found")


def render_channel(scene, tree, channel, out_path, name):
    output = tree.nodes["Group Output"]
    emission = tree.nodes.get("__bake_emission")
    if emission is None:
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "__bake_emission"
        emission.inputs["Strength"].default_value = 1.0

    # Material Output.Surface <- Emission
    for link in list(output.inputs["BSDF"].links):
        tree.links.remove(link)
    tree.links.new(emission.outputs["Emission"], output.inputs["BSDF"])

    if channel == "normal":
        source = tree.nodes["Group"]  # Bump/displacement -> Normal
        encode = tree.nodes.get("__bake_encode")
        if encode is None:
            encode = tree.nodes.new("ShaderNodeVectorMath")
            encode.name = "__bake_encode"
            encode.operation = "MULTIPLY_ADD"
            encode.inputs[1].default_value = (0.5, 0.5, 0.5)
            encode.inputs[2].default_value = (0.5, 0.5, 0.5)
        tree.links.new(source.outputs["Normal"], encode.inputs[0])
        tree.links.new(encode.outputs["Vector"], emission.inputs["Color"])
    else:
        node_name, socket = CHANNELS[channel]
        tree.links.new(
            tree.nodes[node_name].outputs[socket], emission.inputs["Color"])

    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"{name}: {out_path}")


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="Head")
    parser.add_argument("--texture-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args(argv)

    material = bpy.data.materials.get(args.material)
    if material is None:
        raise SystemExit(f"material {args.material} not found")
    assign_textures(material, args.texture_dir, args.prefix)

    # Scene: one plane, UV 0..1, orthographic camera covering it exactly.
    for obj in list(bpy.data.objects):
        if obj.type in {"MESH", "CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.data.materials.clear()
    plane.data.materials.append(material)

    camera_data = bpy.data.cameras.new("BakeCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new("BakeCamera", camera_data)
    camera.location = (0.0, 0.0, 1.0)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    world = bpy.data.worlds.new(
        "BakeWorld") if not bpy.data.worlds else bpy.data.worlds[0]
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        background.inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    # Raw: store the node output values without a display transform.
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    os.makedirs(args.out, exist_ok=True)
    tree = skin_group(material)

    # Restore the original BSDF link afterwards so repeated runs stay stable.
    output = tree.nodes["Group Output"]
    original = output.inputs["BSDF"].links[0].from_socket

    for channel in ("base_color", "roughness", "normal"):
        render_channel(scene, tree, channel, os.path.join(
            args.out, f"{channel}.png"), channel)

    for link in list(output.inputs["BSDF"].links):
        tree.links.remove(link)
    tree.links.new(original, output.inputs["BSDF"])


main()

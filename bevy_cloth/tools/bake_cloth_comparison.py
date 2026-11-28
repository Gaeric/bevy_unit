"""Bake the native Blender Cloth reference for the Bevy visual comparison.

blender --background --factory-startup --python-exit-code 1 \
    --python bevy_cloth/tools/bake_cloth_comparison.py -- /tmp/cloth-visual.txt
"""

import argparse
import math
from pathlib import Path
import sys

import bpy

FPS = 60
FRAME_COUNT = 241  # Blender frame 1 is Bevy tick 0.
LENGTH_SEGMENTS = 20
WIDTH_SEGMENTS = 8
VERTEX_MASS = 0.001  # kg per vertex, not total strip mass
GRAVITY = 9.81
# Blender 4.5 Cloth advances one internal time unit per frame at Speed 1
# and multiplies scene gravity by 0.001 (SIM_mass_spring.cc). Match the
# displacement of g in SI units over 1/FPS seconds, without changing gravity.
# Blender sums float substeps until the frame ends. A quantized speed and
# power-of-two qualities keep that sum exact and avoid an extra substep.
CLOTH_SPEED = round(65536 / (FPS * math.sqrt(0.001))) / 65536
QUALITIES = (32, 64)


def strip_mesh():
    stride = WIDTH_SEGMENTS + 1
    points = [
        (i / LENGTH_SEGMENTS, 0.4 * (j / WIDTH_SEGMENTS - 0.5), 0.0)
        for i in range(LENGTH_SEGMENTS + 1)
        for j in range(stride)
    ]
    triangles = []
    for i in range(LENGTH_SEGMENTS):
        for j in range(WIDTH_SEGMENTS):
            a = i * stride + j
            b = a + stride
            c, d = b + 1, a + 1
            # Mirror the diagonals across the centerline to keep the mesh symmetric.
            if j < WIDTH_SEGMENTS // 2:
                triangles.extend(((a, b, c), (a, c, d)))
            else:
                triangles.extend(((a, b, d), (b, c, d)))
    return points, triangles, list(range(stride))


def bake(quality, points, triangles, pins):
    scene = bpy.context.scene
    scene.frame_set(1)
    mesh = bpy.data.meshes.new("Comparison strip")
    mesh.from_pydata(points, [], triangles)
    mesh.update()
    obj = bpy.data.objects.new("Comparison strip", mesh)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    cloth = obj.modifiers.new("Native Cloth", "CLOTH")
    settings = cloth.settings
    settings.quality = quality
    settings.time_scale = CLOTH_SPEED
    settings.mass = VERTEX_MASS
    if pins:
        group = obj.vertex_groups.new(name="Fixed short edge")
        group.add(pins, 1.0, "REPLACE")
        settings.vertex_group_mass = group.name
    settings.bending_model = "ANGULAR"
    settings.tension_stiffness = 20.0
    settings.compression_stiffness = 20.0
    settings.shear_stiffness = 15.0
    settings.bending_stiffness = 0.02
    settings.air_damping = 0.0
    settings.tension_damping = 0.0
    settings.compression_damping = 0.0
    settings.shear_damping = 0.0
    settings.bending_damping = 0.0
    cloth.collision_settings.use_collision = False
    cloth.collision_settings.use_self_collision = False

    cache = cloth.point_cache
    cache.frame_start = 1
    cache.frame_end = FRAME_COUNT
    with bpy.context.temp_override(object=obj, active_object=obj, point_cache=cache):
        bpy.ops.ptcache.bake(bake=True)
    if not cache.is_baked:
        raise RuntimeError(f"Blender did not bake its Cloth cache at Quality {quality}")

    frames = []
    for frame in range(1, FRAME_COUNT + 1):
        scene.frame_set(frame)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        positions = [tuple(vertex.co) for vertex in evaluated.data.vertices]
        if len(positions) != len(points) or not all(
            math.isfinite(value) for position in positions for value in position
        ):
            raise RuntimeError(f"Invalid Blender mesh at Quality {quality}, frame {frame}")
        frames.append(positions)

    if max(math.dist(a, b) for a, b in zip(frames[0], points)) > 1e-6:
        raise RuntimeError("Blender frame 1 did not preserve the rest mesh")
    if any(math.dist(frame[pin], points[pin]) > 1e-6 for frame in frames for pin in pins):
        raise RuntimeError("A Blender pin moved")
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    return frames


def describe(quality, frames):
    width = WIDTH_SEGMENTS + 1
    average_z = [sum(p[2] for p in frame[-width:]) / width for frame in frames]
    first_low = next(
        (tick for tick in range(1, len(average_z) - 1)
         if average_z[tick] < average_z[tick - 1] and average_z[tick] < average_z[tick + 1]),
        None,
    )
    if first_low is not None:
        print(f"Quality {quality}: first free-edge low {average_z[first_low]:.4f} m "
              f"at tick {first_low} ({first_low / FPS:.2f} s)")
    else:
        print(f"Quality {quality}: no free-edge low before the end of the clip")


def save(path, frames, triangles, pins, quality):
    # One whitespace-delimited header, followed by pins, triangles and frame-major
    # XYZ positions. Frame zero also supplies the rest mesh used in all panels.
    with path.open("w", encoding="ascii") as output:
        print("cloth-visual-v1", bpy.app.version_string.replace(" ", "-"), quality,
              FPS, len(frames), len(frames[0]), len(triangles), len(pins),
              VERTEX_MASS, GRAVITY, file=output)
        print(*pins, file=output)
        for triangle in triangles:
            print(*triangle, file=output)
        for frame in frames:
            for x, y, z in frame:
                # Blender is Z-up. This rotation into Y-up preserves triangle winding.
                print(f"{x:.9g} {z:.9g} {-y:.9g}", file=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT
    scene.gravity = (0.0, 0.0, -GRAVITY)
    points, triangles, pins = strip_mesh()
    low_quality, high_quality = QUALITIES
    low = bake(low_quality, points, triangles, pins)
    high = bake(high_quality, points, triangles, pins)
    describe(low_quality, low)
    describe(high_quality, high)
    difference, tick, vertex = max(
        (math.dist(a, b), tick, vertex)
        for tick, (low_frame, high_frame) in enumerate(zip(low, high))
        for vertex, (a, b) in enumerate(zip(low_frame, high_frame))
    )
    print(f"Max Quality difference: {difference:.4f} m at tick {tick}, vertex {vertex}")
    save(args.output, high, triangles, pins, high_quality)
    print(f"Wrote Blender {bpy.app.version_string} reference: {args.output}")


if __name__ == "__main__":
    main()

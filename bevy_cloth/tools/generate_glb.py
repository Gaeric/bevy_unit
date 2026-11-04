#!/usr/bin/env python3
"""Generate cloth.glb and sphere.glb; Python standard library only.

Run: python3 bevy_cloth/tools/generate_glb.py
Geometry/materials only, without simulation or attachment metadata.
"""

import json
import math
from pathlib import Path
import struct

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "velvet"


def write_glb(name, positions, normals, indices, translation, scale, uvs=None):
    binary, views, accessors = bytearray(), [], []

    def view(data, target=None):
        binary.extend(b"\0" * (-len(binary) % 4))
        item = {"buffer": 0, "byteOffset": len(
            binary), "byteLength": len(data)}
        if target is not None:
            item["target"] = target
        binary.extend(data)
        views.append(item)
        return len(views) - 1

    def attribute(values, kind, bounds=False):
        flat = [x for value in values for x in value]
        item = {"bufferView": view(struct.pack(f"<{len(flat)}f", *flat), 34962),
                "componentType": 5126, "count": len(values), "type": kind}
        if bounds:
            item["min"] = [min(v[i] for v in values) for i in range(3)]
            item["max"] = [max(v[i] for v in values) for i in range(3)]
        accessors.append(item)
        return len(accessors) - 1

    attributes = {"POSITION": attribute(positions, "VEC3", True),
                  "NORMAL": attribute(normals, "VEC3")}
    if uvs is not None:
        attributes["TEXCOORD_0"] = attribute(uvs, "VEC2")
    accessors.append({"bufferView": view(struct.pack(f"<{len(indices)}I", *indices), 34963),
                      "componentType": 5125, "count": len(indices), "type": "SCALAR"})
    material = {"name": name + "Material", "doubleSided": uvs is not None,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.0, 0.21404114, 1.0, 1.0]
                    if uvs is not None else [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    doc = {
        "asset": {"version": "2.0", "generator": "bevy_cloth/tools/generate_glb.py"},
        "scene": 0, "scenes": [{"name": name + "Scene", "nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0, "translation": list(translation),
                   "scale": list(scale)}],
        "meshes": [{"name": name, "primitives": [{"attributes": attributes,
                    "indices": len(accessors) - 1, "material": 0, "mode": 4}]}],
        "materials": [material], "accessors": accessors, "bufferViews": views,
    }
    if uvs is not None:
        image = view((ASSETS / "fabric2.jpg").read_bytes())
        doc["images"] = [{"bufferView": image, "mimeType": "image/jpeg"}]
        doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987,
                            "wrapS": 10497, "wrapT": 10497}]
        doc["textures"] = [{"sampler": 0, "source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    doc["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    output = (struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
              + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
              + struct.pack("<II", len(binary), 0x004E4942) + binary)
    path = ASSETS / (name.lower() + ".glb")
    path.write_bytes(output)
    print(f"{path}: {len(positions)} vertices, {len(indices) // 3} triangles, "
          f"{len(output)} bytes")


def cloth(resolution=16):
    positions, normals, uvs, indices = [], [], [], []
    side = resolution + 1
    for row in range(side):
        for column in range(side):
            u, v = column / resolution, row / resolution
            positions.append((2 * (u - 0.5), -2 * v, 0.0))
            normals.append((0.0, 0.0, 1.0))
            uvs.append((u, v))
    for row in range(resolution):
        for column in range(resolution):
            a = row * side + column
            b, c, d = a + side, a + 1, a + side + 1
            indices.extend((a, b, c, c, b, d))
    return positions, normals, indices, uvs


def normalize(v):
    length = math.sqrt(sum(x * x for x in v))
    return tuple(x / length for x in v)


def sphere(subdivisions=3):
    """Shared-vertex icosphere: outward winding, no UV seam or degenerate poles."""
    t = (1 + math.sqrt(5)) / 2
    positions = [normalize(v) for v in [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        midpoints = {}

        def midpoint(a, b):
            key = tuple(sorted((a, b)))
            if key not in midpoints:
                midpoints[key] = len(positions)
                positions.append(normalize(tuple(positions[a][i] + positions[b][i]
                                                 for i in range(3))))
            return midpoints[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab),
                           (c, ca, bc), (ab, bc, ca)))
        faces = refined
    return positions, [i for face in faces for i in face]


def main():
    positions, normals, indices, uvs = cloth()
    write_glb("Cloth", positions, normals, indices,
              (0.0, 2.5, 0.0), (1.0, 1.0, 1.0), uvs)
    positions, indices = sphere()
    write_glb("Sphere", positions, positions, indices,
              (0.0, 0.6, -1.0), (0.6, 0.6, 0.6))


if __name__ == "__main__":
    main()

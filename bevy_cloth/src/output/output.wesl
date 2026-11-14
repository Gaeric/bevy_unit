// Convert world-space particles into local-space mesh positions and normals.

import super::super::solver::layout::Particle;

// Offsets and stride are measured in four-byte words, matching OutputVertexParams.
struct OutputVertexParams {
    vertex_start: u32,
    stride_words: u32,
    position_offset_words: u32,
    normal_offset_words: u32,
};

@group(0) @binding(0)
var<storage, read> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read> normal_offsets: array<u32>;

@group(0) @binding(2)
var<storage, read> normal_pairs: array<vec2<u32>>;

@group(0) @binding(3)
var<uniform> world_to_local: mat4x4<f32>;

@group(0) @binding(4)
var<uniform> vertex_params: OutputVertexParams;

// Bind the entire allocator vertex buffer at byte offset zero.
@group(0) @binding(5)
var<storage, read_write> output_vertices: array<f32>;

@compute @workgroup_size(64)
fn output_mesh(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= arrayLength(&particles) {
        return;
    }

    let world_position = particles[index].position;

    // Gather oriented, area-weighted face normals in world space.
    var world_normal_sum = vec3<f32>(0.0);
    let start = normal_offsets[index];
    let end = normal_offsets[index + 1u];

    for (var entry = start; entry < end; entry += 1u) {
        let pair = normal_pairs[entry];
        let edge_a = particles[pair.x].position - world_position;
        let edge_b = particles[pair.y].position - world_position;

        world_normal_sum += cross(edge_a, edge_b);
    }

    // cross(A * e1, A * e2) = cofactor(A) * cross(e1, e2).
    // Transform the sum once, preserving nonuniform scale and reflection.
    let a = world_to_local[0].xyz;
    let b = world_to_local[1].xyz;
    let c = world_to_local[2].xyz;
    let area_transform = mat3x3<f32>(
        cross(b, c),
        cross(c, a),
        cross(a, b),
    );
    let local_normal_sum = area_transform * world_normal_sum;

    // Isolated vertices and degenerate faces retain a zero normal.
    var local_normal = vec3<f32>(0.0);
    let length_squared = dot(local_normal_sum, local_normal_sum);
    if length_squared > 0.0 {
        local_normal = local_normal_sum * inverseSqrt(length_squared);
    }

    let vertex_base =
        (vertex_params.vertex_start + index) * vertex_params.stride_words;
    let position_base = vertex_base + vertex_params.position_offset_words;
    let normal_base = vertex_base + vertex_params.normal_offset_words;

    let local_position =
        (world_to_local * vec4<f32>(world_position, 1.0)).xyz;

    output_vertices[position_base] = local_position.x;
    output_vertices[position_base + 1u] = local_position.y;
    output_vertices[position_base + 2u] = local_position.z;

    output_vertices[normal_base] = local_normal.x;
    output_vertices[normal_base + 1u] = local_normal.y;
    output_vertices[normal_base + 2u] = local_normal.z;
}

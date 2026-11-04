// Ranges are in vertex units; stride and attribute offsets are in 32-bit words.
// Host validates the layout; 0xffffffff disables the corresponding attribute.
struct OutputRange {
    start: u32,
    end: u32,
    particle_offset: u32,
    count: u32,
    stride_words: u32,
    position_offset_words: u32,
    normal_offset_words: u32,
    pad1: u32,
    uv_offset_words: u32,
    tangent_offset_words: u32,
    pad2: vec2<u32>,
    world_to_local: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> range: OutputRange;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vertices: array<f32>;
// CSR rows use cloth-local vertex IDs, ordered by source triangle encounter.
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> normal_pairs: array<vec2<u32>>;

// Only called after offsets and widths have been checked against the stride.
fn disjoint(a: u32, a_width: u32, b: u32, b_width: u32) -> bool {
    if a <= b { return b - a >= a_width; }
    return a - b >= b_width;
}

fn vertex_uv(vertex: u32) -> vec2<f32> {
    let word = (range.start + vertex) * range.stride_words + range.uv_offset_words;
    return vec2<f32>(vertices[word], vertices[word + 1u]);
}

@compute @workgroup_size(256)
fn output_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= range.count || range.start >= range.end { return; }
    if i >= range.end - range.start { return; }
    let n = arrayLength(&positions);
    if range.particle_offset >= n || i >= n - range.particle_offset { return; }
    if range.stride_words < 3u { return; }
    if range.position_offset_words > range.stride_words - 3u { return; }
    // Divide first so malformed offsets cannot wrap during multiplication.
    let vertex_count = arrayLength(&vertices) / range.stride_words;
    if range.start >= vertex_count || i >= vertex_count - range.start { return; }
    let p = (range.world_to_local * vec4<f32>(positions[range.particle_offset + i].xyz, 1.0)).xyz;
    let has_normals = range.normal_offset_words != 0xffffffffu;
    let has_tangents = range.tangent_offset_words != 0xffffffffu;
    if has_tangents {
        if !has_normals || range.uv_offset_words == 0xffffffffu { return; }
        if range.stride_words < 4u { return; }
        if range.uv_offset_words > range.stride_words - 2u { return; }
        if range.tangent_offset_words > range.stride_words - 4u { return; }
        if !disjoint(range.uv_offset_words, 2u, range.position_offset_words, 3u) { return; }
        if !disjoint(range.uv_offset_words, 2u, range.normal_offset_words, 3u) { return; }
        if !disjoint(range.tangent_offset_words, 4u, range.position_offset_words, 3u) { return; }
        if !disjoint(range.tangent_offset_words, 4u, range.normal_offset_words, 3u) { return; }
        if !disjoint(range.tangent_offset_words, 4u, range.uv_offset_words, 2u) { return; }
    }
    var normal = vec3<f32>(0.0);
    var tangent_sum = vec3<f32>(0.0);
    var bitangent_sum = vec3<f32>(0.0);
    var tangent = vec4<f32>(0.0);
    if has_normals {
        if range.normal_offset_words > range.stride_words - 3u { return; }
        if !disjoint(range.position_offset_words, 3u, range.normal_offset_words, 3u) { return; }
        let row_count = arrayLength(&offsets);
        // Subtract first: even malformed i cannot wrap the i + 1 row lookup.
        if row_count < 2u { return; }
        if i >= row_count - 1u { return; }
        let begin = offsets[i];
        let end = offsets[i + 1u];
        if begin > end || end > arrayLength(&normal_pairs) { return; }
        let remaining_positions = n - range.particle_offset;
        for (var pair_index = begin; pair_index < end; pair_index += 1u) {
            let pair = normal_pairs[pair_index];
            if pair.x >= range.count || pair.y >= range.count { return; }
            if pair.x >= remaining_positions || pair.y >= remaining_positions { return; }
            let other0 = (range.world_to_local * vec4<f32>(positions[range.particle_offset + pair.x].xyz, 1.0)).xyz;
            let other1 = (range.world_to_local * vec4<f32>(positions[range.particle_offset + pair.y].xyz, 1.0)).xyz;
            let edge0 = other0 - p;
            let edge1 = other1 - p;
            // Cyclic per-vertex orientation preserves the raw triangle area vector.
            let area = cross(edge0, edge1);
            normal += area;
            if has_tangents {
                if pair.x >= range.end - range.start || pair.y >= range.end - range.start { return; }
                if pair.x >= vertex_count - range.start || pair.y >= vertex_count - range.start { return; }
                // Only P/N/T are written by this dispatch. UV words remain immutable,
                // so gathering neighboring UVs needs neither atomics nor a barrier.
                let uv = vertex_uv(i);
                let duv0 = vertex_uv(pair.x) - uv;
                let duv1 = vertex_uv(pair.y) - uv;
                let determinant = duv0.x * duv1.y - duv0.y * duv1.x;
                if determinant != 0.0 && dot(area, area) > 0.0 {
                    // Sum raw triangle UV derivatives (no additional area weighting).
                    tangent_sum += (edge0 * duv1.y - edge1 * duv0.y) / determinant;
                    bitangent_sum += (edge1 * duv0.x - edge0 * duv1.x) / determinant;
                }
            }
        }
        // Match CPU compute_normals: no epsilon, rest-normal or non-finite fallback.
        let normal_length = length(normal);
        if normal_length > 0.0 {
            normal /= normal_length;
        }
        if has_tangents && normal_length > 0.0 {
            let orthogonal = tangent_sum - normal * dot(normal, tangent_sum);
            let tangent_length = length(orthogonal);
            if tangent_length > 0.0 {
                let direction = orthogonal / tangent_length;
                let handedness = dot(cross(normal, direction), bitangent_sum);
                // A degenerate UV frame stays zero; do not invent a rest tangent.
                if handedness != 0.0 {
                    tangent = vec4<f32>(direction, sign(handedness));
                }
            }
        }
    }
    // Divide-first capacity and attribute guards keep writes inside this vertex.
    let word = (range.start + i) * range.stride_words + range.position_offset_words;
    vertices[word] = p.x;
    vertices[word + 1u] = p.y;
    vertices[word + 2u] = p.z;
    if has_normals {
        let normal_word = (range.start + i) * range.stride_words + range.normal_offset_words;
        vertices[normal_word] = normal.x;
        vertices[normal_word + 1u] = normal.y;
        vertices[normal_word + 2u] = normal.z;
    }
    if has_tangents {
        let tangent_word = (range.start + i) * range.stride_words + range.tangent_offset_words;
        vertices[tangent_word] = tangent.x;
        vertices[tangent_word + 1u] = tangent.y;
        vertices[tangent_word + 2u] = tangent.z;
        vertices[tangent_word + 3u] = tangent.w;
    }
}

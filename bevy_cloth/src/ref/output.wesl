// Ranges are in vertex units; stride and attribute offsets are in f32 words.
// Host validates the layout; normal_offset_words == 0xffffffff disables normals.
struct OutputRange {
    start: u32,
    end: u32,
    particle_offset: u32,
    count: u32,
    stride_words: u32,
    position_offset_words: u32,
    normal_offset_words: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> range: OutputRange;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vertices: array<f32>;
// CSR rows use cloth-local vertex IDs, ordered by source triangle encounter.
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> normal_pairs: array<vec2<u32>>;

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
    let p = positions[range.particle_offset + i].xyz;
    let has_normals = range.normal_offset_words != 0xffffffffu;
    var normal = vec3<f32>(0.0);
    if has_normals {
        if range.normal_offset_words > range.stride_words - 3u { return; }
        // Both three-word attributes must fit this vertex without overlapping.
        if range.position_offset_words <= range.normal_offset_words {
            if range.normal_offset_words - range.position_offset_words < 3u { return; }
        } else {
            if range.position_offset_words - range.normal_offset_words < 3u { return; }
        }
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
            let other0 = positions[range.particle_offset + pair.x].xyz;
            let other1 = positions[range.particle_offset + pair.y].xyz;
            // Cyclic per-vertex orientation preserves the raw triangle area vector.
            normal += cross(other0 - p, other1 - p);
        }
        // Match CPU compute_normals: no epsilon, rest-normal or non-finite fallback.
        let normal_length = length(normal);
        if normal_length > 0.0 {
            normal /= normal_length;
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
}

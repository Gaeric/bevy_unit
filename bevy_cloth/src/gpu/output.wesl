// POSITION-only mesh contract: 12-byte stride, ranges in vertex units.
struct OutputRange {
    start: u32,
    end: u32,
    particle_offset: u32,
    count: u32,
};

@group(0) @binding(0) var<uniform> range: OutputRange;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vertices: array<f32>;

@compute @workgroup_size(256)
fn output_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= range.count || range.start >= range.end { return; }
    if i >= range.end - range.start { return; }
    let n = arrayLength(&positions);
    if range.particle_offset >= n || i >= n - range.particle_offset { return; }
    // Divide first so malformed offsets cannot wrap during multiplication.
    let vertex_count = arrayLength(&vertices) / 3u;
    if range.start >= vertex_count || i >= vertex_count - range.start { return; }
    let word = (range.start + i) * 3u;
    let p = positions[range.particle_offset + i].xyz;
    vertices[word] = p.x;
    vertices[word + 1u] = p.y;
    vertices[word + 2u] = p.z;
}

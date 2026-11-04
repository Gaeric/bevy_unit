// Owned 32-byte BVH ABI; only bounds change, never topology/reference ordering.
struct Node {
    min: vec3<f32>, first_index: u32,
    max: vec3<f32>, primitive_count: u32,
}
struct RefitParams { first: u32, count: u32, node_count: u32, triangle_count: u32 }
@group(0) @binding(0) var<uniform> params: RefitParams;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> refs: array<u32>;
@group(0) @binding(5) var<storage, read> work_ids: array<u32>;

fn finite(p: vec3<f32>) -> bool {
    // Bit test catches infinities and all NaNs without fast-math comparisons.
    return all((bitcast<vec3<u32>>(p) & vec3<u32>(0x7f800000u)) != vec3<u32>(0x7f800000u));
}
fn valid_bounds(n: Node) -> bool {
    return finite(n.min) && finite(n.max) && all(n.min <= n.max);
}
// CPU adapter validates unique node IDs and exact-depth, deepest-first scheduling.
// On invalid geometry return without replacing the old bound; no synthetic AABB.
@compute @workgroup_size(256)
fn refit_leaves(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= params.count { return; }
    if params.first > arrayLength(&work_ids) || invocation.x >= arrayLength(&work_ids) - params.first { return; }
    let id = work_ids[params.first + invocation.x];
    if id >= params.node_count || id >= arrayLength(&nodes) { return; }
    let node = nodes[id];
    if node.primitive_count == 0u || node.first_index > arrayLength(&refs) { return; }
    if node.primitive_count > arrayLength(&refs) - node.first_index { return; }
    var lower = vec3<f32>(0.0);
    var upper = vec3<f32>(0.0);
    var initialized = false;
    for (var slot = 0u; slot < node.primitive_count; slot++) {
        let triangle = refs[node.first_index + slot];
        if triangle >= params.triangle_count || triangle >= arrayLength(&indices) / 3u { return; }
        for (var corner = 0u; corner < 3u; corner++) {
            let vertex = indices[triangle * 3u + corner];
            if vertex >= arrayLength(&positions) { return; }
            let point = positions[vertex].xyz;
            if !finite(point) { return; }
            if !initialized {
                lower = point; upper = point; initialized = true;
            } else {
                lower = min(lower, point); upper = max(upper, point);
            }
        }
    }
    nodes[id].min = lower;
    nodes[id].max = upper;
}

// Each exact-depth layer is a separate compute pass: both children are completed
// by earlier passes, and no invocation writes another invocation's inputs.
@compute @workgroup_size(256)
fn refit_internal(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= params.count { return; }
    if params.first > arrayLength(&work_ids) || invocation.x >= arrayLength(&work_ids) - params.first { return; }
    let id = work_ids[params.first + invocation.x];
    if id >= params.node_count || id >= arrayLength(&nodes) { return; }
    let node = nodes[id];
    let available = min(params.node_count, arrayLength(&nodes));
    if node.primitive_count != 0u || available < 2u { return; }
    if node.first_index >= available - 1u { return; }
    let left = nodes[node.first_index];
    let right = nodes[node.first_index + 1u];
    if !valid_bounds(left) || !valid_bounds(right) { return; }
    nodes[id].min = min(left.min, right.min);
    nodes[id].max = max(left.max, right.max);
}

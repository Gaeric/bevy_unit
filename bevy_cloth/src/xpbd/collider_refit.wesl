// Matches the 32-byte node ABI in bvh.rs. Only min/max are modified.
struct BvhNode {
    min: vec3<f32>,
    first_index: u32,
    max: vec3<f32>,
    primitive_count: u32,
};

struct RefitParams {
    first: u32,
    count: u32,
    padding: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> nodes: array<BvhNode>;
@group(0) @binding(3) var<storage, read> triangle_refs: array<u32>;
// Leaves first, followed by internal nodes in deepest-to-root layer order.
@group(0) @binding(4) var<storage, read> refit_nodes: array<u32>;
@group(0) @binding(5) var<uniform> params: RefitParams;

@compute @workgroup_size(64)
fn refit_leaves(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= params.count {
        return;
    }

    let node_id = refit_nodes[params.first + i];
    let node = nodes[node_id];
    let largest = bitcast<f32>(0x7f7fffffu);
    var lower = vec3<f32>(largest);
    var upper = vec3<f32>(-largest);

    for (var offset = 0u; offset < node.primitive_count; offset += 1u) {
        let triangle = triangle_refs[node.first_index + offset];
        let base = triangle * 3u;
        let a = positions[indices[base]].xyz;
        let b = positions[indices[base + 1u]].xyz;
        let c = positions[indices[base + 2u]].xyz;
        lower = min(lower, min(a, min(b, c)));
        upper = max(upper, max(a, max(b, c)));
    }

    nodes[node_id].min = lower;
    nodes[node_id].max = upper;
}

// Each dispatch processes one depth. Children belong to completed earlier stages.
@compute @workgroup_size(64)
fn refit_internal(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= params.count {
        return;
    }

    let node_id = refit_nodes[params.first + i];
    let left = nodes[node_id].first_index;
    let right = left + 1u;
    nodes[node_id].min = min(nodes[left].min, nodes[right].min);
    nodes[node_id].max = max(nodes[left].max, nodes[right].max);
}

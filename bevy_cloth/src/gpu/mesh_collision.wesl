// Step 04: world-space closest-point queries only; no collision response.
// Exact host ABI (all offsets/strides in bytes):
// QueryParams: 16 bytes, four u32 at 0/4/8/12.
// Node: stride 32, min at 0, first at 12, max at 16, count at 28.
// count == 0: children first and first + 1; otherwise triangle_refs[first..first+count].
// Positions/probes: vec4<f32> stride 16, xyz used, w ignored.
// Indices/triangle_refs: u32 stride 4; refs retain ORIGINAL indices/3 triangle IDs.
// Hit: stride 64, point at 0, barycentric at 16, normal at 32,
// triangle_id/visited_nodes/status/pad at 48/52/56/60.
// point.w = squared distance; barycentric.w = normal.w = 0.
// status: 0 success, 1 invalid geometry/query (including arithmetic overflow),
// 2 traversal stack overflow. Failed hits have zero vectors and triangle_id = ~0u.
// Root is node 0. Host must validate the tree and reject max_depth >= 127.
// Each invocation owns exactly one result; no atomics or shared writes.
struct QueryParams {
    query_count: u32,
    node_count: u32,
    triangle_count: u32,
    pad: u32,
};

struct Node {
    min: vec3<f32>,
    first: u32,
    max: vec3<f32>,
    count: u32,
};

struct Hit {
    point: vec4<f32>,
    barycentric: vec4<f32>,
    normal: vec4<f32>,
    triangle_id: u32,
    visited_nodes: u32,
    status: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: QueryParams;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> triangle_refs: array<u32>;
@group(0) @binding(5) var<storage, read> probes: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> results: array<Hit>;

const INVALID_TRIANGLE: u32 = 0xffffffffu;
const STACK_CAPACITY: u32 = 128u;

// Exponent-bit checks do not depend on NaN comparison/fast-math behavior.
fn finite_scalar(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn finite_vector(value: vec3<f32>) -> bool {
    return all((bitcast<vec3<u32>>(value) & vec3<u32>(0x7f800000u)) != vec3<u32>(0x7f800000u));
}

fn max_component(value: vec3<f32>) -> f32 {
    return max(value.x, max(value.y, value.z));
}

fn failed_hit(status: u32, visited: u32) -> Hit {
    return Hit(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
        INVALID_TRIANGLE, visited, status, 0u);
}

fn valid_bounds(node: Node) -> bool {
    return finite_vector(node.min) && finite_vector(node.max) && all(node.min <= node.max);
}

fn bounds_distance_squared(p: vec3<f32>, node: Node) -> f32 {
    let delta = max(max(node.min - p, p - node.max), vec3<f32>(0.0));
    return dot(delta, delta);
}

// Inputs below are uniformly rescaled before dot products. Zero-length edges
// use endpoint 0, so coincident vertices never require division by zero.
fn segment_parameter(offset: vec3<f32>, edge: vec3<f32>) -> f32 {
    let scale = max_component(abs(edge));
    if scale == 0.0 { return 0.0; }
    let direction = edge / scale;
    // Clamp the numerator before dividing by the edge scale: a distant probe
    // may project beyond the segment without requiring an overflowing quotient.
    let numerator = dot(offset, direction);
    let denominator = dot(direction, direction) * scale;
    if numerator <= 0.0 { return 0.0; }
    if numerator >= denominator { return 1.0; }
    return numerator / denominator;
}

fn segment_fallback(ab: vec3<f32>, ac: vec3<f32>, ap: vec3<f32>) -> vec3<f32> {
    let tab = segment_parameter(ap, ab);
    var bary = vec3<f32>(1.0 - tab, tab, 0.0);
    let dab = ap - ab * tab;
    var best = dot(dab, dab);
    let tac = segment_parameter(ap, ac);
    let dac = ap - ac * tac;
    let distance_ac = dot(dac, dac);
    if distance_ac < best {
        best = distance_ac;
        bary = vec3<f32>(1.0 - tac, 0.0, tac);
    }
    let tbc = segment_parameter(ap - ab, ac - ab);
    let dbc = ap - (ab * (1.0 - tbc) + ac * tbc);
    if dot(dbc, dbc) < best {
        bary = vec3<f32>(0.0, 1.0 - tbc, tbc);
    }
    return bary;
}

// Project onto the winding plane, then use oriented areas in its dominant 2D
// plane. Only EDGES are normalized: query-distance scaling followed by quartic
// dot-product determinants can collapse a tiny face beneath a distant probe.
// Area numerators retain world length and are bounded before division, so a
// far outside projection never needs an overflowing query/edge-scale quotient.
fn closest_barycentric(ab: vec3<f32>, ac: vec3<f32>, ap: vec3<f32>, edge_scale: f32) -> vec3<f32> {
    var boundary_offset = ap;
    if edge_scale > 0.0 {
        let u = ab / edge_scale;
        let v = ac / edge_scale;
        let area = cross(u, v);
        let area_scale = max_component(abs(area));
        if area_scale > 0.0 {
            let direction = area / area_scale;
            let normal = direction / length(direction);
            let projected = ap - normal * dot(ap, normal);
            // Every boundary candidate shares the same perpendicular distance.
            // Remove it before comparing edges, or a far probe can round all
            // three total squared distances to the same value on a tiny face.
            boundary_offset = projected;
            var axis = 0u;
            if abs(area.y) > abs(area.x) { axis = 1u; }
            if abs(area.z) > abs(area[axis]) { axis = 2u; }
            let orientation = sign(area[axis]);
            let numerator_b = cross(projected, v)[axis] * orientation;
            let numerator_c = cross(u, projected)[axis] * orientation;
            let denominator = abs(area[axis]) * edge_scale;
            if finite_scalar(denominator) && denominator > 0.0
                && numerator_b >= 0.0 && numerator_c >= 0.0
                && numerator_c <= denominator && numerator_b <= denominator - numerator_c {
                let b = numerator_b / denominator;
                let c = numerator_c / denominator;
                // Clamp only the last-bit residual at the closed BC boundary.
                return vec3<f32>(max(0.0, 1.0 - b - c), b, c);
            }
        }
    }
    // An outside plane projection has its closest point on the boundary.
    // Evaluate all three CLOSED segments independently, also covering collinear
    // or coincident vertices without a world-space degeneracy epsilon. Scaling
    // these distance comparisons is independent of the face-area calculation.
    let query_scale = max(edge_scale, max_component(abs(boundary_offset)));
    if query_scale == 0.0 { return vec3<f32>(1.0, 0.0, 0.0); }
    return segment_fallback(ab / query_scale, ac / query_scale, boundary_offset / query_scale);
}

fn triangle_hit(p: vec3<f32>, triangle_id: u32) -> Hit {
    // Divide before multiplying, and subtract before adding, to avoid u32 wrap.
    if triangle_id >= params.triangle_count || triangle_id >= arrayLength(&indices) / 3u {
        return failed_hit(1u, 0u);
    }
    let base = triangle_id * 3u;
    let ia = indices[base];
    let ib = indices[base + 1u];
    let ic = indices[base + 2u];
    let vertex_count = arrayLength(&positions);
    if ia >= vertex_count || ib >= vertex_count || ic >= vertex_count {
        return failed_hit(1u, 0u);
    }
    let a = positions[ia].xyz;
    let b = positions[ib].xyz;
    let c = positions[ic].xyz;
    if !finite_vector(a) || !finite_vector(b) || !finite_vector(c) {
        return failed_hit(1u, 0u);
    }
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    if !finite_vector(ab) || !finite_vector(ac) || !finite_vector(ap) {
        return failed_hit(1u, 0u);
    }
    let edge_scale = max(max_component(abs(ab)), max_component(abs(ac)));
    let bary = closest_barycentric(ab, ac, ap, edge_scale);
    // Winding face normal, never the probe-to-surface direction. Rescale before
    // cross/normalize to avoid overflow or an unnecessary small-area epsilon.
    var normal = vec3<f32>(0.0);
    if edge_scale > 0.0 {
        let area = cross(ab / edge_scale, ac / edge_scale);
        let normal_scale = max_component(abs(area));
        if normal_scale > 0.0 {
            let direction = area / normal_scale;
            normal = direction / length(direction);
        }
    }
    let point = a * bary.x + b * bary.y + c * bary.z;
    let delta = p - point;
    let distance_squared = dot(delta, delta);
    if !finite_vector(bary) || !finite_vector(point) || !finite_vector(normal) || !finite_scalar(distance_squared) {
        return failed_hit(1u, 0u);
    }
    return Hit(vec4<f32>(point, distance_squared), vec4<f32>(bary, 0.0),
        vec4<f32>(normal, 0.0), triangle_id, 0u, 0u, 0u);
}

// Buffer-independent query interface for the later collision-response entry.
// Geometry still comes from the same read-only bindings; only the wrapper owns
// probe/result indexing. Preserve failure status and visited-node diagnostics.
fn closest_hit(p: vec3<f32>) -> Hit {
    var hit = failed_hit(1u, 0u);
    if !finite_vector(p) || params.node_count == 0u || params.triangle_count == 0u { return hit; }
    if params.node_count > arrayLength(&nodes) || params.triangle_count > arrayLength(&indices) / 3u {
        return hit;
    }
    if params.triangle_count > arrayLength(&triangle_refs) { return hit; }

    var stack: array<u32, 128>;
    stack[0] = 0u;
    var stack_size = 1u;
    var visited = 0u;
    var best_distance = bitcast<f32>(0x7f7fffffu);
    loop {
        if stack_size == 0u { break; }
        // A valid tree visits each node at most once. Bound malformed cycles
        // even when they keep only one pending child and do not fill the stack.
        if visited >= params.node_count {
            return failed_hit(1u, visited);
        }
        stack_size -= 1u;
        let node_id = stack[stack_size];
        if node_id >= params.node_count {
            return failed_hit(1u, visited);
        }
        visited += 1u;
        let node = nodes[node_id];
        if !valid_bounds(node) {
            return failed_hit(1u, visited);
        }
        let lower_bound = bounds_distance_squared(p, node);
        if !finite_scalar(lower_bound) {
            return failed_hit(1u, visited);
        }
        // No radius cutoff. Equal-distance candidates may choose either ID.
        if lower_bound > best_distance { continue; }
        if node.count == 0u {
            if node.first >= params.node_count || params.node_count - node.first < 2u {
                return failed_hit(1u, visited);
            }
            if stack_size > STACK_CAPACITY - 2u {
                return failed_hit(2u, visited);
            }
            // LIFO DFS: left first, right pending. Prune each child on pop.
            stack[stack_size] = node.first + 1u;
            stack[stack_size + 1u] = node.first;
            stack_size += 2u;
        } else {
            let reference_count = arrayLength(&triangle_refs);
            if node.first >= reference_count || node.count > reference_count - node.first {
                return failed_hit(1u, visited);
            }
            for (var offset = 0u; offset < node.count; offset += 1u) {
                let candidate = triangle_hit(p, triangle_refs[node.first + offset]);
                if candidate.status != 0u {
                    return failed_hit(1u, visited);
                }
                if hit.triangle_id == INVALID_TRIANGLE || candidate.point.w < best_distance {
                    hit = candidate;
                    best_distance = candidate.point.w;
                }
            }
        }
    }
    hit.visited_nodes = visited;
    return hit;
}

@compute @workgroup_size(256)
fn query_closest(@builtin(global_invocation_id) id: vec3<u32>) {
    let query_id = id.x;
    if query_id >= params.query_count || query_id >= arrayLength(&results) { return; }
    results[query_id] = failed_hit(1u, 0u);
    if query_id >= arrayLength(&probes) { return; }
    results[query_id] = closest_hit(probes[query_id].xyz);
}

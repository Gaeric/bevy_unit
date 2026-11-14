// Returns the closest-point parameter on a closed segment.
// offset = query point - segment start; edge = segment end - segment start.
fn segment_parameter(
    offset: vec3<f32>,
    edge: vec3<f32>,
) -> f32 {
    let length_squared = dot(edge, edge);

    // A zero-length segment is represented by its start point.
    if length_squared == 0.0 {
        return 0.0;
    }

    return clamp(
        dot(offset, edge) / length_squared,
        0.0,
        1.0,
    );
}

// Returns barycentric weights for the closest point on the triangle boundary.
// ab = B - A; ac = C - A; ap = query point - A.
fn closest_boundary_barycentric(
    ab: vec3<f32>,
    ac: vec3<f32>,
    ap: vec3<f32>,
) -> vec3<f32> {
    // Segment AB.
    let t_ab = segment_parameter(ap, ab);
    let delta_ab = ap - t_ab * ab;

    var best_distance = dot(delta_ab, delta_ab);
    var best_bary = vec3<f32>(1.0 - t_ab, t_ab, 0.0);

    // Segment AC.
    let t_ac = segment_parameter(ap, ac);
    let delta_ac = ap - t_ac * ac;
    let distance_ac = dot(delta_ac, delta_ac);

    if distance_ac < best_distance {
        best_distance = distance_ac;
        best_bary = vec3<f32>(1.0 - t_ac, 0.0, t_ac);
    }

    // Segment BC.
    let t_bc = segment_parameter(ap - ab, ac - ab);
    let delta_bc = ap - (ab + t_bc * (ac - ab));
    let distance_bc = dot(delta_bc, delta_bc);

    if distance_bc < best_distance {
        best_bary = vec3<f32>(0.0, 1.0 - t_bc, t_bc);
    }

    return best_bary;
}

// Returns closest-point weights for vertices A, B and C.
// ab = B - A; ac = C - A; ap = query point - A.
// Reconstruct the relative closest point as bary.y * ab + bary.z * ac.
fn closest_triangle_barycentric(
    ab: vec3<f32>,
    ac: vec3<f32>,
    ap: vec3<f32>,
) -> vec3<f32> {
    let area = cross(ab, ac);

    // A numerically zero area uses the closed-segment fallback.
    if dot(area, area) == 0.0 {
        return closest_boundary_barycentric(ab, ac, ap);
    }

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);

    // Vertex A.
    if d1 <= 0.0 && d2 <= 0.0 {
        return vec3<f32>(1.0, 0.0, 0.0);
    }

    let bp = ap - ab;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);

    // Vertex B.
    if d3 >= 0.0 && d4 <= d3 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }

    let vc = d1 * d4 - d3 * d2;

    // Edge AB.
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return vec3<f32>(1.0 - v, v, 0.0);
    }

    let cp = ap - ac;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);

    // Vertex C.
    if d6 >= 0.0 && d5 <= d6 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }

    let vb = d5 * d2 - d1 * d6;

    // Edge AC.
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return vec3<f32>(1.0 - w, 0.0, w);
    }

    let va = d3 * d6 - d5 * d4;

    // Edge BC.
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vec3<f32>(0.0, 1.0 - w, w);
    }

    // Face interior.
    let inverse_sum = 1.0 / (va + vb + vc);
    let v = vb * inverse_sum;
    let w = vc * inverse_sum;

    return vec3<f32>(1.0 - v - w, v, w);
}

// Matches the 32-byte node ABI in bvh.rs.
struct BvhNode {
    min: vec3<f32>,
    first_index: u32,
    max: vec3<f32>,
    primitive_count: u32,
};

// Internal shader result, not a CPU readback ABI.
// point.w stores squared distance; unused vector lanes are zero.
// A nonzero status invalidates the geometric fields.
struct QueryHit {
    point: vec4<f32>,
    barycentric: vec4<f32>,
    normal: vec4<f32>,
    triangle_id: u32,
    status: u32,
};

const INVALID_TRIANGLE: u32 = 0xffffffffu;
const QUERY_OK: u32 = 0u;
const QUERY_STACK_OVERFLOW: u32 = 1u;
const QUERY_NUMERIC_ERROR: u32 = 2u;
const QUERY_INVALID_TREE: u32 = 3u;
const STACK_CAPACITY: u32 = 64u;

@group(1) @binding(0) var<storage, read> world_positions: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> indices: array<u32>;
@group(1) @binding(2) var<storage, read> nodes: array<BvhNode>;
@group(1) @binding(3) var<storage, read> triangle_refs: array<u32>;

fn bounds_distance_squared(p: vec3<f32>, node: BvhNode) -> f32 {
    let delta = max(max(node.min - p, p - node.max), vec3<f32>(0.0));
    return dot(delta, delta);
}

fn finite_scalar(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn finite_vector(value: vec3<f32>) -> bool {
    let exponent = bitcast<vec3<u32>>(value) & vec3<u32>(0x7f800000u);
    return all(exponent != vec3<u32>(0x7f800000u));
}

fn empty_hit() -> QueryHit {
    return QueryHit(
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
        INVALID_TRIANGLE, QUERY_OK,
    );
}

// Geometry and tree indices are validated by CPU preparation.
// Returns an unsigned global nearest point, not an inside/outside decision.
fn closest_hit(p: vec3<f32>) -> QueryHit {
    var hit = empty_hit();
    if !finite_vector(p) {
        hit.status = QUERY_NUMERIC_ERROR;
        return hit;
    }
    if arrayLength(&nodes) == 0u {
        hit.status = QUERY_INVALID_TREE;
        return hit;
    }

    var best_distance = bitcast<f32>(0x7f7fffffu);
    var stack: array<u32, STACK_CAPACITY>;
    stack[0] = 0u;
    var stack_size = 1u;
    var visited_nodes = 0u;

    loop {
        if stack_size == 0u {
            break;
        }
        // A valid tree visits each node at most once. Also bounds accidental cycles.
        if visited_nodes >= arrayLength(&nodes) {
            hit.status = QUERY_INVALID_TREE;
            return hit;
        }

        stack_size -= 1u;
        let node_id = stack[stack_size];
        let node = nodes[node_id];
        visited_nodes += 1u;

        let lower_bound = bounds_distance_squared(p, node);
        if !finite_scalar(lower_bound) {
            hit.status = QUERY_NUMERIC_ERROR;
            return hit;
        }
        if lower_bound > best_distance {
            continue;
        }

        if node.primitive_count > 0u {
            for (var offset = 0u; offset < node.primitive_count; offset += 1u) {
                let triangle_id = triangle_refs[node.first_index + offset];
                let base = triangle_id * 3u;
                let a = world_positions[indices[base]].xyz;
                let b = world_positions[indices[base + 1u]].xyz;
                let c = world_positions[indices[base + 2u]].xyz;
                let ab = b - a;
                let ac = c - a;
                let ap = p - a;
                let bary = closest_triangle_barycentric(ab, ac, ap);
                let offset_from_a = bary.y * ab + bary.z * ac;
                let point = a + offset_from_a;
                let delta = ap - offset_from_a;
                let distance_squared = dot(delta, delta);

                if !finite_vector(bary) || !finite_vector(point)
                    || !finite_scalar(distance_squared)
                {
                    hit.status = QUERY_NUMERIC_ERROR;
                    return hit;
                }
                if hit.triangle_id == INVALID_TRIANGLE || distance_squared < best_distance {
                    best_distance = distance_squared;
                    hit.point = vec4<f32>(point, distance_squared);
                    hit.barycentric = vec4<f32>(bary, 0.0);
                    hit.triangle_id = triangle_id;
                }
            }
            continue;
        }

        let left = node.first_index;
        let right = left + 1u;
        let left_distance = bounds_distance_squared(p, nodes[left]);
        let right_distance = bounds_distance_squared(p, nodes[right]);
        if !finite_scalar(left_distance) || !finite_scalar(right_distance) {
            hit.status = QUERY_NUMERIC_ERROR;
            return hit;
        }

        var near_id = left;
        var far_id = right;
        var near_distance = left_distance;
        var far_distance = right_distance;
        if right_distance < left_distance {
            near_id = right;
            far_id = left;
            near_distance = right_distance;
            far_distance = left_distance;
        }

        // LIFO: push far first. Recheck its bound after the near subtree finishes.
        if far_distance <= best_distance {
            if stack_size == STACK_CAPACITY {
                hit.status = QUERY_STACK_OVERFLOW;
                return hit;
            }
            stack[stack_size] = far_id;
            stack_size += 1u;
        }
        if near_distance <= best_distance {
            if stack_size == STACK_CAPACITY {
                hit.status = QUERY_STACK_OVERFLOW;
                return hit;
            }
            stack[stack_size] = near_id;
            stack_size += 1u;
        }
    }

    // Normalize only the final winning face. Degenerate faces retain zero normal.
    if hit.triangle_id != INVALID_TRIANGLE {
        let base = hit.triangle_id * 3u;
        let a = world_positions[indices[base]].xyz;
        let b = world_positions[indices[base + 1u]].xyz;
        let c = world_positions[indices[base + 2u]].xyz;
        let area = cross(b - a, c - a);
        let area_squared = dot(area, area);
        if !finite_vector(area) || !finite_scalar(area_squared) {
            hit.status = QUERY_NUMERIC_ERROR;
            return hit;
        }
        if area_squared > 0.0 {
            hit.normal = vec4<f32>(area * inverseSqrt(area_squared), 0.0);
        }
    }
    return hit;
}

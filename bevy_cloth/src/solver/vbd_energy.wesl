// Elastic contributions to a single VBD vertex block. Forces are negative
// energy gradients. The local Hessians use Newton's membrane clamp and hinge
// Gauss–Newton approximation; neither changes the elastic force.
import super::constraint_math::{evaluate_bend, wrap_angle_difference};

struct VbdBlock {
    force: vec3<f32>,
    hessian: mat3x3<f32>,
};

const IDENTITY_3 = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
);

fn outer_self(v: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(v * v.x, v * v.y, v * v.z);
}

// grad1 and grad2 are the rows of the inverse 2D rest-edge matrix. They
// are also the rest-space gradients of the two non-origin shape functions.
fn evaluate_vbd_membrane(
    x: array<vec3<f32>, 3>, corner: u32,
    grad1: vec2<f32>, grad2: vec2<f32>, rest_area: f32,
    shear: f32, area_stiffness: f32,
    start: array<vec3<f32>, 3>, damping_over_dt: f32,
) -> VbdBlock {
    let edge1 = x[1] - x[0];
    let edge2 = x[2] - x[0];
    let f0 = edge1 * grad1.x + edge2 * grad2.x;
    let f1 = edge1 * grad1.y + edge2 * grad2.y;
    let shape_gradients = array(-grad1 - grad2, grad1, grad2);
    let g = shape_gradients[corner];

    let area_vector = cross(f0, f1);
    // Smooth the area norm at collapse so its first and second derivatives
    // use the same denominator. At rest the regularizer rounds away in f32.
    let area_ratio = sqrt(dot(area_vector, area_vector) + 1e-20);
    let normal = area_vector / area_ratio;
    let area_direction = g.x * f1 - g.y * f0;
    let area_gradient = cross(area_direction, normal);

    // Stable Neo-Hookean surface energy, scaled by rest area. Writing the
    // area stress as lambda * (J - 1) - shear avoids division by a material
    // parameter. At rest this term cancels the shear term in the total stress.
    let lambda = area_stiffness + shear;
    let area_stress = lambda * (area_ratio - 1.0) - shear;
    let gradient = shear * (g.x * f0 + g.y * f1) + area_stress * area_gradient;

    // J has a PSD Hessian in a single corner because its area vector is
    // affine in that corner. Clamp only its coefficient, not the force.
    let r = max(area_stress, 0.0) / area_ratio;
    let isotropic = shear * dot(g, g) + r * dot(area_direction, area_direction);
    var force = -gradient;
    var hessian = isotropic * IDENTITY_3
        + (lambda - r) * outer_self(area_gradient)
        - r * outer_self(area_direction);

    if damping_over_dt > 0.0 {
        // Rate of the metric C = F^T F, not of F itself: a rigid rotation
        // does not change C. The two off-diagonal entries contribute equally.
        let start_edge1 = start[1] - start[0];
        let start_edge2 = start[2] - start[0];
        let start_f0 = start_edge1 * grad1.x + start_edge2 * grad2.x;
        let start_f1 = start_edge1 * grad1.y + start_edge2 * grad2.y;
        let delta00 = dot(f0, f0) - dot(start_f0, start_f0);
        let delta01 = dot(f0, f1) - dot(start_f0, start_f1);
        let delta11 = dot(f1, f1) - dot(start_f1, start_f1);
        let grad00 = 2.0 * g.x * f0;
        let grad01 = g.x * f1 + g.y * f0;
        let grad11 = 2.0 * g.y * f1;
        force -= damping_over_dt * (delta00 * grad00 + 2.0 * delta01 * grad01 + delta11 * grad11);
        hessian += damping_over_dt * (outer_self(grad00) + 2.0 * outer_self(grad01) + outer_self(grad11));
    }
    return VbdBlock(rest_area * force, rest_area * hessian);
}

// Points are ordered [opposite0, opposite1, edge_start, edge_end], matching
// the signed-dihedral convention used by the XPBD geometry helper.
fn evaluate_vbd_hinge(
    x: array<vec3<f32>, 4>, corner: u32,
    rest_angle: f32, rest_edge_length: f32, stiffness: f32,
    start: array<vec3<f32>, 4>, damping_over_dt: f32,
) -> VbdBlock {
    let bend = evaluate_bend(x[0], x[1], x[2], x[3], rest_angle);
    var zero: VbdBlock;
    if !bend.valid {
        return zero;
    }

    let gradients = array(bend.g0, bend.g1, bend.g2, bend.g3);
    let gradient = gradients[corner];
    let k = stiffness * rest_edge_length;
    let curvature = outer_self(gradient);
    var block = VbdBlock(-k * bend.error * gradient, k * curvature);
    if damping_over_dt > 0.0 {
        let start_bend = evaluate_bend(start[0], start[1], start[2], start[3], 0.0);
        if start_bend.valid {
            let delta = wrap_angle_difference(bend.angle - start_bend.angle);
            let damping = damping_over_dt * rest_edge_length;
            block.force -= (damping * delta) * gradient;
            block.hessian += damping * curvature;
        }
    }
    return block;
}

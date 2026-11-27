// Elastic contributions to a single VBD vertex block. Forces are negative
// energy gradients. The local Hessians use Newton's membrane clamp and hinge
// Gauss–Newton approximation; neither changes the elastic force.
import super::constraint_math::evaluate_bend;

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
    let hessian = isotropic * IDENTITY_3
        + (lambda - r) * outer_self(area_gradient)
        - r * outer_self(area_direction);

    return VbdBlock(-rest_area * gradient, rest_area * hessian);
}

// Points are ordered [opposite0, opposite1, edge_start, edge_end], matching
// the signed-dihedral convention used by the XPBD geometry helper.
fn evaluate_vbd_hinge(
    x: array<vec3<f32>, 4>, corner: u32,
    rest_angle: f32, rest_edge_length: f32, stiffness: f32,
) -> VbdBlock {
    let bend = evaluate_bend(x[0], x[1], x[2], x[3], rest_angle);
    var zero: VbdBlock;
    if !bend.valid {
        return zero;
    }

    let gradients = array(bend.g0, bend.g1, bend.g2, bend.g3);
    let gradient = gradients[corner];
    let k = stiffness * rest_edge_length;
    return VbdBlock(-k * bend.error * gradient, k * outer_self(gradient));
}

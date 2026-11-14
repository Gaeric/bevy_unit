// Colored in-place XPBD dihedral solve; one accumulated lambda per hinge.

import super::layout::{Particle, StepParams, GroupParams};

// Matches the 32-byte Bend ABI in layout.rs.
struct Bend {
    particles: vec4<u32>,
    rest_angle: f32,
    compliance: f32,
    padding: vec2<u32>,
};

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(2)
var<uniform> group_params: GroupParams;

@group(0) @binding(3)
var<storage, read> bend: array<Bend>;

@group(0) @binding(4)
var<storage, read_write> bend_lambdas: array<f32>;

@compute @workgroup_size(64)
fn solve_bending(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let local_index = invocation.x;
    if local_index >= group_params.count {
        return;
    }

    let index = group_params.start + local_index;
    let constraint = bend[index];
    let ids = constraint.particles;
    let a = particles[ids.x];
    let b = particles[ids.y];
    let c = particles[ids.z];
    let d = particles[ids.w];
    if a.inv_mass + b.inv_mass + c.inv_mass + d.inv_mass == 0.0 {
        return;
    }

    // Same oriented faces as bending.rs: (a,c,d) and (b,d,c).
    let edge = d.position - c.position;
    let edge_squared = dot(edge, edge);
    if edge_squared == 0.0 {
        return;
    }
    let edge_length = sqrt(edge_squared);
    let ca = c.position - a.position;
    let da = d.position - a.position;
    let db = d.position - b.position;
    let cb = c.position - b.position;
    let area0 = cross(ca, da);
    let area1 = cross(db, cb);
    let area0_squared = dot(area0, area0);
    let area1_squared = dot(area1, area1);

    // Same dimensionless skinny-triangle threshold as CPU rest preparation.
    let scale0_squared = max(edge_squared, max(dot(ca, ca), dot(da, da)));
    let scale1_squared = max(edge_squared, max(dot(db, db), dot(cb, cb)));
    if area0_squared <= 1e-12 * scale0_squared * scale0_squared
        || area1_squared <= 1e-12 * scale1_squared * scale1_squared
    {
        return;
    }

    let normal0 = area0 * inverseSqrt(area0_squared);
    let normal1 = area1 * inverseSqrt(area1_squared);
    let sin_angle = dot(edge / edge_length, cross(normal0, normal1));
    let cos_angle = clamp(dot(normal0, normal1), -1.0, 1.0);
    let angle = atan2(sin_angle, cos_angle);

    // Rest angles are signed; choose the shortest difference across +/-pi.
    let pi = 3.141592653589793;
    let two_pi = 6.283185307179586;
    var error = angle - constraint.rest_angle;
    if error > pi {
        error -= two_pi;
    } else if error < -pi {
        error += two_pi;
    }

    // Gradients of our signed atan2 angle, not the unsigned acos angle.
    let scaled0 = area0 / area0_squared;
    let scaled1 = area1 / area1_squared;
    let g0 = -edge_length * scaled0;
    let g1 = -edge_length * scaled1;
    let g2 = -(dot(a.position - d.position, edge) * scaled0
        + dot(b.position - d.position, edge) * scaled1) / edge_length;
    let g3 = -(dot(c.position - a.position, edge) * scaled0
        + dot(c.position - b.position, edge) * scaled1) / edge_length;

    let alpha = constraint.compliance / (step_params.dt * step_params.dt);
    let denominator = a.inv_mass * dot(g0, g0)
        + b.inv_mass * dot(g1, g1)
        + c.inv_mass * dot(g2, g2)
        + d.inv_mass * dot(g3, g3) + alpha;
    if denominator == 0.0 {
        return;
    }
    let lambda = bend_lambdas[index];
    let delta_lambda = (-error - alpha * lambda) / denominator;
    bend_lambdas[index] = lambda + delta_lambda;

    // Coloring guarantees no other invocation in this dispatch touches these IDs.
    if a.inv_mass > 0.0 {
        particles[ids.x].position = a.position + a.inv_mass * delta_lambda * g0;
    }
    if b.inv_mass > 0.0 {
        particles[ids.y].position = b.position + b.inv_mass * delta_lambda * g1;
    }
    if c.inv_mass > 0.0 {
        particles[ids.z].position = c.position + c.inv_mass * delta_lambda * g2;
    }
    if d.inv_mass > 0.0 {
        particles[ids.w].position = d.position + d.inv_mass * delta_lambda * g3;
    }
}

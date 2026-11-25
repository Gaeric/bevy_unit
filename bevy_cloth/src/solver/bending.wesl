// Colored in-place XPBD dihedral solve; one accumulated lambda per hinge.

import super::layout::{Particle, BendConstraint, StepParams, ConstraintRangeParams};
import super::constraint_math::{evaluate_bend, xpbd_delta_lambda};

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(2)
var<uniform> range_params: ConstraintRangeParams;

@group(0) @binding(3)
var<storage, read> bend_constraints: array<BendConstraint>;

@group(0) @binding(4)
var<storage, read_write> bend_lambdas: array<f32>;

@compute @workgroup_size(64)
fn solve_bending(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let local_index = invocation.x;
    if local_index >= range_params.count {
        return;
    }

    let index = range_params.start + local_index;
    let constraint = bend_constraints[index];
    let ids = constraint.particles;
    let a = particles[ids.x];
    let b = particles[ids.y];
    let c = particles[ids.z];
    let d = particles[ids.w];
    if a.inv_mass + b.inv_mass + c.inv_mass + d.inv_mass == 0.0 {
        return;
    }

    let evaluation = evaluate_bend(a.position, b.position, c.position, d.position, constraint.rest_angle);
    if !evaluation.valid {
        return;
    }
    let g0 = evaluation.g0;
    let g1 = evaluation.g1;
    let g2 = evaluation.g2;
    let g3 = evaluation.g3;

    let alpha = constraint.compliance / (step_params.substep_dt * step_params.substep_dt);
    let denominator = a.inv_mass * dot(g0, g0)
        + b.inv_mass * dot(g1, g1)
        + c.inv_mass * dot(g2, g2)
        + d.inv_mass * dot(g3, g3) + alpha;
    if denominator == 0.0 {
        return;
    }
    let lambda = bend_lambdas[index];
    let delta_lambda = xpbd_delta_lambda(evaluation.error, lambda, alpha, denominator);
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

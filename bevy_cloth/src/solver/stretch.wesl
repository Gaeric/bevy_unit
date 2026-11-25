import super::layout::{Particle, StretchConstraint, StepParams, ConstraintRangeParams};
import super::constraint_math::{evaluate_stretch, xpbd_delta_lambda};

// Shared by particle and stretch stages.

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

// Stretch stage: one color range per dispatch.

@group(0) @binding(4)
var<uniform> stretch_range_params: ConstraintRangeParams;

@group(0) @binding(5)
var<storage, read> stretch_constraints: array<StretchConstraint>;

@group(0) @binding(6)
var<storage, read_write> stretch_lambdas: array<f32>;

@compute @workgroup_size(64)
fn solve_stretch(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let local_index = invocation.x;
    if local_index >= stretch_range_params.count {
        return;
    }

    let index = stretch_range_params.start + local_index;
    let constraint = stretch_constraints[index];

    let index_a = constraint.particles.x;
    let index_b = constraint.particles.y;
    let particle_a = particles[index_a];
    let particle_b = particles[index_b];

    let weight_a = particle_a.inv_mass;
    let weight_b = particle_b.inv_mass;

    if weight_a + weight_b == 0.0 {
        return;
    }

    let evaluation = evaluate_stretch(particle_a.position, particle_b.position, constraint.rest_length);
    if !evaluation.valid {
        return;
    }

    let normal = evaluation.direction;
    let dt = step_params.substep_dt;
    let alpha = constraint.compliance / (dt * dt);
    let lambda = stretch_lambdas[index];
    let delta_lambda = xpbd_delta_lambda(evaluation.error, lambda, alpha, weight_a + weight_b + alpha);

    stretch_lambdas[index] = lambda + delta_lambda;

    if weight_a > 0.0 {
        particles[index_a].position =
            particle_a.position + weight_a * delta_lambda * normal;
    }

    if weight_b > 0.0 {
        particles[index_b].position =
            particle_b.position - weight_b * delta_lambda * normal;
    }
}

// Each phase reads unchanged positions until apply_jacobi writes them back.
// Stretch uses its own adjacency; bend and tethers share a second phase.
import super::layout::{Particle, StretchConstraint, BendConstraint, TetherConstraint, StepParams};
import super::constraint_math::{evaluate_stretch, evaluate_bend, xpbd_delta_lambda, solve_tether};

@group(0) @binding(0) var<uniform> step_params: StepParams;
@group(0) @binding(1) var<storage, read> snapshot: array<Particle>;
@group(0) @binding(2) var<storage, read> incident_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> stretch_constraints: array<StretchConstraint>;
@group(0) @binding(4) var<storage, read_write> stretch_lambdas: array<f32>;
@group(0) @binding(5) var<storage, read> bend_constraints: array<BendConstraint>;
@group(0) @binding(6) var<storage, read_write> bend_lambdas: array<f32>;
@group(0) @binding(7) var<storage, read_write> corrections: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(9) var<storage, read> incidences: array<u32>;
@group(0) @binding(10) var<storage, read> tethers: array<TetherConstraint>;
@group(0) @binding(11) var<storage, read_write> tether_lambdas: array<f32>;

fn degree(particle: u32) -> f32 {
    return f32(incident_offsets[particle + 1u] - incident_offsets[particle]);
}

@compute @workgroup_size(64)
fn solve_stretch_jacobi(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= step_params.stretch_count {
        return;
    }

    let constraint = stretch_constraints[index];
    let ids = constraint.particles;
    let a = snapshot[ids.x];
    let b = snapshot[ids.y];
    let slot = 2u * index;
    // Overwrite every slot even when skipping a constraint; never reuse old deltas.
    corrections[slot] = vec4<f32>(0.0);
    corrections[slot + 1u] = vec4<f32>(0.0);
    if a.inv_mass + b.inv_mass == 0.0 {
        return;
    }
    let evaluation = evaluate_stretch(a.position, b.position, constraint.rest_length);
    if !evaluation.valid {
        return;
    }

    let dt = step_params.substep_dt;
    let alpha = constraint.compliance / (dt * dt);
    let denominator = degree(ids.x) * a.inv_mass + degree(ids.y) * b.inv_mass + alpha;
    let lambda = stretch_lambdas[index];
    let delta_lambda = xpbd_delta_lambda(evaluation.error, lambda, alpha, denominator);
    stretch_lambdas[index] = lambda + delta_lambda;
    corrections[slot] = vec4<f32>(a.inv_mass * delta_lambda * evaluation.direction, 0.0);
    corrections[slot + 1u] = vec4<f32>(-b.inv_mass * delta_lambda * evaluation.direction, 0.0);
}

@compute @workgroup_size(64)
fn solve_bend_jacobi(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= step_params.bend_count {
        return;
    }

    let constraint = bend_constraints[index];
    let ids = constraint.particles;
    let a = snapshot[ids.x];
    let b = snapshot[ids.y];
    let c = snapshot[ids.z];
    let d = snapshot[ids.w];
    let slot = 4u * index;
    corrections[slot] = vec4<f32>(0.0);
    corrections[slot + 1u] = vec4<f32>(0.0);
    corrections[slot + 2u] = vec4<f32>(0.0);
    corrections[slot + 3u] = vec4<f32>(0.0);
    if a.inv_mass + b.inv_mass + c.inv_mass + d.inv_mass == 0.0 {
        return;
    }
    let evaluation = evaluate_bend(a.position, b.position, c.position, d.position, constraint.rest_angle);
    if !evaluation.valid {
        return;
    }

    let dt = step_params.substep_dt;
    let alpha = constraint.compliance / (dt * dt);
    let denominator = degree(ids.x) * a.inv_mass * dot(evaluation.g0, evaluation.g0)
        + degree(ids.y) * b.inv_mass * dot(evaluation.g1, evaluation.g1)
        + degree(ids.z) * c.inv_mass * dot(evaluation.g2, evaluation.g2)
        + degree(ids.w) * d.inv_mass * dot(evaluation.g3, evaluation.g3) + alpha;
    if denominator == 0.0 {
        return;
    }
    let lambda = bend_lambdas[index];
    let delta_lambda = xpbd_delta_lambda(evaluation.error, lambda, alpha, denominator);
    bend_lambdas[index] = lambda + delta_lambda;
    corrections[slot] = vec4<f32>(a.inv_mass * delta_lambda * evaluation.g0, 0.0);
    corrections[slot + 1u] = vec4<f32>(b.inv_mass * delta_lambda * evaluation.g1, 0.0);
    corrections[slot + 2u] = vec4<f32>(c.inv_mass * delta_lambda * evaluation.g2, 0.0);
    corrections[slot + 3u] = vec4<f32>(d.inv_mass * delta_lambda * evaluation.g3, 0.0);
}

@compute @workgroup_size(64)
fn solve_tethers_jacobi(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= arrayLength(&tethers) {
        return;
    }
    let tether = tethers[index];
    let update = solve_tether(
        snapshot[tether.particle], snapshot[tether.anchor].position, tether,
        tether_lambdas[index], step_params.substep_dt, degree(tether.particle),
    );
    let slot = 4u * step_params.bend_count + index;
    tether_lambdas[index] = update.lambda;
    corrections[slot] = vec4<f32>(update.correction, 0.0);
}

@compute @workgroup_size(64)
fn apply_jacobi(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let particle = invocation.x;
    if particle >= step_params.particle_count || particles[particle].inv_mass == 0.0 {
        return;
    }
    var delta = vec3<f32>(0.0);
    for (var i = incident_offsets[particle]; i < incident_offsets[particle + 1u]; i += 1u) {
        delta += corrections[incidences[i]].xyz;
    }
    // The same damped delta_lambda updated each constraint's lambda and deltas.
    // Do not average again: this sum is the actual XPBD position update.
    particles[particle].position += delta;
}

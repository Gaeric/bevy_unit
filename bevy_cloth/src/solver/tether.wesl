import super::layout::{Particle, StepParams, TetherConstraint};
import super::constraint_math::solve_tether;

@group(0) @binding(0) var<uniform> step_params: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> tethers: array<TetherConstraint>;
@group(0) @binding(3) var<storage, read_write> lambdas: array<f32>;

@compute @workgroup_size(64)
fn solve_tethers(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= arrayLength(&tethers) {
        return;
    }
    // One constraint per free particle, and anchors are fixed and only read.
    // Therefore the entire tether array is one conflict-free GS batch.
    let tether = tethers[index];
    let particle = particles[tether.particle];
    let update = solve_tether(
        particle, particles[tether.anchor].position, tether,
        lambdas[index], step_params.substep_dt, 1.0,
    );
    lambdas[index] = update.lambda;
    if particle.inv_mass > 0.0 {
        particles[tether.particle].position = particle.position + update.correction;
    }
}

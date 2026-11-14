import super::layout::{Particle, StepParams};

// Shared by particle and stretch stages.

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

// Particle stages: prediction and velocity update.

@group(0) @binding(2)
var<storage, read_write> velocities: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> previous_positions: array<vec4<f32>>;

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= step_params.particle_count {
        return;
    }

    let particle = particles[index];
    previous_positions[index] = vec4<f32>(particle.position, 0.0);

    if particle.inv_mass == 0.0 {
        velocities[index] = vec4<f32>(0.0);
        return;
    }

    let dt = step_params.substep_dt;
    let velocity = velocities[index].xyz + step_params.gravity.xyz * dt;

    velocities[index] = vec4<f32>(velocity, 0.0);
    particles[index].position = particle.position + velocity * dt;
}

@compute @workgroup_size(64)
fn update_velocity(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= step_params.particle_count {
        return;
    }

    let particle = particles[index];

    if particle.inv_mass == 0.0 {
        velocities[index] = vec4<f32>(0.0);
        return;
    }

    let velocity =
        (particle.position - previous_positions[index].xyz) / step_params.substep_dt;

    velocities[index] = vec4<f32>(velocity, 0.0);
}

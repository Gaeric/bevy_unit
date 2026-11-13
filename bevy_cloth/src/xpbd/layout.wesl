// GPU data layouts, resource interfaces and XPBD solver entry points.

import super::bvh_query::{closest_hit, QueryHit, QUERY_OK, INVALID_TRIANGLE};

struct Particle {
    position: vec3<f32>,
    inv_mass: f32,
};

struct Stretch {
    particles: vec2<u32>,
    rest_length: f32,
    compliance: f32,
};

struct StepParams {
    gravity: vec4<f32>,
    dt: f32,
    particle_count: u32,
    stretch_count: u32,
    attachment_count: u32,
    collision_margin: f32,
    double_sided: u32,
    padding: vec2<u32>,
};

struct GroupParams {
    start: u32,
    count: u32,
    padding: vec2<u32>,
};

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

// Stretch stage: one color range per dispatch.

@group(0) @binding(4)
var<uniform> group_params: GroupParams;

@group(0) @binding(5)
var<storage, read> stretch: array<Stretch>;

@group(0) @binding(6)
var<storage, read_write> stretch_lambdas: array<f32>;

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

    let dt = step_params.dt;
    let velocity = velocities[index].xyz + step_params.gravity.xyz * dt;

    velocities[index] = vec4<f32>(velocity, 0.0);
    particles[index].position = particle.position + velocity * dt;
}

@compute @workgroup_size(64)
fn solve_stretch(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let local_index = invocation.x;
    if local_index >= group_params.count {
        return;
    }

    let index = group_params.start + local_index;
    let constraint = stretch[index];

    let index_a = constraint.particles.x;
    let index_b = constraint.particles.y;
    let particle_a = particles[index_a];
    let particle_b = particles[index_b];

    let weight_a = particle_a.inv_mass;
    let weight_b = particle_b.inv_mass;

    if weight_a + weight_b == 0.0 {
        return;
    }

    let separation = particle_a.position - particle_b.position;
    let distance = length(separation);

    // Coincident endpoints have no defined constraint direction.
    if distance == 0.0 {
        return;
    }

    let normal = separation / distance;
    let error = distance - constraint.rest_length;
    let dt = step_params.dt;
    let alpha = constraint.compliance / (dt * dt);
    let lambda = stretch_lambdas[index];

    let delta_lambda =
        (-error - alpha * lambda) / (weight_a + weight_b + alpha);

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

// Solid-shell contact signs the nearest distance using the winning face winding.
// Two-sided contact keeps the particle on its current side of the surface.
fn contact_correction(position: vec3<f32>, hit: QueryHit) -> vec3<f32> {
    if hit.status != QUERY_OK || hit.triangle_id == INVALID_TRIANGLE {
        return vec3<f32>(0.0);
    }

    let face_normal = hit.normal.xyz;
    if dot(face_normal, face_normal) == 0.0 {
        return vec3<f32>(0.0);
    }

    let separation = position - hit.point.xyz;
    let distance = sqrt(hit.point.w);

    var normal = face_normal;
    var signed_distance = distance;

    if distance > 0.0 {
        normal = separation / distance;

        if step_params.double_sided == 0u
            && dot(separation, face_normal) < 0.0
        {
            normal = -normal;
            signed_distance = -distance;
        }
    }

    let correction_length = step_params.collision_margin - signed_distance;
    if correction_length <= 0.0 {
        return vec3<f32>(0.0);
    }

    return correction_length * normal;
}

// Shared by tick-start stabilization and post-prediction substep contact.
@compute @workgroup_size(64)
fn solve_contact(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= step_params.particle_count {
        return;
    }

    let particle = particles[index];
    if particle.inv_mass == 0.0 {
        return;
    }

    let hit = closest_hit(particle.position);
    let correction = contact_correction(particle.position, hit);
    particles[index].position = particle.position + correction;
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
        (particle.position - previous_positions[index].xyz) / step_params.dt;

    velocities[index] = vec4<f32>(velocity, 0.0);
}

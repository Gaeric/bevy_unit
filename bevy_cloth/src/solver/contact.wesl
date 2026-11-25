import super::layout::{Particle, StepParams};
import super::super::collision::bvh_query::{
    closest_hit, QueryHit, QUERY_OK, INVALID_TRIANGLE, finite_scalar, finite_vector,
};

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(2)
var<storage, read> previous_positions: array<vec4<f32>>;

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

fn tangential_correction(index: u32, position: vec3<f32>, correction: vec3<f32>) -> vec3<f32> {
    if step_params.external_friction == 0.0 {
        return vec3<f32>(0.0);
    }
    let depth = length(correction);
    if depth == 0.0 || !finite_scalar(depth) {
        return vec3<f32>(0.0);
    }

    let relative_motion = position - previous_positions[index].xyz;
    let normal = correction / depth;
    let tangent = relative_motion - normal * dot(relative_motion, normal);
    let tangent_length = length(tangent);
    if !finite_vector(tangent) || !finite_scalar(tangent_length) || tangent_length == 0.0 {
        return vec3<f32>(0.0);
    }
    let length_cap = step_params.external_friction * depth;
    return -tangent * min(length_cap / tangent_length, 1.0);
}

fn is_movable(index: u32) -> bool {
    return index < step_params.particle_count && particles[index].inv_mass > 0.0;
}

@compute @workgroup_size(64)
fn stabilize_contact(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if !is_movable(index) {
        return;
    }
    let position = particles[index].position;
    let hit = closest_hit(position);
    particles[index].position = position + contact_correction(position, hit);
}

@compute @workgroup_size(64)
fn solve_contact(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if !is_movable(index) {
        return;
    }
    let position = particles[index].position;
    let hit = closest_hit(position);
    let normal_delta = contact_correction(position, hit);
    let tangent_delta = tangential_correction(index, position, normal_delta);
    particles[index].position = position + normal_delta + tangent_delta;
}

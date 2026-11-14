import super::layout::{Particle, StepParams};
import super::super::collision::bvh_query::{closest_hit, QueryHit, QUERY_OK, INVALID_TRIANGLE};

// Shared by particle and stretch stages.

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

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

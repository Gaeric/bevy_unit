import super::layout::{Particle, StepParams};
import super::super::collision::bvh_query::{
    closest_hit, QueryHit, QUERY_OK, INVALID_TRIANGLE, indices, finite_scalar, finite_vector,
};

@if(XPBD_CONTACT_TRACE)
import super::super::debug::xpbd::contact_trace::{contact_slot, trace_contact, trace_immovable};

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(2)
var<storage, read> previous_positions: array<vec4<f32>>;

@group(1) @binding(4)
var<uniform> surface_delta_per_tick: mat4x4<f32>;

@group(1) @binding(5)
var<storage, read> local_positions: array<vec4<f32>>;

struct ContactProjection {
    @if(XPBD_CONTACT_TRACE) normal: vec3<f32>,
    @if(XPBD_CONTACT_TRACE) signed_gap: f32,
    delta: vec3<f32>,
}

// Solid-shell contact signs the nearest distance using the winning face winding.
// Two-sided contact keeps the particle on its current side of the surface.
fn project_contact(position: vec3<f32>, hit: QueryHit) -> ContactProjection {
    var projection: ContactProjection;
    if hit.status != QUERY_OK || hit.triangle_id == INVALID_TRIANGLE {
        return projection;
    }

    let face_normal = hit.normal.xyz;
    if dot(face_normal, face_normal) == 0.0 {
        return projection;
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

    @if(XPBD_CONTACT_TRACE) projection.normal = normal;
    @if(XPBD_CONTACT_TRACE) projection.signed_gap = signed_distance;
    let correction_length = step_params.collision_margin - signed_distance;
    if correction_length <= 0.0 {
        return projection;
    }
    projection.delta = correction_length * normal;
    return projection;
}

fn tangential_correction(index: u32, position: vec3<f32>, hit: QueryHit, correction: vec3<f32>) -> vec3<f32> {
    if step_params.external_friction == 0.0 {
        return vec3<f32>(0.0);
    }
    let depth = length(correction);
    if depth == 0.0 || !finite_scalar(depth) {
        return vec3<f32>(0.0);
    }

    // Follow the same material point: triangle ID and barycentrics from the
    // current BVH hit, evaluated against the change in the collider's pose.
    let base = 3u * hit.triangle_id;
    let a = local_positions[indices[base]].xyz;
    let b = local_positions[indices[base + 1u]].xyz;
    let c = local_positions[indices[base + 2u]].xyz;
    let local_point = a + hit.barycentric.y * (b - a) + hit.barycentric.z * (c - a);
    let surface_motion = (surface_delta_per_tick * vec4<f32>(local_point, 1.0)).xyz
        / f32(step_params.substeps);
    let relative_motion = (position - previous_positions[index].xyz) - surface_motion;
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
    @if(XPBD_CONTACT_TRACE) let slot = contact_slot(index);
    if !is_movable(index) {
        @if(XPBD_CONTACT_TRACE) if slot != 0xffffffffu {
            trace_immovable(slot, particles[index].position);
        }
        return;
    }
    let position = particles[index].position;
    let hit = closest_hit(position);
    let projection = project_contact(position, hit);
    let after = position + projection.delta;
    particles[index].position = after;
    @if(XPBD_CONTACT_TRACE) if slot != 0xffffffffu {
        trace_contact(
            slot, position, after, hit, projection.normal, projection.signed_gap,
            step_params.collision_margin, projection.delta, vec3<f32>(0.0),
        );
    }
}

@compute @workgroup_size(64)
fn solve_contact(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    @if(XPBD_CONTACT_TRACE) let slot = contact_slot(index);
    if !is_movable(index) {
        @if(XPBD_CONTACT_TRACE) if slot != 0xffffffffu {
            trace_immovable(slot, particles[index].position);
        }
        return;
    }
    let position = particles[index].position;
    let hit = closest_hit(position);
    let projection = project_contact(position, hit);
    let tangent_delta = tangential_correction(index, position, hit, projection.delta);
    let after = position + projection.delta + tangent_delta;
    particles[index].position = after;
    @if(XPBD_CONTACT_TRACE) if slot != 0xffffffffu {
        trace_contact(
            slot, position, after, hit, projection.normal, projection.signed_gap,
            step_params.collision_margin, projection.delta, tangent_delta,
        );
    }
}

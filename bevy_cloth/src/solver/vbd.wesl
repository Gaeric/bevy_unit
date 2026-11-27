import super::layout::{Particle, StepParams, ConstraintRangeParams};
import super::vbd_energy::{IDENTITY_3, outer_self, evaluate_vbd_membrane, evaluate_vbd_hinge};
import super::super::collision::bvh_query::{closest_hit, QUERY_OK, INVALID_TRIANGLE};

struct Membrane {
    particles: vec3<u32>,
    rest_area: f32,
    grad1: vec2<f32>,
    grad2: vec2<f32>,
};

struct Hinge {
    particles: vec4<u32>,
    rest_angle: f32,
    rest_edge_length: f32,
    padding: vec2<u32>,
};

struct Incident {
    kind: u32, // 0 = membrane, 1 = hinge
    element: u32,
    corner: u32,
};

@group(0) @binding(0) var<uniform> step: StepParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
// First particle_count entries are x_n, the rest are fixed inertia targets y.
@group(0) @binding(2) var<storage, read_write> substep_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> color: ConstraintRangeParams;
struct Material {
    shear: f32,
    area: f32,
    bend: f32,
    contact_stiffness: f32,
    membrane_damping: f32,
    bend_damping: f32,
    padding: vec2<f32>,
};

@group(0) @binding(6) var<uniform> material: Material;
// CSR offsets (particle_count + 1), followed by the colored vertex list.
@group(0) @binding(7) var<storage, read> vertex_schedule: array<u32>;
@group(0) @binding(9) var<storage, read> incidents: array<Incident>;
@group(0) @binding(10) var<storage, read> membranes: array<Membrane>;
@group(0) @binding(11) var<storage, read> hinges: array<Hinge>;

struct NormalContact {
    point_and_distance_squared: vec4<f32>,
    normal: vec3<f32>, // zero when no candidate was found
};

@group(0) @binding(12) var<storage, read_write> contacts: array<NormalContact>;

@compute @workgroup_size(64)
fn reset_vbd_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x < step.particle_count {
        var empty: NormalContact;
        contacts[invocation.x] = empty;
    }
}

@compute @workgroup_size(64)
fn gather_vbd_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let vertex = invocation.x;
    if vertex >= step.particle_count || particles[vertex].inv_mass == 0.0 {
        return;
    }
    let position = particles[vertex].position;
    let hit = closest_hit(position);
    if hit.status != QUERY_OK || hit.triangle_id == INVALID_TRIANGLE
        || all(hit.normal.xyz == vec3<f32>(0.0))
    {
        return;
    }
    let old = contacts[vertex];
    if any(old.normal != vec3<f32>(0.0)) && old.point_and_distance_squared.w <= hit.point.w {
        return;
    }

    // Use the distance direction at an edge or corner, signed by winding for
    // solid shells. For two-sided surfaces choose the side at sweep start.
    var normal = hit.normal.xyz;
    let separation = position - hit.point.xyz;
    let distance_squared = dot(separation, separation);
    if distance_squared > 0.0 && distance_squared <= 3.402823466e38 {
        normal = separation * inverseSqrt(distance_squared);
        if step.double_sided == 0u && dot(separation, hit.normal.xyz) < 0.0 {
            normal = -normal;
        }
    }
    contacts[vertex] = NormalContact(hit.point, normal);
}

@compute @workgroup_size(64)
fn predict_vbd(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= step.particle_count {
        return;
    }
    let particle = particles[index];
    substep_positions[index] = vec4<f32>(particle.position, 0.0);
    var predicted = particle.position;
    if particle.inv_mass > 0.0 {
        let dt = step.substep_dt;
        predicted += dt * (velocities[index].xyz + dt * step.gravity.xyz);
        particles[index].position = predicted;
    }
    // Keep y fixed while later colors overwrite current positions.
    substep_positions[step.particle_count + index] = vec4<f32>(predicted, 0.0);
}

struct BlockSolution {
    delta: vec3<f32>,
    valid: bool,
};

fn cholesky(a: mat3x3<f32>, b: vec3<f32>) -> BlockSolution {
    var invalid: BlockSolution;
    if !(a[0].x > 0.0) {
        return invalid;
    }
    let l00 = sqrt(a[0].x);
    let l10 = a[0].y / l00;
    let l20 = a[0].z / l00;
    let pivot1 = a[1].y - l10 * l10;
    if !(pivot1 > 0.0) {
        return invalid;
    }
    let l11 = sqrt(pivot1);
    let l21 = (a[1].z - l20 * l10) / l11;
    let pivot2 = a[2].z - l20 * l20 - l21 * l21;
    if !(pivot2 > 0.0) {
        return invalid;
    }
    let l22 = sqrt(pivot2);

    // Solve L z = b, then L^T dx = z.
    let z0 = b.x / l00;
    let z1 = (b.y - l10 * z0) / l11;
    let z2 = (b.z - l20 * z0 - l21 * z1) / l22;
    let dx2 = z2 / l22;
    let dx1 = (z1 - l21 * dx2) / l11;
    let dx0 = (z0 - l10 * dx1 - l20 * dx2) / l00;
    return BlockSolution(vec3<f32>(dx0, dx1, dx2), true);
}

fn solve_spd(h: mat3x3<f32>, rhs: vec3<f32>) -> BlockSolution {
    var invalid: BlockSolution;
    let scale = max(h[0].x, max(h[1].y, h[2].z));
    if !(scale > 0.0 && scale <= 3.402823466e38) {
        return invalid;
    }
    let a = mat3x3<f32>(h[0] / scale, h[1] / scale, h[2] / scale);
    let b = rhs / scale;
    let result = cholesky(a, b);
    if result.valid {
        return result;
    }
    // A large elastic block can round away the inertia on its diagonal.
    // Retry with a small scale-relative shift; do not change the force.
    return cholesky(a + 1e-6 * IDENTITY_3, b);
}

@compute @workgroup_size(64)
fn solve_vbd_color(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= color.count {
        return;
    }
    let vertex = vertex_schedule[step.particle_count + 1u + color.start + invocation.x];
    let particle = particles[vertex];
    if particle.inv_mass == 0.0 {
        return;
    }

    let dt = step.substep_dt;
    let inertia = (1.0 / particle.inv_mass) / (dt * dt);
    var force = inertia * (substep_positions[step.particle_count + vertex].xyz - particle.position);
    var hessian = inertia * IDENTITY_3;
    for (var i = vertex_schedule[vertex]; i < vertex_schedule[vertex + 1u]; i += 1u) {
        let incident = incidents[i];
        if incident.kind == 0u {
            // Preparation includes an element only when elasticity or damping is enabled.
            let triangle = membranes[incident.element];
            let ids = triangle.particles;
            let x = array(particles[ids.x].position, particles[ids.y].position, particles[ids.z].position);
            var start: array<vec3<f32>, 3>;
            if material.membrane_damping > 0.0 {
                start = array(substep_positions[ids.x].xyz, substep_positions[ids.y].xyz, substep_positions[ids.z].xyz);
            }
            let block = evaluate_vbd_membrane(
                x, incident.corner, triangle.grad1, triangle.grad2, triangle.rest_area,
                material.shear, material.area, start, material.membrane_damping / dt,
            );
            force += block.force;
            hessian += block.hessian;
        } else if incident.kind == 1u {
            let hinge = hinges[incident.element];
            let ids = hinge.particles;
            let x = array(
                particles[ids.x].position, particles[ids.y].position,
                particles[ids.z].position, particles[ids.w].position,
            );
            var start: array<vec3<f32>, 4>;
            if material.bend_damping > 0.0 {
                start = array(
                    substep_positions[ids.x].xyz, substep_positions[ids.y].xyz,
                    substep_positions[ids.z].xyz, substep_positions[ids.w].xyz,
                );
            }
            let block = evaluate_vbd_hinge(
                x, incident.corner, hinge.rest_angle, hinge.rest_edge_length,
                material.bend, start, material.bend_damping / dt,
            );
            force += block.force;
            hessian += block.hessian;
        }
    }

    if material.contact_stiffness > 0.0 {
        let contact = contacts[vertex];
        let normal = contact.normal;
        let gap = dot(particle.position - contact.point_and_distance_squared.xyz, normal);
        let compression = step.collision_margin - gap;
        if compression > 0.0 && any(normal != vec3<f32>(0.0)) {
            force += (material.contact_stiffness * compression) * normal;
            hessian += material.contact_stiffness * outer_self(normal);
        }
    }

    let solution = solve_spd(hessian, force);
    let next_position = particle.position + solution.delta;
    if solution.valid && all(abs(next_position) <= vec3<f32>(3.402823466e38)) {
        particles[vertex].position = next_position;
    }
}

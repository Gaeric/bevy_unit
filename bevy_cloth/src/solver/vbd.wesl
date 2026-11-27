import super::layout::{Particle, StepParams, ConstraintRangeParams};
import super::vbd_energy::{IDENTITY_3, evaluate_vbd_membrane, evaluate_vbd_hinge};

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
@group(0) @binding(2) var<storage, read_write> inertia_targets: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> previous_positions: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> color: ConstraintRangeParams;
@group(0) @binding(6) var<uniform> material: vec4<f32>; // shear, area, bend, unused
@group(0) @binding(7) var<storage, read> vertices: array<u32>;
@group(0) @binding(8) var<storage, read> offsets: array<u32>;
@group(0) @binding(9) var<storage, read> incidents: array<Incident>;
@group(0) @binding(10) var<storage, read> membranes: array<Membrane>;
@group(0) @binding(11) var<storage, read> hinges: array<Hinge>;

@compute @workgroup_size(64)
fn predict_vbd(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= step.particle_count {
        return;
    }
    let particle = particles[index];
    previous_positions[index] = vec4<f32>(particle.position, 0.0);
    var predicted = particle.position;
    if particle.inv_mass > 0.0 {
        let dt = step.substep_dt;
        predicted += dt * (velocities[index].xyz + dt * step.gravity.xyz);
        particles[index].position = predicted;
    }
    // Keep y fixed while later colors overwrite current positions.
    inertia_targets[index] = vec4<f32>(predicted, 0.0);
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
    let vertex = vertices[color.start + invocation.x];
    let particle = particles[vertex];
    if particle.inv_mass == 0.0 {
        return;
    }

    let dt = step.substep_dt;
    let inertia = (1.0 / particle.inv_mass) / (dt * dt);
    var force = inertia * (inertia_targets[vertex].xyz - particle.position);
    var hessian = inertia * IDENTITY_3;
    for (var i = offsets[vertex]; i < offsets[vertex + 1u]; i += 1u) {
        let incident = incidents[i];
        if incident.kind == 0u && (material.x > 0.0 || material.y > 0.0) {
            let triangle = membranes[incident.element];
            let ids = triangle.particles;
            let x = array(particles[ids.x].position, particles[ids.y].position, particles[ids.z].position);
            let block = evaluate_vbd_membrane(
                x, incident.corner, triangle.grad1, triangle.grad2, triangle.rest_area,
                material.x, material.y,
            );
            force += block.force;
            hessian += block.hessian;
        } else if incident.kind == 1u && material.z > 0.0 {
            let hinge = hinges[incident.element];
            let ids = hinge.particles;
            let x = array(
                particles[ids.x].position, particles[ids.y].position,
                particles[ids.z].position, particles[ids.w].position,
            );
            let block = evaluate_vbd_hinge(
                x, incident.corner, hinge.rest_angle, hinge.rest_edge_length,
                material.z,
            );
            force += block.force;
            hessian += block.hessian;
        }
    }

    let solution = solve_spd(hessian, force);
    let next_position = particle.position + solution.delta;
    if solution.valid && all(abs(next_position) <= vec3<f32>(3.402823466e38)) {
        particles[vertex].position = next_position;
    }
}

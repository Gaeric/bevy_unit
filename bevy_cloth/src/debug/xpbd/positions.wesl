import super::super::super::solver::layout::Particle;
import super::contact_trace::{CaptureParams, SelectedParticles};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> selected: SelectedParticles;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: CaptureParams;

@compute @workgroup_size(64)
fn capture_constraint_positions(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let slot = invocation.x;
    if slot >= params.particle_count {
        return;
    }
    let id = selected.ids[slot / 4u][slot % 4u];
    positions[params.output_base + slot] = vec4<f32>(particles[id].position, 0.0);
}

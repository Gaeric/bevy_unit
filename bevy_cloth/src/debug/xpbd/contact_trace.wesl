import super::super::super::collision::bvh_query::{QueryHit, empty_hit};

struct CaptureParams {
    output_base: u32,
    particle_count: u32,
    _padding: vec2<u32>,
}

struct SelectedParticles {
    // A uniform rather than an additional storage binding. Packing into vec4s
    // keeps all 64 IDs in 256 bytes under uniform layout rules.
    ids: array<vec4<u32>, 16>,
}

struct ContactSample {
    before_and_gap: vec4<f32>,
    after_and_margin: vec4<f32>,
    point: vec4<f32>,
    normal: vec4<f32>,
    normal_delta: vec4<f32>,
    friction_delta: vec4<f32>,
    query: vec4<u32>, // status, triangle ID, sample kind (1 queried, 2 immovable), padding
}

@group(2) @binding(0) var<uniform> trace_ids: SelectedParticles;
@group(2) @binding(1) var<storage, read_write> trace_samples: array<ContactSample>;
@group(2) @binding(2) var<uniform> trace_params: CaptureParams;

fn contact_slot(particle: u32) -> u32 {
    var first = 0u;
    var end = trace_params.particle_count;
    while first < end {
        let middle = first + (end - first) / 2u;
        let id = trace_ids.ids[middle / 4u][middle % 4u];
        if id == particle {
            return trace_params.output_base + middle;
        }
        if id < particle {
            first = middle + 1u;
        } else {
            end = middle;
        }
    }
    return 0xffffffffu;
}

fn trace_immovable(slot: u32, position: vec3<f32>) {
    let hit = empty_hit();
    let zero = vec4<f32>(0.0);
    trace_samples[slot] = ContactSample(
        vec4<f32>(position, 0.0), vec4<f32>(position, 0.0),
        zero, zero, zero, zero, vec4<u32>(hit.status, hit.triangle_id, 2u, 0u),
    );
}

fn trace_contact(
    slot: u32, before: vec3<f32>, after: vec3<f32>, hit: QueryHit,
    normal: vec3<f32>, gap: f32, margin: f32,
    normal_delta: vec3<f32>, friction_delta: vec3<f32>,
) {
    trace_samples[slot] = ContactSample(
        vec4<f32>(before, gap), vec4<f32>(after, margin), hit.point,
        vec4<f32>(normal, 0.0), vec4<f32>(normal_delta, 0.0), vec4<f32>(friction_delta, 0.0),
        vec4<u32>(hit.status, hit.triangle_id, 1u, 0u),
    );
}

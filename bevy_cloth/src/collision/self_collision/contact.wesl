import super::super::super::solver::layout::Particle;
import super::self_layout::{params, finite3, largest_component, bump_stat, INVALID_CORRECTIONS};

@group(0) @binding(1) var<storage, read> snapshot: array<Particle>;
@group(0) @binding(3) var<storage, read> rest_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> candidates: array<u32>;
@group(0) @binding(5) var<storage, read_write> gathered: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> corrections: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> particles: array<Particle>;

fn contains_candidate(row: u32, id: u32) -> bool {
    let count = candidates[row];
    var low = 0u;
    var high = count;
    while low < high {
        let mid = low + (high - low) / 2u;
        let value = candidates[row + (mid + 1u) * params.particle_count];
        if value == id { return true; }
        if value < id { low = mid + 1u; }
        else { high = mid; }
    }
    return false;
}

// The caller always orders the endpoints by ID. Both endpoints reconstruct the
// same relative correction; the high endpoint uses its negative.
fn pair_normal_correction(low: u32, high: u32, delta: vec3<f32>) -> vec4<f32> {
    let scale = largest_component(delta);
    if scale > 0.0 {
        let scaled = delta / scale;
        let scaled_length = length(scaled);
        return vec4<f32>(scaled / scaled_length, max(params.diameter - scale * scaled_length, 0.0));
    }

    // Scale endpoints before subtraction if the finite rest coordinates can
    // overflow on subtraction. Rest direction is preferred only at coincidence.
    let rest_low = rest_positions[low].xyz;
    let rest_high = rest_positions[high].xyz;
    var rest_delta = rest_low - rest_high;
    if !finite3(rest_delta) { rest_delta = rest_low * 0.5 - rest_high * 0.5; }
    let rest_scale = largest_component(rest_delta);
    if finite3(rest_delta) && rest_scale > 0.0 {
        return vec4<f32>(normalize(rest_delta / rest_scale), params.diameter);
    }

    let axis = ((low * 92837111u) ^ (high * 689287499u)) % 3u;
    var fallback = vec3<f32>(0.0);
    fallback[axis] = 1.0;
    return vec4<f32>(fallback, params.diameter);
}

@compute @workgroup_size(64)
fn gather_self_contact(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= params.particle_count { return; }
    gathered[i] = vec4<f32>(0.0);
    let weight = snapshot[i].inv_mass;
    if weight == 0.0 { return; }

    var correction = vec3<f32>(0.0);
    let count = candidates[i];
    for (var slot = 1u; slot <= count; slot++) {
        let j = candidates[i + slot * params.particle_count];
        // Truncation must never cause a free endpoint to solve a one-sided pair.
        if !contains_candidate(j, i) { continue; }
        let low = min(i, j);
        let high = max(i, j);
        let delta = snapshot[low].position - snapshot[high].position;
        let contact = pair_normal_correction(low, high, delta);
        let relative = contact.xyz * contact.w;
        let other_weight = snapshot[j].inv_mass;
        let scale = max(weight, other_weight);
        let share = (weight / scale) / (weight / scale + other_weight / scale);
        // Damp crowded rows without changing a pair's relative mass balance:
        // both endpoints use the same (conservative) count before accumulation.
        let pair_degree = f32(max(count, candidates[j]));
        correction += relative * (select(-1.0, 1.0, i == low) * share / pair_degree);
    }
    gathered[i] = vec4<f32>(correction, 0.0);
}

@compute @workgroup_size(64)
fn apply_self_contact(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= params.particle_count || particles[i].inv_mass == 0.0 { return; }
    // Preserve invalid particles for diagnosis rather than turning them into
    // valid positions. The hash already reports invalid coordinates.
    let correction = corrections[i].xyz;
    if all(correction == vec3<f32>(0.0)) { return; }
    let corrected = particles[i].position + correction;
    if !finite3(corrected) {
        bump_stat(INVALID_CORRECTIONS);
        return;
    }
    particles[i].position = corrected;
}

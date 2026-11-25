import super::super::super::solver::layout::Particle;
import super::self_layout::{params, stats, bump_stat, SUBSTEPS, MAX_REQUIRED, OVERFLOW_ROWS,
    INVALID_PARTICLES, finite3, cell_bucket, distance_ratio_squared};

@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> bucket_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bucket_cursors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> bucket_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> bucket_particles: array<u32>;
@group(0) @binding(6) var<storage, read> rest_positions: array<vec4<f32>>;
// Slot-major: the first N entries are row lengths; each following N entries are one slot.
@group(0) @binding(7) var<storage, read_write> candidates: array<u32>;

fn valid_cell(scaled: vec3<f32>) -> bool {
    // Check before converting to i32. The bound and the 2*diameter cell width
    // leave room for f32 rounding without missing pairs across cell boundaries.
    return finite3(scaled) && all(abs(scaled) <= vec3<f32>(4194304.0));
}

@compute @workgroup_size(64)
fn clear_hash(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let bucket = invocation.x;
    if bucket >= params.bucket_count { return; }
    atomicStore(&bucket_counts[bucket], 0u);
    atomicStore(&bucket_cursors[bucket], 0u);
    if bucket == 0u { bump_stat(SUBSTEPS); }
}

@compute @workgroup_size(64)
fn count_hash(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= params.particle_count { return; }
    let scaled = particles[i].position / params.cell_width;
    if !valid_cell(scaled) {
        bump_stat(INVALID_PARTICLES);
        return;
    }
    let bucket = cell_bucket(vec3<i32>(floor(scaled)), params.bucket_count);
    atomicAdd(&bucket_counts[bucket], 1u);
}

@compute @workgroup_size(64)
fn scatter_hash(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= params.particle_count { return; }
    let scaled = particles[i].position / params.cell_width;
    if !valid_cell(scaled) { return; }
    let bucket = cell_bucket(vec3<i32>(floor(scaled)), params.bucket_count);
    let slot = atomicAdd(&bucket_cursors[bucket], 1u);
    bucket_particles[bucket_offsets[bucket] + slot] = i;
}

fn candidate_index(particle: u32, slot: u32) -> u32 {
    return particle + (slot + 1u) * params.particle_count;
}

fn insert_candidate(i: u32, j: u32, retained: u32) -> u32 {
    // Keep the smallest IDs in ascending order even if scatter order changes.
    var insertion = retained;
    if retained == params.capacity {
        insertion = retained - 1u;
        if j >= candidates[candidate_index(i, insertion)] { return retained; }
    }
    while insertion > 0u {
        let previous = candidates[candidate_index(i, insertion - 1u)];
        if previous < j { break; }
        candidates[candidate_index(i, insertion)] = previous;
        insertion--;
    }
    candidates[candidate_index(i, insertion)] = j;
    return min(retained + 1u, params.capacity);
}

@compute @workgroup_size(64)
fn find_candidates(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= params.particle_count { return; }
    let point = particles[i];
    let scaled = point.position / params.cell_width;
    candidates[i] = 0u;
    if !valid_cell(scaled) { return; }
    let home = vec3<i32>(floor(scaled));
    let rest = rest_positions[i].xyz;
    var required = 0u;
    var retained = 0u;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let cell = home + vec3<i32>(x, y, z);
                let bucket = cell_bucket(cell, params.bucket_count);
                let begin = bucket_offsets[bucket];
                let end = begin + atomicLoad(&bucket_counts[bucket]);
                for (var entry = begin; entry < end; entry++) {
                    let j = bucket_particles[entry];
                    if i == j { continue; }
                    let other = particles[j];
                    if point.inv_mass == 0.0 && other.inv_mass == 0.0 { continue; }
                    // Visit each true cell once. An ID in an aliased bucket can match
                    // only one queried cell, so no duplicate can reach the row.
                    if any(vec3<i32>(floor(other.position / params.cell_width)) != cell) { continue; }
                    if distance_ratio_squared(point.position - other.position, params.diameter) >= 1.0 { continue; }
                    if distance_ratio_squared(rest - rest_positions[j].xyz, params.diameter) <= 1.0 { continue; }
                    required++;
                    retained = insert_candidate(i, j, retained);
                }
            }
        }
    }
    candidates[i] = retained;
    if required != 0u { atomicMax(&stats[MAX_REQUIRED], required); }
    if required > params.capacity { bump_stat(OVERFLOW_ROWS); }
}

// gpu solver (plan.org M6). one entry point per phase; the host records the whole schedule into a
// single compute pass, so the phase boundaries exist because WGSL has no device-wide barrier.
//
// the constraint phases are gather (one thread per particle walking its own incident constraints)
// instead of Velvet's scatter + float atomic, which standard WebGPU does not have.

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 1e-6;

struct Params {
  gravity: vec4<f32>,
  num_particles: u32,
  substep_time: f32,
  max_speed: f32,
  damping: f32,
  relaxation_factor: f32,
  long_range_stretchiness: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> predicted: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> deltas: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> delta_counts: array<i32>;
@group(0) @binding(6) var<storage, read> inv_masses: array<f32>;
// per-particle incidence into the stretch constraints: `other` is the far endpoint and `rest` its
// rest length, so a solve step never has to read the flat constraint arrays
@group(0) @binding(7) var<storage, read> stretch_inc_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> stretch_inc_other: array<u32>;
@group(0) @binding(9) var<storage, read> stretch_inc_rest: array<f32>;

struct AttachEntry {
  slot: u32,
  rest: f32,
};
@group(0) @binding(10) var<storage, read> attach_inc_offsets: array<u32>;
@group(0) @binding(11) var<storage, read> attach_inc_entries: array<AttachEntry>;
@group(0) @binding(12) var<storage, read> attach_slot_positions: array<vec4<f32>>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_particles) {
    return;
  }

  let velocity = velocities[i].xyz + params.gravity.xyz * params.substep_time;
  velocities[i] = vec4<f32>(velocity, 0.0);
  predicted[i] = vec4<f32>(positions[i].xyz + velocity * params.substep_time, 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn solve_stretch(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_particles) {
    return;
  }

  let weight = inv_masses[i];
  var delta = vec3<f32>(0.0);
  var count = 0;

  let start = stretch_inc_offsets[i];
  let end = stretch_inc_offsets[i + 1];
  for (var k = start; k < end; k = k + 1u) {
    let other = stretch_inc_other[k];
    let weight_other = inv_masses[other];
    let denom = weight + weight_other;
    if (denom <= 0.0) {
      continue;
    }

    let diff = predicted[i].xyz - predicted[other].xyz;
    let distance = length(diff);
    let rest = stretch_inc_rest[k];
    if (distance == rest) {
      continue;
    }

    let gradient = diff / (distance + EPSILON);
    let lambda = (distance - rest) / denom;
    delta -= weight * lambda * gradient;
    count = count + 1;
  }

  deltas[i] = vec4<f32>(delta, 0.0);
  delta_counts[i] = count;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn solve_attachment(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_particles) {
    return;
  }

  // Append AFTER stretch, keeping its contributions (including zero corrections at pins).
  var delta = deltas[i].xyz;
  var count = delta_counts[i];
  let start = attach_inc_offsets[i];
  let end = attach_inc_offsets[i + 1u];
  for (var k = start; k < end; k = k + 1u) {
    let entry = attach_inc_entries[k];
    let target_dist = entry.rest * params.long_range_stretchiness;
    // CPU/CUDA test the scaled target, not the raw rest length.
    if (inv_masses[i] == 0.0 && target_dist > 0.0) {
      continue;
    }
    let diff = predicted[i].xyz - attach_slot_positions[entry.slot].xyz;
    let dist = length(diff);
    if (dist > target_dist) {
      delta += -diff + diff / dist * target_dist;
      count = count + 1;
    }
  }
  deltas[i] = vec4<f32>(delta, 0.0);
  delta_counts[i] = count;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_deltas(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_particles) {
    return;
  }

  let count = delta_counts[i];
  if (count > 0) {
    predicted[i] = vec4<f32>(
      predicted[i].xyz + deltas[i].xyz / f32(count) * params.relaxation_factor,
      0.0,
    );
    deltas[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    delta_counts[i] = 0;
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_particles) {
    return;
  }

  let position = positions[i].xyz;
  var new_position = predicted[i].xyz;
  var velocity = (new_position - position) / params.substep_time;

  let speed = length(velocity);
  if (speed > params.max_speed) {
    velocity = velocity / speed * params.max_speed;
    new_position = position + velocity * params.substep_time;
  }

  velocities[i] = vec4<f32>(velocity * (1.0 - params.damping * params.substep_time), 0.0);
  positions[i] = vec4<f32>(new_position, 0.0);
}

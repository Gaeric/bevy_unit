// m6 v1: only `predict` / `finalize` so far; the rest of the phase schedule lands later. the host
// records the whole schedule into one compute pass, so the phase boundaries exist because WGSL has
// no device-wide barrier (storage memory is only guaranteed visible across dispatches).

const WORKGROUP_SIZE: u32 = 256u;

struct Params {
  gravity: vec4<f32>,
  num_particles: u32,
  substep_time: f32,
  max_speed: f32,
  damping: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> predicted: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4<f32>>;

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

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
  bend_compliance: f32,
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

struct BendConstraint {
  indices: vec4<u32>,
  rest_angle: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};
@group(0) @binding(13) var<storage, read> bend_inc_offsets: array<u32>;
// Each entry is constraint_id * 4 + endpoint role, including repeated roles of one particle.
@group(0) @binding(14) var<storage, read> bend_inc_entries: array<u32>;
@group(0) @binding(15) var<storage, read> bend_constraints: array<BendConstraint>;

struct SdfParams {
  num_colliders: u32,
  collision_margin: f32,
  friction: f32,
  step_time: f32,
};
struct SdfCollider {
  kind: u32,
  delta_time: f32,
  pad0: u32,
  pad1: u32,
  position: vec4<f32>,
  scale: vec4<f32>,
  cur_transform: mat3x3<f32>,
  inv_cur_transform: mat4x4<f32>,
  last_transform: mat4x4<f32>,
};
@group(0) @binding(16) var<uniform> sdf_params: SdfParams;
@group(0) @binding(17) var<storage, read> sdf_colliders: array<SdfCollider>;

// sign(0) is zero, not +1; preserve even the -0 sign bit for CPU cube-face ties.
fn copy_sign(magnitude: f32, source: f32) -> f32 {
  return bitcast<f32>((bitcast<u32>(magnitude) & 0x7fffffffu)
    | (bitcast<u32>(source) & 0x80000000u));
}

fn sdf_correction(collider: SdfCollider, point: vec3<f32>) -> vec3<f32> {
  let margin = sdf_params.collision_margin;
  if (collider.kind == 0u) {
    // Plane rotation is deliberately ignored, as in the CPU reference.
    let offset = point.y - (collider.position.y + margin);
    if (offset < 0.0) {
      return vec3<f32>(0.0, -offset, 0.0);
    }
  } else if (collider.kind == 1u) {
    let diff = point - collider.position.xyz;
    let distance = length(diff);
    let offset = distance - (collider.scale.x + margin);
    if (offset < 0.0 && distance > 0.0) {
      return -offset * (diff / distance);
    }
  } else if (collider.kind == 2u) {
    let local = (collider.inv_cur_transform * vec4<f32>(point, 1.0)).xyz;
    let cube_size = vec3<f32>(0.5) + vec3<f32>(margin) / collider.scale.xyz;
    let offset = abs(local) - cube_size;
    let max_val = max(max(offset.x, offset.y), offset.z);
    let min_val = min(min(offset.x, offset.y), offset.z);
    let mid_val = offset.x + offset.y + offset.z - max_val - min_val;
    var scalar = 1.0;
    var correction = vec3<f32>(0.0);
    if (max_val < 0.0) {
      if (mid_val > -0.03) {
        scalar = 0.2;
      }
      if (min_val > -0.03) {
        let mask = select(vec3<f32>(0.0), sign(local), offset < vec3<f32>(0.0));
        let rounded = offset + vec3<f32>(0.03);
        let len = length(rounded);
        if (len < 0.03 && len > 0.0) {
          correction = mask * (rounded / len) * (0.03 - len);
        }
      } else if (offset.x == max_val) {
        correction.x = copy_sign(-offset.x, local.x);
      } else if (offset.y == max_val) {
        correction.y = copy_sign(-offset.y, local.y);
      } else if (offset.z == max_val) {
        correction.z = copy_sign(-offset.z, local.z);
      }
    }
    // This is the CPU's scaled model matrix, NOT inverse-transpose normal projection.
    return (collider.cur_transform * scalar) * correction;
  }
  return vec3<f32>(0.0);
}

fn sdf_velocity_at(collider: SdfCollider, point: vec3<f32>) -> vec3<f32> {
  if (collider.delta_time == 0.0) {
    return vec3<f32>(0.0);
  }
  let last = ((collider.last_transform * collider.inv_cur_transform)
    * vec4<f32>(point, 1.0)).xyz;
  return (point - last) / collider.delta_time;
}

fn sdf_friction(correction: vec3<f32>, displacement: vec3<f32>) -> vec3<f32> {
  let correction_length = length(correction);
  if (sdf_params.friction > 0.0 && correction_length > 0.0) {
    let normal = correction / correction_length;
    let tangent = displacement - normal * dot(displacement, normal);
    let tangent_length = length(tangent);
    // Includes friction=0 / normal-only motion without a speculative zero division.
    if (tangent_length > 0.0) {
      return -tangent * min(correction_length * sdf_params.friction / tangent_length, 1.0);
    }
  }
  return vec3<f32>(0.0);
}

fn project_sdfs(old_position: vec3<f32>, initial: vec3<f32>, h: f32) -> vec3<f32> {
  var pred = initial;
  // Each invocation owns its particle; preserve collider order, including zero-mass pins.
  for (var k = 0u; k < sdf_params.num_colliders; k = k + 1u) {
    let collider = sdf_colliders[k];
    let correction = sdf_correction(collider, pred);
    pred += correction;
    if (dot(correction, correction) > 0.0) {
      let displacement = pred - old_position - sdf_velocity_at(collider, pred) * h;
      pred += sdf_friction(correction, displacement);
    }
  }
  return pred;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn stabilize_sdf(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_particles || sdf_params.num_colliders == 0u) {
    return;
  }
  // Own local read snapshot replaces CUDA's aliased positions input/output binding.
  // Do not touch predicted or velocities; predict refreshes predicted next dispatch.
  let old_position = positions[i].xyz;
  positions[i] = vec4<f32>(project_sdfs(old_position, old_position, sdf_params.step_time), 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn collide_sdf(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_particles || sdf_params.num_colliders == 0u) {
    return;
  }
  predicted[i] = vec4<f32>(
    project_sdfs(positions[i].xyz, predicted[i].xyz, params.substep_time), 0.0);
}

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
fn solve_bending(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.num_particles || !(params.substep_time > 0.0)) {
    return;
  }

  // Append AFTER attachment; predicted is immutable until the following apply dispatch.
  var delta = deltas[i].xyz;
  var count = delta_counts[i];
  let compliance = params.bend_compliance / (params.substep_time * params.substep_time);
  for (var k = bend_inc_offsets[i]; k < bend_inc_offsets[i + 1u]; k = k + 1u) {
    let encoded = bend_inc_entries[k];
    let constraint = bend_constraints[encoded / 4u];
    let ids = constraint.indices;
    let p0 = predicted[ids.x].xyz;
    let p1 = predicted[ids.y].xyz;
    let p2 = predicted[ids.z].xyz;
    let p3 = predicted[ids.w].xyz;
    // The shared edge is p2--p3, not p0--p1.
    let edge = p3 - p2;
    let edge_length = length(edge);
    if (edge_length < EPSILON) {
      continue;
    }
    let cross1 = cross(p2 - p0, p3 - p0);
    let cross2 = cross(p3 - p1, p2 - p1);
    let cross1_sq = dot(cross1, cross1);
    let cross2_sq = dot(cross2, cross2);
    if (cross1_sq < EPSILON || cross2_sq < EPSILON) {
      continue;
    }
    var n1 = cross1 / cross1_sq;
    var n2 = cross2 / cross2_sq;
    let d0 = edge_length * n1;
    let d1 = edge_length * n2;
    let inv_edge_length = 1.0 / edge_length;
    let d2 = dot(p0 - p3, edge) * inv_edge_length * n1
           + dot(p1 - p3, edge) * inv_edge_length * n2;
    let d3 = dot(p2 - p0, edge) * inv_edge_length * n1
           + dot(p2 - p1, edge) * inv_edge_length * n2;
    n1 = normalize(n1);
    n2 = normalize(n2);
    let angle = acos(clamp(dot(n1, n2), -1.0, 1.0));
    let denominator = inv_masses[ids.x] * dot(d0, d0)
                    + inv_masses[ids.y] * dot(d1, d1)
                    + inv_masses[ids.z] * dot(d2, d2)
                    + inv_masses[ids.w] * dot(d3, d3) + compliance;
    // CPU guards the full denominator AFTER compliance. Ordered comparisons also reject NaN;
    // bitcast gives the largest finite f32 exactly, without relying on an isFinite builtin.
    if (!(denominator >= EPSILON && denominator <= bitcast<f32>(0x7f7fffffu))) {
      continue;
    }
    var lambda = (angle - constraint.rest_angle) / denominator;
    if (dot(cross(n1, n2), edge) > 0.0) {
      lambda = -lambda;
    }
    var gradients = array<vec3<f32>, 4>(d0, d1, d2, d3);
    delta += -inv_masses[i] * lambda * gradients[encoded % 4u];
    // Valid zero-error constraints and zero-weight endpoint roles still affect Jacobi averaging.
    count = count + 1;
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

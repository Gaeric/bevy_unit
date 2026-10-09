// Resolves one intermediate buffer into ViewTarget.

const VIEW_PREPASS_DEPTH: u32 = 0u;
const VIEW_PREPASS_NORMAL: u32 = 1u;
const VIEW_PREPASS_MOTION: u32 = 2u;
const VIEW_SOLID: u32 = 3u;
const VIEW_TRACE_DEPTH: u32 = 4u;
const VIEW_TRACE_DIFF: u32 = 5u;

const BACKGROUND: vec3<f32> = vec3<f32>(0.10, 0.04, 0.16);
const TRACE_DEPTH_RANGE: f32 = 5.0;
const TRACE_DIFF_SCALE: f32 = 20.0;

struct CompositeUniform {
   view_mode: u32, 
   _pad0: u32,
   _pad1: u32,
   _pad2: u32,
};

@group(0) @binding(0) var<uniform> composite: CompositeUniform;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var motion_vectors_tex: texture_2d<f32>;
@group(0) @binding(4) var trace_tex: texture_2d<f32>;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32>
{
  let coords = vec2<i32>(frag_coord.xy);
  var color = vec3<f32>(0.10, 0.04, 0.16);

  switch composite.view_mode {
    case VIEW_PREPASS_DEPTH: {
      // same as `bevy_dev_tools::debug_overlay`'s DEBUG_DEPTH arm: bevy's reverse-z
      // depth is already "1.0 at the near plane .. 0.0 at infinite far", so it is
      // shown as-is instead of being linearized into a distance.
      color = vec3<f32>(textureLoad(depth_tex, coords, 0));
    }
    case VIEW_PREPASS_NORMAL: {
      let normal = normalize(textureLoad(normal_tex, coords, 0).xyz * 2.0 - 1.0);
      color = normal * 0.5 + 0.5;
    }
    case VIEW_PREPASS_MOTION: {
      let motion = textureLoad(motion_vectors_tex, coords, 0).xy;
      color = vec3<f32>(motion * 0.5 + 0.5, 0.0);
    }
    case VIEW_SOLID: {
      color = vec3<f32>(0.10, 0.04, 0.16);
    }
    case VIEW_TRACE_DEPTH: {
      let t = textureLoad(trace_tex, coords, 0).g;
      color = select(BACKGROUND, vec3<f32>(1.0 - saturate(t / TRACE_DEPTH_RANGE)), t > 0.0);
    }
    case VIEW_TRACE_DIFF: {
      let trace = textureLoad(trace_tex, coords, 0).rg;
      let analytic_hit = trace.r > 0.0;
      let prepass_hit = trace.g > 0.0;

      if (analytic_hit && prepass_hit) {
        color = vec3<f32>(saturate(abs(trace.r - trace.g) * TRACE_DIFF_SCALE));
      } else if (analytic_hit != prepass_hit) {
        color = vec3<f32>(1.0, 0.0, 1.0);
      } else {
        color = BACKGROUND;
      }
    }
    default: {}
  }
  
  return vec4<f32>(color, 1.0);
}

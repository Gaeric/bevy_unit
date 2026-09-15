#import bevy_render::view::{position_ndc_to_world, uv_to_ndc};

const WORKGROUP_SIZE: u32 = 8u;
const NO_HIT: f32 = -1.0;
const FAR_T: f32 = 1e30;

const SPHERE_CENTER: vec3<f32> = vec3<f32>(0.0, 0.0, -1.0);
const SPHERE_RADIUS: f32 = 0.5;
const GROUND_CENTER: vec3<f32> = vec3<f32>(0.0, -100.5, -1.0);
const GROUND_RADIUS: f32 = 100.0;

struct ShineCamera {
  world_from_clip: mat4x4<f32>,
  camera_position: vec4<f32>,
  viewport: vec4<u32>,
};

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>,
}

@group(0) @binding(0) var<uniform> camera: ShineCamera;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var trace_tex: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {

  if (gid.x >= camera.viewport.z || gid.y >= camera.viewport.w) {
    return;
  }

  let ndc = pixel_to_ndc(gid.xy);
  let pixel = vec2<i32>(camera.viewport.xy + gid.xy);

  let analytic_t = intersect_scene(primary_ray(ndc));
  let prepass_t = depth_to_ray_t(ndc, textureLoad(depth_tex, pixel, 0));

  textureStore(trace_tex, pixel, vec4<f32>(analytic_t, prepass_t, 0.0, 1.0));
}

/// `+ 0.5` is the pixel center. bevy's `frag_coord_to_ndc` cannot be used here: its
fn pixel_to_ndc(pixel: vec2<u32>) -> vec2<f32> {
  let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(camera.viewport.zw);
  return uv_to_ndc(uv);
}

fn primary_ray(ndc: vec2<f32>) -> Ray {
  let origin = camera.camera_position.xyz;
  let near_point = position_ndc_to_world(vec3<f32>(ndc, 1.0),
camera.world_from_clip);
  return Ray(origin, normalize(near_point - origin));
}

fn depth_to_ray_t(ndc: vec2<f32>, depth: f32) -> f32 {
  if (depth <= 0.0) {
    return NO_HIT;
  }

  let hit = position_ndc_to_world(vec3<f32>(ndc, depth),
camera.world_from_clip);
  return distance(camera.camera_position.xyz, hit);
}

fn intersect_scene(ray: Ray) -> f32 {
  var t = FAR_T;
  t = min(t, intersect_sphere(ray, SPHERE_CENTER, SPHERE_RADIUS));
  t = min(t, intersect_sphere(ray, GROUND_CENTER, GROUND_RADIUS));

  return select(t, NO_HIT, t >= FAR_T);
}

fn intersect_sphere(ray: Ray, center: vec3<f32>, radius: f32) -> f32 {
  let oc = ray.origin - center;

  let h = dot(oc, ray.direction);

  let c = dot(oc, oc) - radius * radius;

  let disc = h * h - c;

  if (disc < 0.0) {
    return FAR_T;
  }

  let sqrt_disc = sqrt(disc);
  let near_t = -h - sqrt_disc;
  if (near_t > 0.0) {
    return near_t;
  }

  let far_t = -h + sqrt_disc;
  return select(FAR_T, far_t, far_t > 0.0);
}

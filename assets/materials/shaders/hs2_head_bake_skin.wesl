// Bakes the HS2 skin node graph into the three textures a plain
// `StandardMaterial` consumes: base color, tangent-space normal and
// metallic-roughness.
//
// Source graph (Blender):
//   Head texture shader / Torso Texture Shader -> Skin shader -> Principled BSDF
//
// The Blender-side helpers are intentionally not ported:
//   * `Skin effect`      - Voronoi/Noise "alternate skin" pores, off by default
//                          and not part of the game material.
//   * `Bump Adjust`      - only exists to recenter the MeshExporter normal maps;
//                          the same thing is done here with `bump_gamma`.
//   * `Bump` (height)    - needs screen-space derivatives, which a texture-space
//                          compute bake cannot provide.
//
// The bake is evaluated in texture space: one invocation per output texel, and
// the UV of that texel is exactly the pixel coordinate. This matches the
// reference renders produced by `tools/bake_reference.py`.

struct SkinBakeParams {
  // xy = scale, zw = offset applied to the mesh UV before sampling.
  uv_transform: vec4<f32>,
  // xy = repeat count of the detail gloss map (Blender: (24, 12) * scale).
  detail_uv: vec4<f32>,
  // x = rotation in degrees, y = scale, z = arch strength,
  // w = trailing offset applied after the eyebrow transform (Blender: 2.0).
  eyebrow: vec4<f32>,
  // xy = eyebrow translation.
  eyebrow_offset: vec4<f32>,
  // Eyebrow overlay color (the "hair color" the importer writes into `RGB`).
  eyebrow_color: vec4<f32>,
  // x = textured gloss, y = detail gloss delta, z = base gloss,
  // w = clear coat input.
  gloss: vec4<f32>,
  // x = bump scale, y = bump scale 2, z = normal strength, w = unused.
  bump: vec4<f32>,
  // Per-channel recentering of the normal maps, estimated from their medians
  // in `importer.py::estimate_bump_gamma` (R ~= 1.0, G ~= 1.59..1.62).
  bump_gamma: vec4<f32>,
  // rgb = subsurface color, a = subsurface/MainTex mix factor.
  subsurface: vec4<f32>,
  // x = bitfield, yzw reserved.
  flags: vec4<u32>,
};

const FLAG_EYEBROW: u32 = 1u;
const FLAG_MIRROR_EYEBROW: u32 = 2u;
const FLAG_COAT_FROM_DETAIL: u32 = 4u;
const FLAG_SUBSURFACE: u32 = 8u;

@group(0) @binding(0) var main_texture: texture_2d<f32>;
@group(0) @binding(1) var main_sampler: sampler;
@group(0) @binding(2) var detail_texture: texture_2d<f32>;
@group(0) @binding(3) var detail_sampler: sampler;
@group(0) @binding(4) var detail_gloss_texture: texture_2d<f32>;
@group(0) @binding(5) var detail_gloss_sampler: sampler;
@group(0) @binding(6) var eyebrow_texture: texture_2d<f32>;
@group(0) @binding(7) var eyebrow_sampler: sampler;
@group(0) @binding(8) var bump_texture: texture_2d<f32>;
@group(0) @binding(9) var bump_sampler: sampler;
@group(0) @binding(10) var bump2_texture: texture_2d<f32>;
@group(0) @binding(11) var bump2_sampler: sampler;
@group(0) @binding(12) var<uniform> params: SkinBakeParams;

@group(0) @binding(13) var out_base_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(14) var out_normal: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(15) var out_metallic_roughness: texture_storage_2d<rgba8unorm, write>;

fn sample_level(tex: texture_2d<f32>, smp: sampler, uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(tex, smp, uv, 0.0);
}

// NodeGroup.008 -> NodeGroup.006 in the prefab:
//   uv = max(uv, 1 - uv)                     X mirror, one eyebrow sheet for both sides
//   uv = rotate(uv, center = (0.65, 0.6), radians(degrees))
//   uv = uv * scale
//   uv += (offset.x, offset.y + (uv.x - 0.6)^2 * arch)
//   uv -= (offset, offset)
fn eyebrow_uv(uv: vec2<f32>) -> vec2<f32> {
  var q = uv;
  if ((params.flags.x & FLAG_MIRROR_EYEBROW) != 0u) {
    q.x = max(q.x, 1.0 - q.x);
  }

  let center = vec2<f32>(0.65, 0.6);
  let angle = params.eyebrow.x * 3.14159265 / 180.0;
  let d = q - center;
  q = vec2<f32>(d.x * cos(angle) - d.y * sin(angle), d.x * sin(angle) + d.y * cos(angle)) + center;

  q *= params.eyebrow.y;

  let arch = (q.x - 0.6) * (q.x - 0.6) * params.eyebrow.z;
  q += vec2<f32>(params.eyebrow_offset.x, params.eyebrow_offset.y + arch);

  return q - vec2<f32>(params.eyebrow.w);
}

// Decode a MeshExporter *_converted normal map.
// The blue channel is always 1.0 in these files, so the z component is fixed
// at 1 and only R/G carry the perturbation.
fn decode_normal(texel: vec4<f32>, gamma: vec2<f32>, scale: f32) -> vec2<f32> {
  let recentered = pow(max(texel.rg, vec2<f32>(0.0)), gamma);
  return (recentered - 0.5) * 2.0 * scale;
}

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
  let size = vec2<u32>(textureDimensions(out_base_color));
  if (any(invocation_id.xy >= size)) {
    return;
  }

  let uv = (vec2<f32>(invocation_id.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
  let suv = uv * params.uv_transform.xy + params.uv_transform.zw;

  // --- base color -------------------------------------------------------
  var color = sample_level(main_texture, main_sampler, suv);

  // Subsurface mixer: mix(MainTex, Subsurface Color, factor).
  if ((params.flags.x & FLAG_SUBSURFACE) != 0u) {
    color = vec4<f32>(mix(color.rgb, params.subsurface.rgb, params.subsurface.a), color.a);
  }

  var eyebrow_amount = 0.0;
  if ((params.flags.x & FLAG_EYEBROW) != 0u) {
    let eyebrow = sample_level(eyebrow_texture, eyebrow_sampler, eyebrow_uv(suv));
    eyebrow_amount = eyebrow.b;
    // Skin shader: Mix(factor = eyebrow.B, A = skin, B = eyebrow color).
    color = vec4<f32>(mix(color.rgb, params.eyebrow_color.rgb, eyebrow_amount), color.a);
  }

  // --- gloss / roughness ------------------------------------------------
  let detail_gloss = sample_level(
    detail_gloss_texture,
    detail_gloss_sampler,
    suv * params.detail_uv.xy,
  );

  var textured_gloss = params.gloss.x;
  if ((params.flags.x & FLAG_EYEBROW) != 0u) {
    // Head texture shader: textured gloss = (1 - eyebrow.B) * input.
    textured_gloss *= 1.0 - eyebrow_amount;
  }

  // Gloss group with "alternate skin" disabled:
  //   gloss = base + textured * 2 * (detail_gloss.a - delta)
  let gloss = params.gloss.z + textured_gloss * 2.0 * (detail_gloss.a - params.gloss.y);

  var clear_coat = params.gloss.w;
  if ((params.flags.x & FLAG_COAT_FROM_DETAIL) != 0u) {
    clear_coat += sample_level(detail_texture, detail_sampler, suv).b;
  }

  // Skin shader: roughness = (0.5 - 0.3 * gloss) * (1 - clear coat).
  let roughness = clamp((0.5 - 0.3 * gloss) * (1.0 - clear_coat), 0.04, 1.0);

  // --- normal -----------------------------------------------------------
  let normal_a = decode_normal(
    sample_level(bump_texture, bump_sampler, suv),
    params.bump_gamma.xy,
    params.bump.x,
  );
  let normal_b = decode_normal(
    sample_level(bump2_texture, bump2_sampler, suv),
    params.bump_gamma.zw,
    params.bump.y,
  );

  var normal = normalize(vec3<f32>(normal_a + normal_b, 1.0));
  normal = normalize(vec3<f32>(normal.xy * params.bump.z, normal.z));

  // --- store ------------------------------------------------------------
  let pixel = vec2<i32>(invocation_id.xy);
  textureStore(out_base_color, pixel, vec4<f32>(color.rgb, 1.0));
  textureStore(out_normal, pixel, vec4<f32>(normal * 0.5 + 0.5, 1.0));
  textureStore(out_metallic_roughness, pixel, vec4<f32>(1.0, roughness, 0.0, 1.0));
}

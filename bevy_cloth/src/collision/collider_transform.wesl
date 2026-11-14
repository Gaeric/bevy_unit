// Local points have w = 1, so the affine world transform includes translation.
struct TransformParams {
    world_from_local: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> params: TransformParams;
@group(0) @binding(1) var<storage, read> local_positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;

@compute @workgroup_size(64)
fn transform_vertices(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= arrayLength(&local_positions) {
        return;
    }

    positions[i] = params.world_from_local * local_positions[i];
}

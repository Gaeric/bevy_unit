struct SkinVertex {
    position: vec4<f32>,
    joints: vec4<u32>,
    weights: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> vertices: array<SkinVertex>;
@group(0) @binding(1) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read_write> skinned_positions: array<vec4<f32>>;

// Keep the workgroup size in sync with WORKGROUP_SIZE in compute_skinning.rs.
@compute @workgroup_size(64)
fn skin(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let i = invocation.x;
    if i >= arrayLength(&vertices) {
        return;
    }

    let vertex = vertices[i];
    let joints = vertex.joints;
    let weights = vertex.weights;
    let world_from_bind = weights.x * joint_matrices[joints.x]
        + weights.y * joint_matrices[joints.y]
        + weights.z * joint_matrices[joints.z]
        + weights.w * joint_matrices[joints.w];
    let world_position = world_from_bind * vertex.position;
    skinned_positions[i] = vec4<f32>(world_position.xyz, 1.0);
}

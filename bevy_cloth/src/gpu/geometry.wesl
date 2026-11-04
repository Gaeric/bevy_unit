// Uniform ABI: matrices 0/64, wave 128, pivot 144, count/padding 160..176.
struct PositionParams {
    current_model: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    // amplitude, current angular phase, previous angular phase, unused
    wave: vec4<f32>,
    pivot: vec4<f32>,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: PositionParams;
@group(0) @binding(1) var<storage, read> rest: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> current: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> previous: array<vec4<f32>>;

@compute @workgroup_size(256)
fn update_positions(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.count {
        return;
    }
    let initial = rest[index];
    let point = vec4<f32>(initial.xyz, 1.0);
    let phase = initial.x - params.pivot.x;
    let base = sin(phase);
    let current_wave = params.wave.x * (sin(params.wave.y + phase) - base);
    let previous_wave = params.wave.x * (sin(params.wave.z + phase) - base);
    let current_point = (params.current_model * point).xyz;
    let previous_point = (params.previous_model * point).xyz;
    current[index] = vec4<f32>(current_point + vec3<f32>(0.0, 0.0, current_wave), initial.w);
    previous[index] = vec4<f32>(previous_point + vec3<f32>(0.0, 0.0, previous_wave), initial.w);
}

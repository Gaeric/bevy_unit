const POSITIONS: array<vec3<f32>, 6> =
    array<vec3<f32>, 6>(
    vec3<f32>(-1.0,  1.0, 0.0),
    vec3<f32>(-1.0, -1.0, 0.0),
    vec3<f32>( 1.0,  1.0, 0.0),
    vec3<f32>( 1.0,  1.0, 0.0),
    vec3<f32>(-1.0, -1.0, 0.0),
    vec3<f32>( 1.0, -1.0, 0.0),
);

struct Prop {
 width: u32,
 height: u32,
 pad_a: u32,
 pad_b: u32,
};

@group(0) @binding(0)
var<uniform> property: array<Prop, 1u>;

const WIDTH: u32 = 800u;
const HEIGHT: u32 = 600u;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(POSITIONS[in_vertex_index], 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.clip_position.xy / vec2<f32>(f32(property[0].width - 1u), f32(property[0].height - 1u));
    return vec4<f32>(color, 0.0, 1.0);
}

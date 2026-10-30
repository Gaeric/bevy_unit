import bevy_pbr::render::{
    forward_io::{Vertex, VertexOutput, decompress_vertex},
    mesh_functions,
    view_transformations::position_world_to_clip,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> color: vec4<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<uniform> clip_bias: f32;

// POSITION-only lines, no skinning/morphs. Keep legacy lines at zero bias.
// wgpu disallows fixed-function depth bias on LineList; bias clip Z instead.
@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    let local = decompress_vertex(vertex_in, vertex_in.instance_index);
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    var out: VertexOutput;
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local, vec4<f32>(local.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    // Reverse-Z: small positive depth shift, preserving depth-tested occlusion.
    if clip_bias > 0.0 && out.position.w > 0.0
        && out.position.z >= 0.0 && out.position.z <= out.position.w {
        out.position.z = min(out.position.w, out.position.z + clip_bias * out.position.w);
    }
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex_in.instance_index;
    return out;
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    return color;
}

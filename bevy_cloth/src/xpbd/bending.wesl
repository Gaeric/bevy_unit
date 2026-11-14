// Dihedral data and resource interface. Solver entry is added in the next batch.

import super::layout::{Particle, StepParams, GroupParams};

// Matches the 32-byte Bend ABI in layout.rs.
struct Bend {
    particles: vec4<u32>,
    rest_angle: f32,
    compliance: f32,
    padding: vec2<u32>,
};

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(2)
var<uniform> group_params: GroupParams;

@group(0) @binding(3)
var<storage, read> bend: array<Bend>;

@group(0) @binding(4)
var<storage, read_write> bend_lambdas: array<f32>;

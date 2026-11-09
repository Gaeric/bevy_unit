// GPU data layouts and resource interfaces; no solver entry points yet.

struct Particle {
    position: vec3<f32>,
    inv_mass: f32,
};

struct Stretch {
    particles: vec2<u32>,
    rest_length: f32,
    compliance: f32,
};

struct StepParams {
    gravity: vec4<f32>,
    dt: f32,
    particle_count: u32,
    stretch_count: u32,
    attachment_count: u32,
};

struct GroupParams {
    start: u32,
    count: u32,
    padding: vec2<u32>,
};

// Shared by particle and stretch stages.

@group(0) @binding(0)
var<uniform> step_params: StepParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

// Particle stages: prediction and velocity update.

@group(0) @binding(2)
var<storage, read_write> velocities: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> previous_positions: array<vec4<f32>>;

// Stretch stage: one color range per dispatch.

@group(0) @binding(4)
var<uniform> group_params: GroupParams;

@group(0) @binding(5)
var<storage, read> stretch: array<Stretch>;

@group(0) @binding(6)
var<storage, read_write> stretch_lambdas: array<f32>;

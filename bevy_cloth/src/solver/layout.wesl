// GPU data layouts.

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
    collision_margin: f32,
    double_sided: u32,
    padding: vec2<u32>,
};

struct GroupParams {
    start: u32,
    count: u32,
    padding: vec2<u32>,
};

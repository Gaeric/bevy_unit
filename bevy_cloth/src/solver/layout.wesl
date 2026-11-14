// GPU data layouts.

struct Particle {
    position: vec3<f32>,
    inv_mass: f32,
};

struct StretchConstraint {
    particles: vec2<u32>,
    rest_length: f32,
    compliance: f32,
};

struct StepParams {
    gravity: vec4<f32>,
    substep_dt: f32,
    particle_count: u32,
    stretch_count: u32,
    attachment_count: u32,
    collision_margin: f32,
    double_sided: u32,
    padding: vec2<u32>,
};

struct ConstraintRangeParams {
    start: u32,
    count: u32,
    padding: vec2<u32>,
};

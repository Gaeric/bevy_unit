import super::layout::Particle;

struct AttachmentTarget {
    position: vec3<f32>,
    particle: u32,
};

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<storage, read> targets: array<AttachmentTarget>;

@compute @workgroup_size(64)
fn apply_attachment_targets(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= arrayLength(&targets) {
        return;
    }

    // Preparation validates unique destinations and sets their inverse mass to zero.
    let pin = targets[invocation.x];
    particles[pin.particle].position = pin.position;
}

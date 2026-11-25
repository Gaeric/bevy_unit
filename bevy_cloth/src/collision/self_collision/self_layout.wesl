// Shared hash/query ABI. All world-space distances use the prepared cloth's units.
struct SelfContactParams {
    particle_count: u32,
    bucket_count: u32,
    capacity: u32,
    diameter: f32,
    cell_width: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// For pairs at or inside the diameter, scaling first prevents a squared-length
// overflow/underflow. Values outside the sphere need not return an exact ratio.
fn distance_ratio_squared(delta: vec3<f32>, diameter: f32) -> f32 {
    if !finite3(delta) || largest_component(delta) > diameter {
        return 2.0;
    }
    let scaled = delta / diameter;
    return dot(scaled, scaled);
}

// Test the bits so the check does not depend on backend floating-point fast-math flags.
fn finite3(value: vec3<f32>) -> bool {
    return all((bitcast<vec3<u32>>(value) & vec3<u32>(0x7f800000u)) != vec3<u32>(0x7f800000u));
}

fn largest_component(value: vec3<f32>) -> f32 {
    let magnitude = abs(value);
    return max(magnitude.x, max(magnitude.y, magnitude.z));
}

fn cell_bucket(cell: vec3<i32>, bucket_count: u32) -> u32 {
    let bits = bitcast<vec3<u32>>(cell);
    return ((bits.x * 92837111u) ^ (bits.y * 689287499u) ^ (bits.z * 283923481u)) % bucket_count;
}

@group(0) @binding(0) var<uniform> params: SelfContactParams;

// Direct global references keep atomic pointer arguments out of portable WGSL.
const SUBSTEPS: u32 = 0u;
const MAX_REQUIRED: u32 = 1u;
const OVERFLOW_ROWS: u32 = 2u;
const INVALID_PARTICLES: u32 = 3u;
@group(0) @binding(8) var<storage, read_write> stats: array<atomic<u32>, 4>;

// Cumulative telemetry must not wrap back to zero if readback is delayed.
fn bump_stat(index: u32) {
    var current = atomicLoad(&stats[index]);
    while current != 0xffffffffu {
        let update = atomicCompareExchangeWeak(&stats[index], current, current + 1u);
        if update.exchanged { return; }
        current = update.old_value;
    }
}

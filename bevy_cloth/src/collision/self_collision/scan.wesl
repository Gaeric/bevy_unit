// Exclusive counting-hash scan: scan blocks upward, then add block offsets downward.
@group(0) @binding(0) var<storage, read> scan_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> scan_output: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
var<workgroup> prefix: array<u32, 64>;

@compute @workgroup_size(64)
fn scan_blocks(
    @builtin(global_invocation_id) invocation: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) block: vec3<u32>,
) {
    // Atomic bucket counts are viewed as ordinary u32s in this later dispatch.
    // Tail lanes contribute zero and participate in every barrier.
    let index = invocation.x;
    var value = 0u;
    if index < arrayLength(&scan_input) { value = scan_input[index]; }
    prefix[lane] = value;
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride *= 2u) {
        var sum = prefix[lane];
        if lane >= stride { sum += prefix[lane - stride]; }
        workgroupBarrier();
        prefix[lane] = sum;
        workgroupBarrier();
    }
    if index < arrayLength(&scan_output) { scan_output[index] = prefix[lane] - value; }
    if lane == 63u { block_sums[block.x] = prefix[lane]; }
}

@compute @workgroup_size(64)
fn add_block_offsets(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index < arrayLength(&scan_output) { scan_output[index] += scan_input[index / 64u]; }
}

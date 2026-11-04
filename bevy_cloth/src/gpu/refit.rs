//! Fixed-topology GPU refit. Parent registers refit.wesl; caller owns mutable nodes.
use crate::cloth::bvh::TriangleBvh;
use bevy::{
    asset::load_embedded_asset,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::{
        render_resource::{
            BindGroupEntries, BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages,
            CommandEncoderDescriptor, ComputePassDescriptor, PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};

const WORKGROUP_SIZE: u32 = 256;
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RefitParams {
    first: u32,
    count: u32,
    node_count: u32,
    triangle_count: u32,
}

fn supported(l: &wgpu::Limits) -> bool {
    l.max_storage_buffers_per_shader_stage >= 5
        && l.max_uniform_buffers_per_shader_stage >= 1
        && l.max_bind_groups >= 1
        && l.max_bindings_per_bind_group >= 6
        && l.max_compute_invocations_per_workgroup >= WORKGROUP_SIZE
        && l.max_compute_workgroup_size_x >= WORKGROUP_SIZE
        && l.max_uniform_buffer_binding_size >= 16
        && l.max_buffer_size >= 16
}
fn storage_fits(bytes: u64, l: &wgpu::Limits) -> bool {
    bytes > 0 && bytes <= l.max_buffer_size && bytes <= u64::from(l.max_storage_buffer_binding_size)
}

// The CPU adapter guarantees topology, unique IDs, reference permutation and exact depths.
// Check sizes/ranges here, not a second BVH builder or a per-tick CPU traversal.
fn schedule(tree: &TriangleBvh, l: &wgpu::Limits) -> Result<Vec<RefitParams>, String> {
    let fail = || "BVH refit exceeds device limits or has inconsistent dispatch ranges".to_owned();
    let n = u32::try_from(tree.nodes.len()).map_err(|_| fail())?;
    let t = u32::try_from(tree.triangle_refs.len()).map_err(|_| fail())?;
    let leaves = u32::try_from(tree.leaf_nodes.len()).map_err(|_| fail())?;
    let internal = u32::try_from(tree.internal_nodes.len()).map_err(|_| fail())?;
    if !supported(l)
        || n == 0
        || t == 0
        || leaves == 0
        || leaves.checked_add(internal) != Some(n)
        || !storage_fits(u64::from(n) * 32, l)
        || !storage_fits(u64::from(t) * 12, l)
    {
        return Err(fail());
    }
    let mut ranges = vec![RefitParams {
        first: 0,
        count: leaves,
        node_count: n,
        triangle_count: t,
    }];
    let mut next = 0u32;
    let mut depth = u32::MAX;
    for layer in &tree.refit_layers {
        if layer.first != next || layer.count == 0 || layer.depth >= depth {
            return Err(fail());
        }
        next = next.checked_add(layer.count).ok_or_else(fail)?;
        if next > internal {
            return Err(fail());
        }
        ranges.push(RefitParams {
            first: leaves + layer.first,
            count: layer.count,
            node_count: n,
            triangle_count: t,
        });
        depth = layer.depth;
    }
    if next != internal
        || ranges
            .iter()
            .any(|r| r.count.div_ceil(WORKGROUP_SIZE) > l.max_compute_workgroups_per_dimension)
    {
        return Err(fail());
    }
    Ok(ranges)
}

#[derive(Resource)]
pub(super) struct RefitPipeline {
    layout: BindGroupLayoutDescriptor,
    pipelines: Option<[CachedComputePipelineId; 2]>,
}
impl FromWorld for RefitPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "refit.wesl");
        let layout = BindGroupLayoutDescriptor::new(
            "BVH refit",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        );
        let pipelines = supported(&world.resource::<RenderDevice>().limits()).then(|| {
            ["refit_leaves", "refit_internal"].map(|entry| {
                world.resource::<PipelineCache>().queue_compute_pipeline(
                    ComputePipelineDescriptor {
                        label: Some(entry.into()),
                        layout: vec![layout.clone()],
                        shader: shader.clone(),
                        entry_point: Some(entry.into()),
                        ..default()
                    },
                )
            })
        });
        Self { layout, pipelines }
    }
}
impl RefitPipeline {
    pub(super) fn ready(&self, cache: &PipelineCache) -> bool {
        self.pipelines.is_some_and(|ids| {
            ids.iter()
                .all(|&id| cache.get_compute_pipeline(id).is_some())
        })
    }
}

/// Immutable work IDs and one immutable uniform per dispatch (never queue-write a shared uniform).
/// Nodes, positions (vec4 stride), indices and references must match the original CPU topology.
/// Submit after positions and before queries on the same queue. Invalid vertex data is unsupported:
/// guards preserve previous bounds, not fabricated bounds; true means submitted, not GPU validation.
/// No original node copy is needed here: the caller retains its initialized STORAGE node buffer.
pub(super) struct GpuRefit {
    work_ids: Buffer,
    layers: Vec<(Buffer, RefitParams)>,
}
impl GpuRefit {
    pub(super) fn new(device: &RenderDevice, tree: &TriangleBvh) -> Result<Self, String> {
        let ranges = schedule(tree, &device.limits())?;
        let ids: Vec<_> = tree
            .leaf_nodes
            .iter()
            .chain(&tree.internal_nodes)
            .copied()
            .collect();
        let work_ids = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("BVH refit work IDs"),
            contents: bytemuck::cast_slice(&ids),
            usage: BufferUsages::STORAGE,
        });
        let layers = ranges
            .into_iter()
            .map(|range| {
                let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("BVH refit immutable range"),
                    contents: bytemuck::bytes_of(&range),
                    usage: BufferUsages::UNIFORM,
                });
                (uniform, range)
            })
            .collect();
        Ok(Self { work_ids, layers })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_or_submit(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &RefitPipeline,
        nodes: &Buffer,
        positions: &Buffer,
        indices: &Buffer,
        refs: &Buffer,
    ) -> bool {
        let Some(ids) = pipeline.pipelines else {
            return false;
        };
        let (Some(leaves), Some(internal)) = (
            cache.get_compute_pipeline(ids[0]),
            cache.get_compute_pipeline(ids[1]),
        ) else {
            return false;
        };
        let l = device.limits();
        let p = self.layers[0].1;
        if nodes.size() != u64::from(p.node_count) * 32
            || positions.size() % 16 != 0
            || positions.size() / 16 > u64::from(u32::MAX)
            || indices.size() != u64::from(p.triangle_count) * 12
            || refs.size() != u64::from(p.triangle_count) * 4
            || [nodes, positions, indices, refs]
                .iter()
                .any(|b| !storage_fits(b.size(), &l) || !b.usage().contains(BufferUsages::STORAGE))
        {
            return false;
        }
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BVH refit"),
        });
        for (i, (uniform, range)) in self.layers.iter().enumerate() {
            let group = device.create_bind_group(
                "BVH refit",
                &cache.get_bind_group_layout(&pipeline.layout),
                &BindGroupEntries::sequential((
                    uniform.as_entire_binding(),
                    nodes.as_entire_binding(),
                    positions.as_entire_binding(),
                    indices.as_entire_binding(),
                    refs.as_entire_binding(),
                    self.work_ids.as_entire_binding(),
                )),
            );
            // Ending each pass creates an explicit storage dependency / GPU barrier before parents.
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("BVH refit layer"),
                ..default()
            });
            pass.set_pipeline(if i == 0 { leaves } else { internal });
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(range.count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit([encoder.finish()]);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::bvh::{BvhNode, build_triangle_bvh};
    use std::mem::{align_of, offset_of};

    #[test]
    fn uniform_abi_and_limits() {
        assert_eq!(
            (size_of::<RefitParams>(), align_of::<RefitParams>()),
            (16, 16)
        );
        assert_eq!(
            [
                offset_of!(RefitParams, first),
                offset_of!(RefitParams, count),
                offset_of!(RefitParams, node_count),
                offset_of!(RefitParams, triangle_count)
            ],
            [0, 4, 8, 12]
        );
        let tree = build_triangle_bvh(&[Vec3::ZERO, Vec3::X, Vec3::Y], &[0, 1, 2]).unwrap();
        let l = wgpu::Limits::default();
        assert_eq!(schedule(&tree, &l).unwrap().len(), 1);
        for case in 0..9 {
            let mut small = l.clone();
            match case {
                0 => small.max_storage_buffers_per_shader_stage = 4,
                1 => small.max_bindings_per_bind_group = 5,
                2 => small.max_uniform_buffer_binding_size = 15,
                3 => small.max_compute_workgroup_size_x = 255,
                4 => small.max_compute_invocations_per_workgroup = 255,
                5 => small.max_compute_workgroups_per_dimension = 0,
                6 => small.max_buffer_size = 31,
                7 => small.max_storage_buffer_binding_size = 31,
                _ => small.max_uniform_buffers_per_shader_stage = 0,
            }
            assert!(schedule(&tree, &small).is_err());
        }
        assert!(storage_fits(32, &l));
        assert!(!storage_fits(0, &l));
        assert!(!storage_fits(
            u64::from(l.max_storage_buffer_binding_size) + 1,
            &l
        ));
        // Range-only fixtures: topology is trusted by schedule, not dereferenced here.
        let mut wide = tree.clone();
        wide.nodes.resize(256, tree.nodes[0]);
        wide.leaf_nodes.resize(256, 0);
        let mut one_group = l.clone();
        one_group.max_compute_workgroups_per_dimension = 1;
        assert!(schedule(&wide, &one_group).is_ok());
        wide.nodes.push(tree.nodes[0]);
        wide.leaf_nodes.push(0);
        assert!(schedule(&wide, &one_group).is_err());
        let mut bad = tree.clone();
        bad.leaf_nodes.clear();
        assert!(schedule(&bad, &l).is_err());
    }

    // Oracle recursively collects actual descendant vertices; never unions cached child bounds.
    fn oracle(id: usize, tree: &TriangleBvh, positions: &[Vec3], indices: &[u32]) -> (Vec3, Vec3) {
        fn vertices(id: usize, tree: &TriangleBvh, indices: &[u32], out: &mut Vec<u32>) {
            let n = tree.nodes[id];
            if n.is_leaf() {
                for &t in &tree.triangle_refs
                    [n.first_index as usize..(n.first_index + n.primitive_count) as usize]
                {
                    out.extend_from_slice(&indices[t as usize * 3..t as usize * 3 + 3]);
                }
            } else {
                vertices(n.first_index as usize, tree, indices, out);
                vertices(n.first_index as usize + 1, tree, indices, out);
            }
        }
        let mut ids = Vec::new();
        vertices(id, tree, indices, &mut ids);
        ids.into_iter().map(|i| positions[i as usize]).fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), p| (min.min(p), max.max(p)),
        )
    }

    #[test]
    fn layered_manual_refit_matches_independent_geometry_at_every_node() {
        let mut rest = Vec::new();
        let mut indices = Vec::new();
        for i in 0..73 {
            let p = Vec3::new(((i * 37) % 73) as f32 * 2.0, (i % 5) as f32, 0.0);
            let base = rest.len() as u32;
            rest.extend([p, p + Vec3::X, p + Vec3::Y]);
            indices.extend([base, base + 1, base + 2]);
        }
        let tree = build_triangle_bvh(&rest, &indices).unwrap();
        let ranges = schedule(&tree, &wgpu::Limits::default()).unwrap();
        assert!(ranges.len() > 2);
        let ids: Vec<_> = tree
            .leaf_nodes
            .iter()
            .chain(&tree.internal_nodes)
            .copied()
            .collect();
        for deform in [false, true] {
            let positions: Vec<_> = rest
                .iter()
                .map(|p| {
                    *p + Vec3::new(
                        3.0,
                        -2.0,
                        if deform { (p.x * 0.2).sin() * 5.0 } else { 7.0 },
                    )
                })
                .collect();
            let mut nodes: Vec<BvhNode> = tree.nodes.clone();
            let mut done = vec![false; nodes.len()];
            for (layer, range) in ranges.iter().enumerate() {
                let mut updates = Vec::new();
                for &id in &ids[range.first as usize..(range.first + range.count) as usize] {
                    let n = nodes[id as usize];
                    let (min, max) = if layer == 0 {
                        assert!(n.is_leaf());
                        let mut min = Vec3::splat(f32::INFINITY);
                        let mut max = Vec3::splat(f32::NEG_INFINITY);
                        for slot in n.first_index..n.first_index + n.primitive_count {
                            let triangle = tree.triangle_refs[slot as usize] as usize;
                            for corner in 0..3 {
                                let point = positions[indices[triangle * 3 + corner] as usize];
                                min = min.min(point);
                                max = max.max(point);
                            }
                        }
                        (min, max)
                    } else {
                        assert!(!n.is_leaf());
                        let c = n.first_index as usize;
                        assert!(
                            done[c] && done[c + 1],
                            "child must complete in an earlier pass"
                        );
                        (
                            Vec3::from_array(nodes[c].min).min(Vec3::from_array(nodes[c + 1].min)),
                            Vec3::from_array(nodes[c].max).max(Vec3::from_array(nodes[c + 1].max)),
                        )
                    };
                    assert_eq!((min, max), oracle(id as usize, &tree, &positions, &indices));
                    updates.push((id as usize, min.to_array(), max.to_array()));
                }
                for (id, min, max) in updates {
                    nodes[id].min = min;
                    nodes[id].max = max;
                    assert!(!done[id]);
                    done[id] = true;
                }
            }
            assert!(done.iter().all(|&b| b));
            assert_eq!(
                Vec3::from_array(nodes[0].min),
                oracle(0, &tree, &positions, &indices).0
            );
        }
        let mut bad = tree.clone();
        bad.refit_layers[0].first = 1;
        assert!(schedule(&bad, &wgpu::Limits::default()).is_err());
    }
}

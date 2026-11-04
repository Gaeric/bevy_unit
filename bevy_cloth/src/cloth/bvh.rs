//! Pure CPU BVH2 construction and an owned GPU-facing ABI; no upload or runtime policy.
//!
//! Coordinates are supplied by the caller (initially world/solver space). Triangle IDs
//! always refer to `indices.chunks_exact(3)`; vertices and triangles are never reordered.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use obvhs::{BvhBuildParams, bvh2::builder::build_bvh2_from_tris, triangle::Triangle};
use std::{fmt, time::Duration};

/// Independent of obvhs' feature-dependent node layout. WGSL storage layout:
/// `min: vec3<f32>, first_index: u32, max: vec3<f32>, primitive_count: u32`.
/// A zero count denotes an internal node with children `first_index` and
/// `first_index + 1`; otherwise it denotes a contiguous range in triangle_refs.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct BvhNode {
    pub min: [f32; 3],
    pub first_index: u32,
    pub max: [f32; 3],
    pub primitive_count: u32,
}

impl BvhNode {
    pub fn is_leaf(&self) -> bool {
        self.primitive_count != 0
    }
}

/// One dispatch range in `TriangleBvh::internal_nodes`, at an exact root-relative depth.
/// Layers are stored deepest first; finish each dispatch before starting the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BvhRefitLayer {
    pub depth: u32,
    pub first: u32,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub struct TriangleBvh {
    /// Root is node 0. All nodes are reachable; there are no vacant slots.
    pub nodes: Vec<BvhNode>,
    /// Each original triangle ID occurs exactly once (no spatial splits).
    pub triangle_refs: Vec<u32>,
    /// Node IDs for the initial leaf-AABB refit dispatch.
    pub leaf_nodes: Vec<u32>,
    /// Internal node IDs grouped by exact depth, not by node-array order.
    pub internal_nodes: Vec<u32>,
    /// Leaf refit first, then these ranges in their stored deepest-first order.
    pub refit_layers: Vec<BvhRefitLayer>,
    /// Exact longest root-to-leaf edge count; a single leaf has depth 0.
    /// This is NOT obvhs' `max_depth`, which is a traversal-stack capacity bound.
    pub max_depth: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BvhBuildError(pub String);

impl fmt::Display for BvhBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BvhBuildError {}

/// Build ordinary BVH2 using obvhs 0.3.3 PLOC, without spatial splitting.
///
/// Finite point/line/zero-area and duplicate triangles are retained: their bounds
/// are valid, and deciding contact normals for them belongs to the query stage.
/// Empty/malformed input, non-finite positions (including unused vertices), and
/// bounds whose center/extent/SAH arithmetic overflows f32 are rejected.
/// No welding, thickness expansion, triangle filtering, or geometry copies are
/// retained in the result. Caller must keep the matching positions and indices.
pub fn build_triangle_bvh(
    positions: &[Vec3],
    indices: &[u32],
) -> Result<TriangleBvh, BvhBuildError> {
    let fail = |message: &str| BvhBuildError(message.to_owned());
    if positions.is_empty() || indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(fail("expected nonempty indexed triangles"));
    }
    let triangle_count = indices.len() / 3;
    // PLOC first allocates the uncollapsed binary tree (2*n - 1 nodes).
    if positions.len() > u32::MAX as usize || triangle_count > u32::MAX as usize / 2 {
        return Err(fail("BVH exceeds u32 index capacity"));
    }
    if positions.iter().any(|position| !position.is_finite()) {
        return Err(fail("non-finite vertex position"));
    }
    let mut triangles = Vec::with_capacity(triangle_count);
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for (id, ids) in indices.chunks_exact(3).enumerate() {
        if ids.iter().any(|&index| index as usize >= positions.len()) {
            return Err(BvhBuildError(format!("triangle {id}: index out of range")));
        }
        let [a, b, c] = [
            positions[ids[0] as usize],
            positions[ids[1] as usize],
            positions[ids[2] as usize],
        ];
        min = min.min(a).min(b).min(c);
        max = max.max(a).max(b).max(c);
        triangles.push(Triangle {
            v0: a.into(),
            v1: b.into(),
            v2: c.into(),
        });
    }
    let extent = max - min;
    let half_area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
    if !(min + max).is_finite() || !extent.is_finite() || !half_area.is_finite() {
        return Err(fail("bounds overflow f32 BVH arithmetic"));
    }

    // A small fixed initial-build policy, not a measured quality/performance choice.
    // Explicitly disable pre-splitting even if a future preset changes its default.
    // No reinsertion pass: PLOC + leaf collapse suffices for this first adapter.
    let config = BvhBuildParams {
        pre_split: false,
        max_prims_per_leaf: 4,
        ..BvhBuildParams::fastest_build()
    };
    let mut core_build_time = Duration::ZERO;
    let bvh = build_bvh2_from_tris(&triangles, config, &mut core_build_time);
    if bvh.uses_spatial_splits || bvh.primitive_indices.len() != triangle_count {
        return Err(fail(
            "builder returned spatial splits or an invalid reference count",
        ));
    }
    // Copy fields, never cast the library's nodes: its ABI depends on features.
    let nodes: Vec<_> = bvh
        .nodes
        .iter()
        .map(|node| BvhNode {
            min: node.aabb().min.to_array(),
            first_index: node.first_index,
            max: node.aabb().max.to_array(),
            primitive_count: node.prim_count,
        })
        .collect();
    if nodes.is_empty() {
        return Err(fail("builder returned no root"));
    }

    let mut leaf_nodes = Vec::new();
    let mut by_depth: Vec<Vec<u32>> = Vec::new();
    let mut max_depth = 0;
    let mut visited = vec![false; nodes.len()];
    let mut used_slots = vec![false; triangle_count];
    let mut used_triangles = vec![false; triangle_count];
    let mut stack = vec![(0u32, 0u32)];
    while let Some((id, depth)) = stack.pop() {
        let Some(node) = nodes.get(id as usize) else {
            return Err(fail("builder returned an out-of-range child"));
        };
        if std::mem::replace(&mut visited[id as usize], true) {
            return Err(fail("builder returned a cycle or shared child"));
        }
        max_depth = max_depth.max(depth);
        if node.is_leaf() {
            leaf_nodes.push(id);
            let first = node.first_index as usize;
            let Some(end) = first.checked_add(node.primitive_count as usize) else {
                return Err(fail("builder returned an overflowing leaf range"));
            };
            let Some(refs) = bvh.primitive_indices.get(first..end) else {
                return Err(fail("builder returned an out-of-range leaf"));
            };
            for (slot, &triangle) in (first..end).zip(refs) {
                let Some(used) = used_triangles.get_mut(triangle as usize) else {
                    return Err(fail("builder returned an invalid triangle ID"));
                };
                if std::mem::replace(used, true) || std::mem::replace(&mut used_slots[slot], true) {
                    return Err(fail("builder returned duplicate triangle references"));
                }
            }
        } else {
            if by_depth.len() <= depth as usize {
                by_depth.resize_with(depth as usize + 1, Vec::new);
            }
            by_depth[depth as usize].push(id);
            let Some(right) = node.first_index.checked_add(1) else {
                return Err(fail("builder returned an overflowing child index"));
            };
            stack.push((right, depth + 1));
            stack.push((node.first_index, depth + 1));
        }
    }
    if visited.contains(&false) || used_slots.contains(&false) || used_triangles.contains(&false) {
        return Err(fail(
            "builder returned unreachable nodes or missing triangle references",
        ));
    }
    let mut internal_nodes = Vec::new();
    let mut refit_layers = Vec::new();
    for (depth, ids) in by_depth.into_iter().enumerate().rev() {
        if ids.is_empty() {
            continue;
        }
        refit_layers.push(BvhRefitLayer {
            depth: depth as u32,
            first: internal_nodes.len() as u32,
            count: ids.len() as u32,
        });
        internal_nodes.extend(ids);
    }
    Ok(TriangleBvh {
        nodes,
        triangle_refs: bvh.primitive_indices,
        leaf_nodes,
        internal_nodes,
        refit_layers,
        max_depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    fn contains(node: &BvhNode, point: Vec3) -> bool {
        point.cmpge(Vec3::from_array(node.min)).all()
            && point.cmple(Vec3::from_array(node.max)).all()
    }

    /// Independent walk checks geometry, unique references and actual root-relative depths.
    fn assert_invariants(tree: &TriangleBvh, positions: &[Vec3], indices: &[u32]) {
        let mut visited = vec![false; tree.nodes.len()];
        let mut depths = vec![0; tree.nodes.len()];
        let mut refs = Vec::new();
        let mut leaves = Vec::new();
        let mut stack = vec![(0usize, 0u32)];
        while let Some((id, depth)) = stack.pop() {
            assert!(!std::mem::replace(&mut visited[id], true));
            depths[id] = depth;
            let node = &tree.nodes[id];
            assert!(Vec3::from_array(node.min).is_finite());
            assert!(Vec3::from_array(node.max).is_finite());
            assert!(
                Vec3::from_array(node.min)
                    .cmple(Vec3::from_array(node.max))
                    .all()
            );
            if node.is_leaf() {
                leaves.push(id as u32);
                let first = node.first_index as usize;
                let end = first + node.primitive_count as usize;
                for &triangle in &tree.triangle_refs[first..end] {
                    refs.push(triangle);
                    for &vertex in &indices[triangle as usize * 3..triangle as usize * 3 + 3] {
                        assert!(contains(node, positions[vertex as usize]));
                    }
                }
            } else {
                for child in [node.first_index, node.first_index + 1] {
                    let child_node = &tree.nodes[child as usize];
                    assert!(contains(node, Vec3::from_array(child_node.min)));
                    assert!(contains(node, Vec3::from_array(child_node.max)));
                    stack.push((child as usize, depth + 1));
                }
            }
        }
        assert!(visited.iter().all(|&value| value));
        assert_eq!(tree.max_depth, *depths.iter().max().unwrap());
        refs.sort_unstable();
        assert_eq!(refs, (0..(indices.len() / 3) as u32).collect::<Vec<_>>());
        leaves.sort_unstable();
        let mut actual_leaves = tree.leaf_nodes.clone();
        actual_leaves.sort_unstable();
        assert_eq!(leaves, actual_leaves);
        let mut covered = vec![false; tree.nodes.len()];
        let mut next = 0;
        let mut previous_depth = u32::MAX;
        for layer in &tree.refit_layers {
            assert!(layer.count > 0);
            assert!(layer.depth < previous_depth);
            assert_eq!(layer.first, next);
            previous_depth = layer.depth;
            next += layer.count;
            for &id in &tree.internal_nodes[layer.first as usize..next as usize] {
                assert!(!std::mem::replace(&mut covered[id as usize], true));
                let node = &tree.nodes[id as usize];
                assert!(!node.is_leaf());
                assert_eq!(depths[id as usize], layer.depth);
                for child in [node.first_index, node.first_index + 1] {
                    assert_eq!(depths[child as usize], layer.depth + 1);
                    assert!(tree.nodes[child as usize].is_leaf() || covered[child as usize]);
                }
            }
        }
        assert_eq!(next as usize, tree.internal_nodes.len());
        for (id, node) in tree.nodes.iter().enumerate() {
            assert_eq!(covered[id], !node.is_leaf());
        }
    }

    #[test]
    fn node_abi_is_owned_32_bytes() {
        assert_eq!(size_of::<BvhNode>(), 32);
        assert_eq!(align_of::<BvhNode>(), 16);
        assert_eq!(offset_of!(BvhNode, min), 0);
        assert_eq!(offset_of!(BvhNode, first_index), 12);
        assert_eq!(offset_of!(BvhNode, max), 16);
        assert_eq!(offset_of!(BvhNode, primitive_count), 28);
        let node = BvhNode {
            min: [1.0, 2.0, 3.0],
            first_index: 17,
            max: [4.0, 5.0, 6.0],
            primitive_count: 2,
        };
        let bytes = bytemuck::bytes_of(&node);
        assert_eq!(&bytes[12..16], &17u32.to_ne_bytes());
        assert_eq!(&bytes[28..32], &2u32.to_ne_bytes());
        assert_eq!(bytemuck::pod_read_unaligned::<BvhNode>(bytes), node);
    }

    #[test]
    fn one_triangle_has_root_leaf_and_zero_edge_depth() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let tree = build_triangle_bvh(&positions, &[0, 1, 2]).unwrap();
        assert_invariants(&tree, &positions, &[0, 1, 2]);
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.leaf_nodes, [0]);
        assert_eq!(tree.triangle_refs, [0]);
        assert_eq!(tree.max_depth, 0);
        assert!(tree.internal_nodes.is_empty());
        assert!(tree.refit_layers.is_empty());
    }

    #[test]
    fn references_bounds_and_deepest_first_layers() {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        // Nonuniform clusters and a flat axis exercise sorting and leaf collapse.
        for i in 0..73 {
            let x = ((i * 37) % 73) as f32;
            let base = positions.len() as u32;
            positions.extend([
                Vec3::new(x * x, (i % 5) as f32, 0.0),
                Vec3::new(x * x + 0.2, (i % 5) as f32, 0.0),
                Vec3::new(x * x, (i % 5) as f32 + 0.3, 0.0),
            ]);
            indices.extend([base, base + 1, base + 2]);
        }
        let tree = build_triangle_bvh(&positions, &indices).unwrap();
        assert_invariants(&tree, &positions, &indices);
        assert!(tree.max_depth > 1);
        assert!(tree.refit_layers.len() > 1);
    }

    #[test]
    fn finite_degenerate_and_duplicate_triangles_keep_original_ids() {
        for positions in [
            vec![Vec3::ZERO; 3],
            vec![Vec3::splat(7.0); 3],
            vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0],
        ] {
            let indices: Vec<_> = [0, 1, 2].into_iter().cycle().take(36).collect();
            let tree = build_triangle_bvh(&positions, &indices).unwrap();
            assert_invariants(&tree, &positions, &indices);
        }
        let tree = build_triangle_bvh(&[Vec3::ZERO], &[0, 0, 0]).unwrap();
        assert_invariants(&tree, &[Vec3::ZERO], &[0, 0, 0]);
    }

    #[test]
    fn invalid_input_is_rejected_before_builder() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(build_triangle_bvh(&[], &[0, 1, 2]).is_err());
        for indices in [vec![], vec![0, 1], vec![0, 1, 2, 0], vec![0, 1, 3]] {
            assert!(build_triangle_bvh(&positions, &indices).is_err());
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = positions.to_vec();
            invalid.push(Vec3::splat(bad)); // Unused vertices are checked too.
            assert!(build_triangle_bvh(&invalid, &[0, 1, 2]).is_err());
        }
        assert!(build_triangle_bvh(&[Vec3::splat(f32::MAX)], &[0, 0, 0]).is_err());
        let overflow = [Vec3::splat(-1e30), Vec3::splat(1e30), Vec3::ZERO];
        assert!(build_triangle_bvh(&overflow, &[0, 1, 2]).is_err());
    }
}

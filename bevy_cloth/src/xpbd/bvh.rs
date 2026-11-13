//! GPU-facing BVH2 node layout and CPU-owned tree data.
//!
//! Geometry is stored separately: world-space positions and original triangle
//! indices. A triangle ID selects indices[id * 3..id * 3 + 3].
//! CPU construction only; no GPU allocation, traversal or refit execution.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use obvhs::{BvhBuildParams, bvh2::builder::build_bvh2_from_tris, triangle::Triangle};
use std::time::Duration;

/// One GPU node, with a 32-byte array stride. Node IDs index the nodes array.
///
/// WGSL equivalent:
/// struct BvhNode {
///     min: vec3<f32>,
///     first_index: u32,
///     max: vec3<f32>,
///     primitive_count: u32,
/// }
///
/// Internal node: primitive_count == 0; children are nodes[first_index] and
/// nodes[first_index + 1]. Only siblings must be adjacent, not entire layers.
///
/// Leaf node: primitive_count > 0; triangle_refs[first_index..first_index +
/// primitive_count] contains its original triangle IDs, not vertex indices.
///
/// GPU refit updates only min/max; tree connections and leaf ranges stay fixed.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct BvhNode {
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

/// One CPU dispatch range in TriangleBvh::internal_nodes, not in nodes.
///
/// All entries in a range have the same root-relative depth. Process ranges
/// deepest first, after all leaves, so children finish before their parents.
/// This record is CPU scheduling metadata, not a GPU buffer payload.
#[derive(Clone, Copy, Debug)]
pub(super) struct BvhRefitLayer {
    pub depth: u32,
    pub first: u32,
    pub count: u32,
}

/// CPU-owned build result; upload its arrays separately, not this struct.
///
/// The tree does not own positions or triangle indices. Keep the matching
/// geometry separately; original vertex and triangle numbering is preserved.
#[derive(Clone, Debug)]
pub(super) struct TriangleBvh {
    /// All nodes; the root is nodes[0]. Upload as array<BvhNode>.
    pub nodes: Vec<BvhNode>,

    /// Original triangle IDs, contiguous per leaf; each occurs exactly once.
    /// Upload as array<u32>. Reordering references does not reorder geometry.
    pub triangle_refs: Vec<u32>,

    /// Node IDs for the leaf-refit stage. Upload as array<u32>.
    /// Leaves can all run in parallel, regardless of depth.
    pub leaf_nodes: Vec<u32>,

    /// Internal node IDs grouped by depth, deepest first. Upload as array<u32>.
    /// This is a processing list, not a reordered copy of nodes.
    pub internal_nodes: Vec<u32>,

    /// CPU ranges selecting one depth at a time from internal_nodes.
    pub refit_layers: Vec<BvhRefitLayer>,

    /// Longest root-to-leaf edge count; a root leaf has depth 0.
    /// CPU metadata, not a GPU array or the number of refit layers.
    pub max_depth: u32,
}

/// Validate geometry before calling the BVH builder.
fn validate_geometry(positions: &[Vec3], indices: &[u32]) -> Result<(), String> {
    if positions.is_empty() || indices.is_empty() {
        return Err("collider geometry must not be empty".into());
    }

    if !indices.len().is_multiple_of(3) {
        return Err("triangle index count must be a multiple of three".into());
    }

    let triangle_count = indices.len() / 3;

    // The initial binary tree can contain 2 * triangle_count - 1 nodes.
    if positions.len() > u32::MAX as usize || triangle_count > u32::MAX as usize / 2 {
        return Err("collider geometry exceeds u32 index capacity".into());
    }

    if positions.iter().any(|position| !position.is_finite()) {
        return Err("collider positions must be finite".into());
    }

    let mut lower = Vec3::splat(f32::INFINITY);
    let mut upper = Vec3::splat(f32::NEG_INFINITY);

    for &index in indices {
        let Some(&position) = positions.get(index as usize) else {
            return Err("triangle index is out of range".into());
        };

        lower = lower.min(position);
        upper = upper.max(position);
    }

    // Finite coordinates alone do not guarantee finite BVH arithmetic.
    let extent = upper - lower;
    let half_area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;

    if !(lower + upper).is_finite() || !extent.is_finite() || !half_area.is_finite() {
        return Err("collider bounds overflow f32 BVH arithmetic".into());
    }

    Ok(())
}

/// Build BVH2 without spatial splitting and prepare future GPU refit ranges.
///
/// Positions are world-space. Finite degenerate and duplicate triangles retain
/// their original IDs; deciding contact normals belongs to the query stage.
/// No geometry is reordered or retained in the returned tree.
pub(super) fn build_triangle_bvh(
    positions: &[Vec3],
    indices: &[u32],
) -> Result<TriangleBvh, String> {
    validate_geometry(positions, indices)?;

    let triangles: Vec<Triangle> = indices
        .chunks_exact(3)
        .map(|triangle| Triangle {
            v0: positions[triangle[0] as usize].into(),
            v1: positions[triangle[1] as usize].into(),
            v2: positions[triangle[2] as usize].into(),
        })
        .collect();

    // Initial policy, not a measured optimum. Keep one reference per triangle.
    let config = BvhBuildParams {
        pre_split: false,
        max_prims_per_leaf: 4,
        ..BvhBuildParams::fastest_build()
    };

    // Required by the library API; not GPU or end-to-end timing.
    let mut build_time = Duration::ZERO;
    let built = build_bvh2_from_tris(&triangles, config, &mut build_time);

    // Copy fields instead of casting obvhs nodes: its layout depends on features.
    let nodes: Vec<BvhNode> = built
        .nodes
        .iter()
        .map(|node| BvhNode {
            min: node.aabb().min.to_array(),
            first_index: node.first_index,
            max: node.aabb().max.to_array(),
            primitive_count: node.prim_count,
        })
        .collect();

    let mut leaf_nodes = Vec::new();
    let mut internal_by_depth: Vec<Vec<u32>> = Vec::new();
    let mut max_depth = 0;

    // Each entry contains a node ID and its root-relative depth.
    let mut stack = vec![(0_u32, 0_u32)];

    while let Some((node_id, depth)) = stack.pop() {
        let node = &nodes[node_id as usize];
        max_depth = max_depth.max(depth);

        if node.is_leaf() {
            leaf_nodes.push(node_id);
            continue;
        }

        if internal_by_depth.len() <= depth as usize {
            internal_by_depth.resize_with(depth as usize + 1, Vec::new);
        }

        internal_by_depth[depth as usize].push(node_id);

        let left = node.first_index;
        let right = left + 1;

        // Stack is LIFO: push right first so left is visited first.
        stack.push((right, depth + 1));
        stack.push((left, depth + 1));
    }

    let mut internal_nodes = Vec::new();
    let mut refit_layers = Vec::new();

    for (depth, node_ids) in internal_by_depth.into_iter().enumerate().rev() {
        if node_ids.is_empty() {
            continue;
        }

        refit_layers.push(BvhRefitLayer {
            depth: depth as u32,
            first: internal_nodes.len() as u32,
            count: node_ids.len() as u32,
        });

        internal_nodes.extend(node_ids);
    }

    Ok(TriangleBvh {
        nodes,
        triangle_refs: built.primitive_indices,
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

    /// Walk the exported tree independently, then check the refit schedule.
    fn assert_tree(tree: &TriangleBvh, positions: &[Vec3], indices: &[u32]) {
        let mut visited = vec![false; tree.nodes.len()];
        let mut depths = vec![0_u32; tree.nodes.len()];
        let mut references = Vec::new();
        let mut leaves = Vec::new();
        let mut used_slots = vec![false; tree.triangle_refs.len()];
        let mut stack = vec![(0_usize, 0_u32)];

        while let Some((id, depth)) = stack.pop() {
            assert!(!visited[id]);
            visited[id] = true;
            depths[id] = depth;

            let node = &tree.nodes[id];

            assert!(Vec3::from_array(node.min).is_finite());
            assert!(Vec3::from_array(node.max).is_finite());
            assert!(contains(node, Vec3::from_array(node.min)));
            assert!(contains(node, Vec3::from_array(node.max)));

            if node.is_leaf() {
                leaves.push(id as u32);

                let first = node.first_index as usize;
                let end = first + node.primitive_count as usize;

                for (slot, &triangle) in (first..end).zip(&tree.triangle_refs[first..end]) {
                    assert!(!used_slots[slot]);
                    used_slots[slot] = true;
                    references.push(triangle);

                    let base = triangle as usize * 3;
                    for &vertex in &indices[base..base + 3] {
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
        assert!(used_slots.iter().all(|&value| value));
        assert_eq!(tree.max_depth, *depths.iter().max().unwrap());

        references.sort_unstable();
        assert_eq!(
            references,
            (0..(indices.len() / 3) as u32).collect::<Vec<_>>()
        );

        leaves.sort_unstable();
        let mut listed_leaves = tree.leaf_nodes.clone();
        listed_leaves.sort_unstable();
        assert_eq!(leaves, listed_leaves);

        // Leaves are updated before any internal layer.
        let mut ready = vec![false; tree.nodes.len()];
        for &leaf in &tree.leaf_nodes {
            ready[leaf as usize] = true;
        }

        let mut next = 0;
        let mut previous_depth = u32::MAX;

        for layer in &tree.refit_layers {
            assert!(layer.count > 0);
            assert!(layer.depth < previous_depth);
            assert_eq!(layer.first, next);

            previous_depth = layer.depth;
            next += layer.count;

            let ids = &tree.internal_nodes[layer.first as usize..next as usize];

            // Validate the whole layer before marking it ready: no same-layer
            // parent/child dependency may rely on sequential CPU iteration.
            for &id in ids {
                let node = &tree.nodes[id as usize];

                assert!(!node.is_leaf());
                assert!(!ready[id as usize]);
                assert_eq!(depths[id as usize], layer.depth);
                assert!(ready[node.first_index as usize]);
                assert!(ready[node.first_index as usize + 1]);
            }

            for &id in ids {
                assert!(!ready[id as usize]);
                ready[id as usize] = true;
            }
        }

        assert_eq!(next as usize, tree.internal_nodes.len());
        assert!(ready.iter().all(|&value| value));
    }

    #[test]
    fn node_layout_matches_wgsl() {
        assert_eq!(size_of::<BvhNode>(), 32);
        assert_eq!(align_of::<BvhNode>(), 16);
        assert_eq!(offset_of!(BvhNode, min), 0);
        assert_eq!(offset_of!(BvhNode, first_index), 12);
        assert_eq!(offset_of!(BvhNode, max), 16);
        assert_eq!(offset_of!(BvhNode, primitive_count), 28);
    }

    #[test]
    fn single_triangle_is_a_root_leaf() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let indices = [0, 1, 2];

        let tree = build_triangle_bvh(&positions, &indices).unwrap();

        assert_tree(&tree, &positions, &indices);
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.leaf_nodes, [0]);
        assert_eq!(tree.triangle_refs, [0]);
        assert_eq!(tree.max_depth, 0);
        assert!(tree.internal_nodes.is_empty());
        assert!(tree.refit_layers.is_empty());
    }

    #[test]
    fn multiple_triangles_have_valid_bounds_and_refit_layers() {
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Nonuniform clusters and a flat axis exercise sorting and leaf collapse.
        for i in 0..73 {
            let x = ((i * 37) % 73) as f32;
            let base = positions.len() as u32;
            let origin = Vec3::new(x * x, (i % 5) as f32, 0.0);

            positions.extend([origin, origin + Vec3::X * 0.2, origin + Vec3::Y * 0.3]);
            indices.extend([base, base + 1, base + 2]);
        }

        let tree = build_triangle_bvh(&positions, &indices).unwrap();

        assert_tree(&tree, &positions, &indices);
        assert!(tree.max_depth > 1);
        assert!(tree.refit_layers.len() > 1);
    }

    #[test]
    fn degenerate_and_duplicate_triangles_keep_their_ids() {
        for positions in [
            vec![Vec3::ZERO; 3],
            vec![Vec3::splat(7.0); 3],
            vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0],
        ] {
            let indices = [0, 1, 2].repeat(8);
            let tree = build_triangle_bvh(&positions, &indices).unwrap();

            assert_tree(&tree, &positions, &indices);
        }

        let positions = [Vec3::ZERO];
        let indices = [0, 0, 0];
        let tree = build_triangle_bvh(&positions, &indices).unwrap();

        assert_tree(&tree, &positions, &indices);
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y];

        assert!(build_triangle_bvh(&[], &[0, 1, 2]).is_err());
        assert!(build_triangle_bvh(&positions, &[]).is_err());
        assert!(build_triangle_bvh(&positions, &[0, 1]).is_err());
        assert!(build_triangle_bvh(&positions, &[0, 1, 2, 0]).is_err());
        assert!(build_triangle_bvh(&positions, &[0, 1, 3]).is_err());

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = positions.to_vec();
            invalid.push(Vec3::splat(bad)); // Check unused vertices too.
            assert!(build_triangle_bvh(&invalid, &[0, 1, 2]).is_err());
        }

        let overflow = [Vec3::splat(-1e30), Vec3::splat(1e30), Vec3::ZERO];
        assert!(build_triangle_bvh(&overflow, &[0, 1, 2]).is_err());
        assert!(build_triangle_bvh(&[Vec3::splat(f32::MAX)], &[0, 0, 0]).is_err());
    }
}

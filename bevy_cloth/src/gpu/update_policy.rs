//! Forced BVH update modes for controlled experiments, not a measured strategy.
//! Automatic selection is disabled: no measured query-cost threshold is available.
//! Motion/deformation travel is not tree quality and is not used as a trigger.

use bevy::{
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
};
use std::num::NonZeroU64;

use crate::cloth::bvh::TriangleBvh;

/// Fixed-topology performance experiments only. Topology changes still require
/// preparation/reinitialization, independently of this policy. Manual rebuild
/// requests remain available in either mode. All ticks continue to GPU-refit.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, ExtractResource)]
#[extract_app(RenderApp)]
pub enum BvhUpdatePolicy {
    #[default]
    RefitOnly,
    /// Request a snapshot rebuild at nonzero completed tick multiples. Missed
    /// boundaries (multi-tick frames or busy rebuilds) are skipped, not queued.
    /// NonZeroU64 makes a zero interval unrepresentable.
    RebuildEvery { ticks: NonZeroU64 },
}

impl BvhUpdatePolicy {
    pub(crate) fn due(self, index: u64) -> bool {
        match self {
            Self::RefitOnly => false,
            Self::RebuildEvery { ticks } => index != 0 && index.is_multiple_of(ticks.get()),
        }
    }
}

/// Observed epoch/index, not display-frame count. Consume boundaries even while
/// busy so finishing a rebuild while paused cannot repeat the same request.
#[derive(Default)]
pub(crate) struct IntervalBoundary(Option<(u64, u64)>);

impl IntervalBoundary {
    pub(crate) fn observe(
        &mut self,
        policy: BvhUpdatePolicy,
        stamp: Option<(u64, u64)>,
        busy: bool,
    ) -> bool {
        let Some(stamp) = stamp else { return false };
        if self.0 == Some(stamp) {
            return false;
        }
        self.0 = Some(stamp);
        !busy && policy.due(stamp.1)
    }
}

/// CPU snapshot tree statistic: unit traversal/intersection-cost SAH proxy,
/// sum(area(node) * weight) / area(root), weight=1 internally and primitive
/// count at leaves. Not GPU traversal cost, overlap, or a measured trigger.
/// Degenerate root area has no normalized score; do not fabricate a value.
pub(crate) fn snapshot_sah(tree: &TriangleBvh) -> Option<f64> {
    fn area(node: &crate::cloth::bvh::BvhNode) -> Option<f64> {
        let extent: [f64; 3] =
            std::array::from_fn(|i| f64::from(node.max[i]) - f64::from(node.min[i]));
        if extent.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return None;
        }
        Some(2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0]))
    }
    let root = area(tree.nodes.first()?)?;
    if root <= 0.0 {
        return None;
    }
    let mut cost = 0.0;
    for node in &tree.nodes {
        cost += area(node)? * f64::from(node.primitive_count.max(1));
    }
    let score = cost / root;
    score.is_finite().then_some(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::bvh::{BvhNode, BvhRefitLayer};

    fn interval() -> BvhUpdatePolicy {
        BvhUpdatePolicy::RebuildEvery {
            ticks: NonZeroU64::new(3).unwrap(),
        }
    }

    #[test]
    fn default_refits_and_interval_requires_nonzero_completed_multiple() {
        assert_eq!(BvhUpdatePolicy::default(), BvhUpdatePolicy::RefitOnly);
        assert!(NonZeroU64::new(0).is_none());
        assert!(!interval().due(0));
        assert_eq!((0..10).filter(|&i| interval().due(i)).count(), 3);
        assert!(!(0..10).any(|i| BvhUpdatePolicy::default().due(i)));
    }

    #[test]
    fn repeated_frames_busy_boundaries_and_new_epochs() {
        let mut boundary = IntervalBoundary::default();
        assert!(!boundary.observe(interval(), None, false));
        assert!(boundary.observe(interval(), Some((1, 3)), false));
        assert!(!boundary.observe(interval(), Some((1, 3)), false));
        assert!(!boundary.observe(interval(), Some((1, 6)), true));
        assert!(!boundary.observe(interval(), Some((1, 6)), false));
        assert!(!boundary.observe(interval(), Some((2, 0)), false));
        assert!(boundary.observe(interval(), Some((2, 3)), false));
        // Jump past a boundary: no catch-up request.
        assert!(!boundary.observe(interval(), Some((2, 7)), false));
    }

    fn tree() -> TriangleBvh {
        TriangleBvh {
            nodes: vec![
                BvhNode {
                    min: [0.0; 3],
                    max: [2.0, 1.0, 1.0],
                    first_index: 1,
                    primitive_count: 0,
                },
                BvhNode {
                    min: [0.0; 3],
                    max: [1.0; 3],
                    first_index: 0,
                    primitive_count: 1,
                },
                BvhNode {
                    min: [1.0, 0.0, 0.0],
                    max: [2.0, 1.0, 1.0],
                    first_index: 1,
                    primitive_count: 1,
                },
            ],
            triangle_refs: vec![0, 1],
            leaf_nodes: vec![1, 2],
            internal_nodes: vec![0],
            refit_layers: vec![BvhRefitLayer {
                depth: 0,
                first: 0,
                count: 1,
            }],
            max_depth: 1,
        }
    }

    #[test]
    fn cpu_snapshot_cost_translation_and_overlapping_deformation() {
        let original = tree();
        // Root area 10, leaf areas 6 each: (10+6+6)/10 = 2.2.
        assert!((snapshot_sah(&original).unwrap() - 2.2).abs() < 1e-12);
        let mut translated = original.clone();
        for node in &mut translated.nodes {
            for i in 0..3 {
                node.min[i] += 16.0;
                node.max[i] += 16.0;
            }
        }
        assert_eq!(snapshot_sah(&original), snapshot_sah(&translated));
        let mut overlapping = original.clone();
        overlapping.nodes[1].max = original.nodes[0].max;
        overlapping.nodes[2].min = original.nodes[0].min;
        assert_eq!(snapshot_sah(&overlapping), Some(3.0));
        // Hand-constructed CPU bounds, not simulated deformation or GPU evidence.
        for node in &mut overlapping.nodes {
            node.max = node.min;
        }
        assert_eq!(snapshot_sah(&overlapping), None);
    }
}

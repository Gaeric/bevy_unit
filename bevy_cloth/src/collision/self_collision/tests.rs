//! CPU models for candidate search; these tests do not execute or validate WESL.

use glam::Vec3;

use super::{LANES, PreparedSelfCollision};
use crate::solver::layout::Particle;

#[derive(Clone, Copy)]
struct Point {
    current: Vec3,
    rest: Vec3,
    inv_mass: f32,
}

fn cell(position: Vec3, diameter: f32) -> Option<[i32; 3]> {
    let scaled = position / (2.0 * diameter);
    (scaled.is_finite() && scaled.abs().max_element() <= 4_194_304.0)
        .then(|| scaled.floor().as_ivec3().to_array())
}

fn bucket([x, y, z]: [i32; 3], buckets: usize) -> usize {
    let bits = (x as u32).wrapping_mul(92837111)
        ^ (y as u32).wrapping_mul(689287499)
        ^ (z as u32).wrapping_mul(283923481);
    bits as usize % buckets
}

fn block_prefix(counts: &[usize]) -> Vec<usize> {
    let mut prefix = Vec::with_capacity(counts.len());
    let mut totals = Vec::new();
    for block in counts.chunks(LANES as usize) {
        let mut count = 0;
        for item in block {
            prefix.push(count);
            count += item;
        }
        totals.push(count);
    }
    if totals.len() > 1 {
        let block_offsets = block_prefix(&totals);
        for (i, value) in prefix.iter_mut().enumerate() {
            *value += block_offsets[i / LANES as usize];
        }
    }
    prefix
}

fn ratio_squared(delta: Vec3, diameter: f32) -> f32 {
    if !delta.is_finite() || delta.abs().max_element() > diameter {
        return 2.0;
    }
    (delta / diameter).length_squared()
}

#[derive(Debug, PartialEq, Eq)]
struct Candidates {
    rows: Vec<Vec<usize>>,
    max_required: usize,
    overflow_rows: usize,
    invalid_particles: usize,
}

fn hash_candidates(points: &[Point], diameter: f32, buckets: usize, capacity: usize) -> Candidates {
    let cells: Vec<_> = points.iter().map(|p| cell(p.current, diameter)).collect();
    let mut counts = vec![0; buckets];
    for &key in cells.iter().flatten() {
        counts[bucket(key, buckets)] += 1;
    }
    let offsets = block_prefix(&counts);
    let mut cursors = offsets.clone();
    let mut ids = vec![usize::MAX; points.len()];
    // Deliberately reverse the scatter order relative to the reference search.
    for (i, key) in cells.iter().enumerate().rev() {
        let Some(key) = key else { continue; };
        let index = bucket(*key, buckets);
        ids[cursors[index]] = i;
        cursors[index] += 1;
    }
    let mut out = Candidates {
        rows: vec![Vec::new(); points.len()],
        max_required: 0,
        overflow_rows: 0,
        invalid_particles: cells.iter().filter(|key| key.is_none()).count(),
    };
    for (i, key) in cells.iter().enumerate() {
        let Some([x, y, z]) = *key else { continue; };
        let mut required = 0;
        let row = &mut out.rows[i];
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let target = [x + dx, y + dy, z + dz];
                    let k = bucket(target, buckets);
                    for &j in &ids[offsets[k]..offsets[k] + counts[k]] {
                        if i == j || cells[j] != Some(target)
                            || (points[i].inv_mass == 0.0 && points[j].inv_mass == 0.0)
                            || ratio_squared(points[i].current - points[j].current, diameter) >= 1.0
                            || ratio_squared(points[i].rest - points[j].rest, diameter) <= 1.0
                        {
                            continue;
                        }
                        required += 1;
                        let index = row.binary_search(&j).expect_err("true-cell filtering prevents duplicates");
                        if index < capacity {
                            row.insert(index, j);
                            if row.len() > capacity { row.pop(); }
                        }
                    }
                }
            }
        }
        out.max_required = out.max_required.max(required);
        out.overflow_rows += (required > capacity) as usize;
    }
    out
}

fn brute_force(points: &[Point], diameter: f64, capacity: usize) -> Candidates {
    let valid = |p: Point| {
        p.current.to_array().iter().all(|&x| {
            (f64::from(x) / (2.0 * diameter)).abs() <= 4_194_304.0
        })
    };
    let mut out = Candidates {
        rows: vec![Vec::new(); points.len()],
        max_required: 0,
        overflow_rows: 0,
        invalid_particles: points.iter().filter(|&&p| !valid(p)).count(),
    };
    for (i, a) in points.iter().copied().enumerate().filter(|&(_, p)| valid(p)) {
        for (j, b) in points.iter().copied().enumerate().skip(i + 1).filter(|&(_, p)| valid(p)) {
            if a.inv_mass == 0.0 && b.inv_mass == 0.0 { continue; }
            if a.current.as_dvec3().distance(b.current.as_dvec3()) < diameter
                && a.rest.as_dvec3().distance(b.rest.as_dvec3()) > diameter
            {
                out.rows[i].push(j);
                out.rows[j].push(i);
            }
        }
    }
    for row in &mut out.rows {
        out.max_required = out.max_required.max(row.len());
        out.overflow_rows += (row.len() > capacity) as usize;
        row.truncate(capacity);
    }
    out
}

fn point(x: f32, rest_x: f32, inv_mass: f32) -> Point {
    Point { current: Vec3::new(x, 0.0, 0.0), rest: Vec3::new(rest_x, 0.0, 0.0), inv_mass }
}

#[test]
fn counting_hash_matches_brute_force_with_aliases_and_overflow() {
    let points = [
        point(-0.125, 0.0, 0.0),
        point(0.125, 2.0, 1.0),
        point(0.0, 1.0, 0.0), // Excluded at exactly one diameter at rest.
        point(0.0, 6.0, 2.0), // Coincident with ID 2 now, not at rest.
        point(1.125, 4.0, 1.0), // Exactly one diameter from ID 1 now.
        point(100.0, 5.0, 1.0), // Distant true cell, aliased when there is one bucket.
        point(f32::NAN, 8.0, 1.0),
        point(10_000_000.0, 9.0, 1.0),
    ];
    for buckets in [1, 2, 7, 65, 4097] {
        for capacity in [1, 2, 7] {
            assert_eq!(hash_candidates(&points, 1.0, buckets, capacity), brute_force(&points, 1.0, capacity));
        }
    }
}

#[test]
fn folded_layers_make_contacts_but_raw_rest_splits_do_not() {
    let points = [
        Point { current: Vec3::ZERO, rest: Vec3::ZERO, inv_mass: 0.0 },
        Point { current: Vec3::ZERO, rest: Vec3::ZERO, inv_mass: 1.0 },
        Point { current: Vec3::Z * 0.25, rest: Vec3::X * 3.0, inv_mass: 1.0 },
        Point { current: Vec3::Z * 0.25 + Vec3::Y * 3.0, rest: Vec3::Y * 3.0, inv_mass: 1.0 },
    ];
    let candidates = hash_candidates(&points, 1.0, 1, 3);
    assert_eq!(candidates, brute_force(&points, 1.0, 3));
    assert_eq!(candidates.rows, [vec![2], vec![2], vec![0, 1], vec![]]);
}

#[test]
fn hierarchical_scan_matches_prefix_with_partial_blocks() {
    for length in [1, 63, 64, 65, 4096, 4097] {
        let counts: Vec<_> = (0..length).map(|i| (i * 13) % 7).collect();
        let mut expected = 0;
        for (prefix, count) in block_prefix(&counts).into_iter().zip(counts) {
            assert_eq!(prefix, expected);
            expected += count;
        }
    }
}

#[test]
fn preparation_keeps_raw_rest_positions_and_bounds_candidate_indices() {
    let particles = [Particle { position: [2.0, 3.0, 4.0], inv_mass: 1.0 }; 2];
    let prepared = PreparedSelfCollision::new(0.125, u32::MAX, &particles)
        .expect("capacity clamps to the number of other particles");
    assert_eq!(prepared.params.capacity, 1);
    assert_eq!(prepared.candidate_words, 4);
    assert_eq!(prepared.rest_positions, [[2.0, 3.0, 4.0, 0.0]; 2]);
    assert!(PreparedSelfCollision::new(0.125, 0, &particles).is_err());
}

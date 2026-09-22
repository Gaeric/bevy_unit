//! spatial hash against brute force, plus the layout / sentinel / clamp rules (plan.org §5.1).

use bevy_cloth::sim::{hash::SpatialHash, params::SimParams};
use glam::Vec3;

/// xorshift32, so the point sets repeat exactly without pulling in `rand`.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }

    fn points(&mut self, count: usize, extent: f32) -> Vec<Vec3> {
        (0..count)
            .map(|_| Vec3::new(self.next(), self.next(), self.next()) * extent)
            .collect()
    }
}

fn params(max_num_neighbors: u32, particle_diameter: f32, cell_scalar: f32) -> SimParams {
    SimParams {
        max_num_neighbors,
        particle_diameter,
        hash_cell_size_scalar: cell_scalar,
        ..Default::default()
    }
}

/// every `j != i` within `spacing`, minus the pairs that were already touching, sorted by id.
fn brute_force(
    positions: &[Vec3],
    originals: &[Vec3],
    spacing: f32,
    diameter: f32,
) -> Vec<Vec<u32>> {
    (0..positions.len())
        .map(|i| {
            let mut neighbors: Vec<u32> = (0..positions.len() as u32)
                .filter(|&j| {
                    let j = j as usize;
                    j != i
                        && (positions[i] - positions[j]).length_squared() < spacing * spacing
                        && (originals[i] - originals[j]).length_squared() > diameter * diameter
                })
                .collect();
            neighbors.sort_unstable();
            neighbors
        })
        .collect()
}

fn rebuild(positions: &[Vec3], originals: &[Vec3], params: &SimParams) -> SpatialHash {
    let mut hash = SpatialHash::new(params.particle_diameter, positions.len(), params);
    hash.rebuild(positions, originals, params);
    hash
}

fn collected(hash: &SpatialHash, i: usize) -> Vec<u32> {
    let mut neighbors: Vec<u32> = hash.neighbors_of(i).collect();
    println!("neighbors is {:?}", neighbors);
    neighbors.sort_unstable();
    neighbors
}

#[test]
fn matches_brute_force() {
    // spacing = 2, so a 10^3 box holds only a handful of points per cell: neither the cap nor the
    // per-cell clamp can drop anything
    let params = params(8096, 1.0, 1.0);
    let mut rng = Rng(0x1234_5678);
    let positions = rng.points(20000, 10.0);
    let mut rng = Rng(0x2234_5678);
    let originals = rng.points(20000, 10.0);

    println!("positions: {positions:?}");
    println!(" originals: {originals:?}");

    let hash = rebuild(&positions, &originals, &params);

    assert_eq!(hash.spacing, 1.0);
    let expected = brute_force(
        &positions,
        &originals,
        hash.spacing,
        params.particle_diameter,
    );
    assert!(expected.iter().all(|neighbors| neighbors.len() < 64));

    for (i, want) in expected.iter().enumerate() {
        let mut got = collected(&hash, i);
        got.dedup();
        assert_eq!(&got, want, "particle {i}");
    }
}

//! uniform grid spatial hash, 1:1 with `SpatialHashGPU.hpp` / `.cu` / `.cuh` (plan.org §1.6).

use glam::Vec3;

use crate::sim::params::SimParams;

/// the source's empty-slot sentinel `0xffffffff`.
pub const EMPTY: u32 = u32::MAX;

/// `SpatialHashGPU.cu:17`, `ComputeIntCoord`. C++ `(int)floor` truncates; rust saturates on overflow.
pub fn compute_int_coord(value: f32, spacing: f32) -> i32 {
    (value / spacing).floor() as i32
}

/// `SpatialHashGPU.cu:22`, `HashCoords`: `abs(h % tableSize)` with
/// `h = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481)`. the C++ ints overflow (UB), so we wrap;
/// `unsigned_abs` avoids the `abs(i32::MIN)` overflow.
pub fn hash_coords(x: i32, y: i32, z: i32, table_size: i32) -> u32 {
    let h = x.wrapping_mul(92_837_111) ^ y.wrapping_mul(689_287_499) ^ z.wrapping_mul(283_923_481);
    h.wrapping_rem(table_size).unsigned_abs()
}

/// `SpatialHashGPU.cu:26`, `HashPosition`.
pub fn hash_position(position: Vec3, spacing: f32, table_size: i32) -> u32 {
    hash_coords(
        compute_int_coord(position.x, spacing),
        compute_int_coord(position.y, spacing),
        compute_int_coord(position.z, spacing),
        table_size,
    )
}

/// `SpatialHashGPU.hpp:13`, `SpatialHashGPU`.
///
/// the slots of particle `i` live at `neighbors[i + num_particles * slot]`; the stride is the
/// particle count, not `max_num_neighbors` (§1.6).
#[derive(Clone, Debug, Default)]
pub struct SpatialHash {
    /// `particle_diameter * hash_cell_size_scalar`
    pub spacing: f32,
    /// `2 * num_particles`
    pub table_size: usize,
    pub max_num_neighbors: usize,
    pub num_particles: usize,
    pub particle_hash: Vec<u32>,
    /// particle ids sorted by hash; ascending id inside one cell
    pub particle_index: Vec<u32>,
    /// `EMPTY` when the cell holds no particle
    pub cell_start: Vec<u32>,
    /// one past the last slot; only meaningful where `cell_start != EMPTY`
    pub cell_end: Vec<u32>,
    /// `neighbors[i + num_particles * slot]`, `EMPTY` terminated
    pub neighbors: Vec<u32>,
}

impl SpatialHash {
    /// `SpatialHashGPU.hpp:16`. the source re-creates the hash on every `AddCloth`, sized for the
    /// whole solver.
    pub fn new(particle_diameter: f32, max_num_objects: usize, params: &SimParams) -> Self {
        let max_num_neighbors = params.max_num_neighbors as usize;
        Self {
            spacing: particle_diameter * params.hash_cell_size_scalar,
            table_size: 2 * max_num_objects,
            max_num_neighbors,
            num_particles: max_num_objects,
            particle_hash: vec![0; max_num_objects],
            particle_index: vec![0; max_num_objects],
            cell_start: vec![EMPTY; 2 * max_num_objects],
            cell_end: vec![0; 2 * max_num_objects],
            neighbors: vec![EMPTY; max_num_objects * max_num_neighbors],
        }
    }

    /// the neighbours of `i` in slot order, stopping at the sentinel.
    pub fn neighbors_of(&self, i: usize) -> impl Iterator<Item = u32> + '_ {
        (0..self.max_num_neighbors).map_while(move |slot| {
            let j = self.neighbors[i + self.num_particles * slot];
            println!("slot {slot} neighbors: {j}");
            (j as usize <= self.num_particles).then_some(j)
        })
    }

    /// `SpatialHashGPU.cu:160`, `HashObjects`: hash, sort, cell ranges, neighbour cache.
    pub fn rebuild(&mut self, positions: &[Vec3], original_positions: &[Vec3], params: &SimParams) {
        assert_eq!(
            positions.len(),
            original_positions.len(),
            "the original-position snapshot must cover every particle"
        );
        if positions.is_empty() {
            return;
        }
        assert_eq!(
            positions.len(),
            self.num_particles,
            "SpatialHash::new fixes the particle count"
        );

        let table_size = self.table_size as i32;
        for (i, position) in positions.iter().enumerate() {
            self.particle_index[i] = i as u32;
            self.particle_hash[i] = hash_position(*position, self.spacing, table_size);
        }

        // the source radix-sorts over `ceil(log2(tableSize))` bits; the hashes are already below
        // `tableSize`, so a stable sort by hash value is the same order
        let hashes = &self.particle_hash;
        self.particle_index.sort_by_key(|&id| hashes[id as usize]);

        self.cell_start.fill(EMPTY);
        self.cell_end.fill(0);
        let mut start = 0;
        while start < self.num_particles {
            let hash = hashes[self.particle_index[start] as usize] as usize;
            let mut end = start + 1;
            while end < self.num_particles
                && hashes[self.particle_index[end] as usize] as usize == hash
            {
                end += 1;
            }
            self.cell_start[hash] = start as u32;
            self.cell_end[hash] = end as u32;
            start = end;
        }

        self.cache_neighbors(positions, original_positions, params);
    }

    /// `SpatialHashGPU.cu:66`, `CacheNeighbors_Kernel`.
    fn cache_neighbors(
        &mut self,
        positions: &[Vec3],
        original_positions: &[Vec3],
        params: &SimParams,
    ) {
        // the source leaves the stale slots behind the sentinel in place; we clear them instead so
        // that a rebuild is deterministic
        self.neighbors.fill(EMPTY);

        let n = self.num_particles;
        let max = self.max_num_neighbors;
        let cell_spacing2 = self.spacing * self.spacing;
        let particle_diameter2 = params.particle_diameter * params.particle_diameter;
        let table_size = self.table_size as i32;

        let a = hash_coords(3, -1, 1, table_size);
        let b = hash_coords(3, 1, 1, table_size);

        println!("table_size {table_size}, a is {a}, b is {b}");

        for id in 0..n {
            let position = positions[id];
            let original = original_positions[id];
            let ix = compute_int_coord(position.x, self.spacing);
            let iy = compute_int_coord(position.y, self.spacing);
            let iz = compute_int_coord(position.z, self.spacing);

            let mut slot = 0;
            'cells: for x in ix - 1..=ix + 1 {
                for y in iy - 1..=iy + 1 {
                    for z in iz - 1..=iz + 1 {
                        let cell = hash_coords(x, y, z, table_size) as usize;
                        let start = self.cell_start[cell];
                        if start == EMPTY {
                            continue;
                        }
                        // the source clamps the cell to `maxNumNeighbors` (its own BUG_LOG), so a
                        // dense cell can drop neighbours
                        let end = self.cell_end[cell].min(start + max as u32);
                        if id == 0 {
                            println!("for {x}|{y}|{z} cell {cell} start/end {start}/{end}")
                        }
                        for i in start..end {
                            let other = self.particle_index[i as usize] as usize;
                            if other == id {
                                continue;
                            }
                            if (position - positions[other]).length_squared() >= cell_spacing2 {
                                continue;
                            }
                            // collision filtering: pairs that already touched at the start are out
                            if (original - original_positions[other]).length_squared()
                                <= particle_diameter2
                            {
                                continue;
                            }

                            self.neighbors[id + n * slot] = other as u32;
                            if id == 0 {
                                println!("id 0 slot {slot} {i} neighbors: {other}");
                            }

                            slot += 1;
                            if slot >= max {
                                // the source returns here, so a full list gets no sentinel
                                break 'cells;
                            }
                        }
                    }
                }
            }

            if slot < max {
                self.neighbors[id + n * slot] = EMPTY;
            }
        }
    }
}

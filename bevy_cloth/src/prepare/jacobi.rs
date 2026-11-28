//! Particle-to-correction adjacency for the Jacobi gather pass.

use crate::solver::layout::{BendConstraint, StretchConstraint, TetherConstraint};

#[derive(Clone, Debug)]
pub(crate) struct JacobiSchedule {
    pub offsets: Vec<u32>,
    pub slots: Vec<u32>,
}

impl JacobiSchedule {
    pub fn new(
        particle_count: usize,
        stretch: &[StretchConstraint],
        bend: &[BendConstraint],
        tethers: &[TetherConstraint],
    ) -> Result<Self, String> {
        let slot_count = u32::try_from(
            2_u64 * stretch.len() as u64 + 4_u64 * bend.len() as u64 + tethers.len() as u64,
        )
        .map_err(|_| "Jacobi endpoint count exceeds u32")?;
        // Same order as jacobi.wesl: two slots per stretch, four per bend,
        // then one per tether. The fixed tether anchor needs no correction slot.
        let endpoints = {
            stretch
                .iter()
                .flat_map(|edge| edge.particles)
                .chain(bend.iter().flat_map(|hinge| hinge.particles))
                .chain(tethers.iter().map(|tether| tether.particle))
        };
        Self::from_endpoints(particle_count, slot_count, endpoints)
    }

    fn from_endpoints(
        particle_count: usize,
        slot_count: u32,
        endpoints: impl Iterator<Item = u32> + Clone,
    ) -> Result<Self, String> {
        let mut offsets = vec![0_u32; particle_count + 1];
        for particle in endpoints.clone() {
            offsets[particle as usize + 1] += 1;
        }
        for particle in 0..particle_count {
            offsets[particle + 1] += offsets[particle];
        }
        let mut cursors = offsets[..particle_count].to_vec();
        let mut slots = vec![0; slot_count as usize];
        for (slot, particle) in endpoints.enumerate() {
            let cursor = &mut cursors[particle as usize];
            slots[*cursor as usize] = slot as u32;
            *cursor += 1;
        }
        Ok(Self { offsets, slots })
    }
}

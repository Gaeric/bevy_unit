//! Particle-to-correction adjacency for the two Jacobi gather passes.

use crate::solver::layout::{BendConstraint, StretchConstraint, TetherConstraint};

#[derive(Clone, Debug)]
pub(crate) struct JacobiSchedule {
    pub stretch: JacobiAdjacency,
    pub bend_and_tethers: JacobiAdjacency,
}

#[derive(Clone, Debug)]
pub(crate) struct JacobiAdjacency {
    pub offsets: Vec<u32>,
    /// Slot indices address the shared correction array.
    pub slots: Vec<u32>,
}

impl JacobiSchedule {
    pub fn new(
        particle_count: usize,
        stretch: &[StretchConstraint],
        bend: &[BendConstraint],
        tethers: &[TetherConstraint],
    ) -> Result<Self, String> {
        // Match jacobi.wesl: stretch has two slots per edge. The second phase
        // has four per hinge, then one per tether (only its free endpoint).
        let mut schedule = Self {
            stretch: JacobiAdjacency::new(
                particle_count,
                2 * stretch.len() as u64,
                stretch.iter().flat_map(|edge| edge.particles),
            )?,
            bend_and_tethers: JacobiAdjacency::new(
                particle_count,
                4 * bend.len() as u64 + tethers.len() as u64,
                bend.iter()
                    .flat_map(|hinge| hinge.particles)
                    .chain(tethers.iter().map(|tether| tether.particle)),
            )?,
        };
        let stretch_slots = schedule.stretch.slots.len() as u32;
        for slot in &mut schedule.bend_and_tethers.slots {
            *slot = slot.checked_add(stretch_slots).ok_or("Jacobi endpoint count exceeds u32")?;
        }
        Ok(schedule)
    }
}

impl JacobiAdjacency {
    fn new(
        particle_count: usize,
        slot_count: u64,
        endpoints: impl Iterator<Item = u32> + Clone,
    ) -> Result<Self, String> {
        let slot_count =
            u32::try_from(slot_count).map_err(|_| "Jacobi endpoint count exceeds u32")?;
        if slot_count == 0 {
            return Ok(Self {
                offsets: Vec::new(),
                slots: Vec::new(),
            });
        }
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

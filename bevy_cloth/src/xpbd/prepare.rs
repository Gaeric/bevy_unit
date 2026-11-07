//! Validate mesh inputs and prepare initial cloth data.

use bevy::{mesh::PrimitiveTopology, prelude::*};
use std::collections::BTreeSet;

use super::{
    StretchCloth,
    layout::{FixedAttachment, Particle, PreparedCloth, Stretch},
};

/// Validate all inputs before preparing particles and constraints.
///
/// Returns readable positions and a validated triangle-index array.
fn validate_inputs<'a>(
    mesh: &'a Mesh,
    transform: &GlobalTransform,
    config: &StretchCloth,
) -> Result<(&'a [[f32; 3]], Vec<u32>), String> {
    // Validate configuration inputs, not derived calculation results.
    if !config.particle_mass.is_finite() || config.particle_mass <= 0.0 {
        return Err("particle_mass must be positive and finite".into());
    }

    if !config.stretch_compliance.is_finite() || config.stretch_compliance < 0.0 {
        return Err("stretch_compliance must be nonnegative and finite".into());
    }

    // Validate mesh topology, position format and source values.
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err("cloth mesh must use TriangleList topology".into());
    }

    let positions = mesh
        .try_attribute(Mesh::ATTRIBUTE_POSITION)
        .map_err(|error| format!("cannot read POSITION: {error}"))?
        .as_float3()
        .ok_or("POSITION must use Float32x3")?;

    if positions.iter().flatten().any(|value| !value.is_finite()) {
        return Err("POSITION contains non-finite values".into());
    }

    if !transform.to_matrix().is_finite() {
        return Err("world transform must be finite".into());
    }

    let particle_count =
        u32::try_from(positions.len()).map_err(|_| "particle count exceeds u32")?;

    // Normalize both supported index formats into u32.
    // An unindexed TriangleList uses consecutive triples of vertices.
    let indices: Vec<u32> = match mesh
        .try_indices_option()
        .map_err(|error| format!("cannot read indices: {error}"))?
    {
        Some(indices) => indices.iter().map(|index| index as u32).collect(),
        None => (0..particle_count).collect(),
    };

    // Validate triangle completeness and all index references once.
    if !indices.len().is_multiple_of(3) {
        return Err("triangle index count must be a multiple of three".into());
    }

    if indices.iter().any(|&index| index >= particle_count) {
        return Err("triangle index is out of range".into());
    }

    // Validate fixed attachment references and targets.
    let mut attached_particles = BTreeSet::new();

    for attachment in &config.attachments {
        if attachment.particle >= particle_count {
            return Err("attachment particle index is out of range".into());
        }

        if !attachment.position.is_finite() {
            return Err("attachment target must be finite".into());
        }

        if !attached_particles.insert(attachment.particle) {
            return Err("a particle has multiple attachment entries".into());
        }
    }

    Ok((positions, indices))
}

/// Prepare particles and stretch constraints from validated inputs.
pub(super) fn prepare_mesh(
    mesh: &Mesh,
    transform: &GlobalTransform,
    config: &StretchCloth,
) -> Result<PreparedCloth, String> {
    let (positions, indices) = validate_inputs(mesh, transform, config)?;

    // Preserve source vertex indices and convert positions to world space.
    let inv_mass = config.particle_mass.recip();

    let mut particles: Vec<Particle> = positions
        .iter()
        .map(|position| Particle {
            position: transform.transform_point(Vec3::from(*position)).to_array(),
            inv_mass,
        })
        .collect();

    let velocities = vec![[0.0; 4]; particles.len()];

    // Extract unique undirected edges. Self-connections add no constraint.
    let mut edges = BTreeSet::new();

    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [triangle[0], triangle[1], triangle[2]];

        for (i, j) in [(a, b), (b, c), (c, a)] {
            if i != j {
                edges.insert([i.min(j), i.max(j)]);
            }
        }
    }

    // Measure rest lengths before applying attachment targets.
    let stretch = edges
        .into_iter()
        .map(|[i, j]| {
            let a = Vec3::from(particles[i as usize].position);
            let b = Vec3::from(particles[j as usize].position);

            Stretch {
                particles: [i, j],
                rest_length: a.distance(b),
                compliance: config.stretch_compliance,
            }
        })
        .collect();

    // Fixed targets affect the initial state, not the rest geometry.
    let mut attachments = Vec::with_capacity(config.attachments.len());

    for attachment in &config.attachments {
        let target = attachment.position.to_array();
        let particle = &mut particles[attachment.particle as usize];

        particle.position = target;
        particle.inv_mass = 0.0;

        attachments.push(FixedAttachment {
            target,
            particle: attachment.particle,
        });
    }

    Ok(PreparedCloth {
        particles,
        velocities,
        stretch,
        attachments,
        groups: Vec::new(),
    })
}

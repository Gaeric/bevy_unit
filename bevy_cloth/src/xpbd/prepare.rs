//! Validate mesh inputs and prepare initial cloth data.

use bevy::{camera::visibility::NoFrustumCulling, mesh::PrimitiveTopology, prelude::*};
use std::collections::BTreeSet;

use super::{
    StretchCloth,
    layout::{ConstraintGroup, FixedAttachment, Particle, PreparedCloth, StepParams, Stretch},
    tick::ClothStepSettings,
};

/// Prepare newly added cloths once, after world transforms have propagated.
/// Source meshes must already be loaded and CPU-readable.
pub(super) fn prepare_added_cloths(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    time: Res<Time<Fixed>>,
    cloths: Query<
        (
            Entity,
            &Mesh3d,
            &GlobalTransform,
            &StretchCloth,
            &ClothStepSettings,
        ),
        Added<StretchCloth>,
    >,
) {
    for (entity, mesh_handle, transform, config, settings) in &cloths {
        if !settings.gravity.is_finite() || settings.substeps == 0 || settings.iterations == 0 {
            warn!(
                ?entity,
                "Cloth settings require finite gravity and nonzero substeps/iterations"
            );
            continue;
        }

        let mesh = meshes
            .get(&mesh_handle.0)
            .expect("adding StretchCloth requires a loaded Mesh");
        let prepared = prepare_mesh(mesh, transform, config).expect("cloth preparation failed");
        let step = StepParams {
            gravity: settings.gravity.extend(0.0).to_array(),
            dt: time.timestep().as_secs_f32() / settings.substeps as f32,
            particle_count: prepared.particles.len() as u32,
            stretch_count: prepared.stretch.len() as u32,
            attachment_count: prepared.attachments.len() as u32,
        };

        info!(
            ?entity,
            particles = prepared.particles.len(),
            stretch = prepared.stretch.len(),
            colors = prepared.groups.len(),
            substeps = settings.substeps,
            iterations = settings.iterations,
            "XPBD cloth prepared"
        );

        // GPU deformation does not update the mesh's CPU-side bounds.
        commands
            .entity(entity)
            .insert((prepared, step, NoFrustumCulling));
    }
}

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

/// Color stretch constraints and arrange them into contiguous groups.
///
/// Constraints within one group share no particles.
/// Particle indices must already be validated.
fn color_stretch(
    particle_count: usize,
    stretch: Vec<Stretch>,
) -> Result<(Vec<Stretch>, Vec<ConstraintGroup>), String> {
    // Group ranges use u32 indices.
    u32::try_from(stretch.len()).map_err(|_| "stretch count exceeds u32")?;

    let constraint_count = stretch.len();

    // used_colors[particle] records colors already touching this particle.
    let mut used_colors: Vec<Vec<usize>> = vec![Vec::new(); particle_count];

    // buckets[color] contains all constraints assigned to this color.
    let mut buckets: Vec<Vec<Stretch>> = Vec::new();

    for constraint in stretch {
        let [a, b] = constraint.particles.map(|index| index as usize);

        // Find the first color unused by both particles.
        let mut color = 0;

        while used_colors[a].contains(&color) || used_colors[b].contains(&color) {
            color += 1;
        }

        // Record this color on both particles.
        used_colors[a].push(color);
        used_colors[b].push(color);

        // Create a bucket when this is a new color.
        if color == buckets.len() {
            buckets.push(Vec::new());
        }

        buckets[color].push(constraint);
    }

    // Join the buckets into one array and record each group's range.
    let mut reordered = Vec::with_capacity(constraint_count);
    let mut groups = Vec::with_capacity(buckets.len());

    for bucket in buckets {
        let start = reordered.len() as u32;
        let count = bucket.len() as u32;

        reordered.extend(bucket);

        groups.push(ConstraintGroup { start, count });
    }

    Ok((reordered, groups))
}

/// Build CSR normal adjacency from validated triangle indices.
/// Vertex i's oriented neighbor pairs occupy offsets[i]..offsets[i + 1].
fn build_normal_adjacency(particle_count: usize, indices: &[u32]) -> (Vec<u32>, Vec<[u32; 2]>) {
    // Count triangle incidences one slot to the right for the prefix sum.
    let mut offsets = vec![0_u32; particle_count + 1];

    for &vertex in indices {
        offsets[vertex as usize + 1] += 1;
    }

    // Convert counts into each vertex's range boundaries.
    for vertex in 0..particle_count {
        offsets[vertex + 1] += offsets[vertex];
    }

    // Temporary cursors track the next free slot; offsets remain unchanged.
    let mut cursors = offsets[..particle_count].to_vec();
    let mut pairs = vec![[0_u32; 2]; indices.len()];

    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [triangle[0], triangle[1], triangle[2]];

        // Cyclic ordering preserves the triangle winding for every vertex.
        for (vertex, pair) in [(a, [b, c]), (b, [c, a]), (c, [a, b])] {
            let cursor = &mut cursors[vertex as usize];
            pairs[*cursor as usize] = pair;
            *cursor += 1;
        }
    }

    (offsets, pairs)
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

    // Reorder constraints by color and record each group's range.
    let (stretch, groups) = color_stretch(particles.len(), stretch)?;

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

    let (normal_offsets, normal_pairs) = build_normal_adjacency(particles.len(), &indices);

    Ok(PreparedCloth {
        particles,
        velocities,
        stretch,
        attachments,
        groups,
        normal_offsets,
        normal_pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::build_normal_adjacency;

    #[test]
    fn normal_adjacency_groups_oriented_pairs_by_vertex() {
        // Vertex 4 is isolated; its CSR range must be empty.
        let (offsets, pairs) = build_normal_adjacency(5, &[0, 1, 2, 0, 2, 3]);

        assert_eq!(offsets, vec![0, 2, 3, 5, 6, 6]);
        assert_eq!(pairs, vec![[1, 2], [2, 3], [2, 0], [0, 1], [3, 0], [0, 2]]);
    }
}

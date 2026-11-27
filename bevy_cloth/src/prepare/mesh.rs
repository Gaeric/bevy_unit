//! Validate mesh inputs and prepare initial cloth data.

use bevy::{mesh::PrimitiveTopology, prelude::*};
use std::collections::BTreeSet;

use crate::{
    collision::self_collision::PreparedSelfCollision,
    components::{
        Cloth, ClothLongRangeAttachments, ClothSelfCollision, ClothSolver, ClothVbdMaterial,
        DynamicAttachment,
    },
    solver::layout::{BendConstraint, Particle, StretchConstraint, TetherConstraint},
};

use super::{
    ColoredGsSchedule, ConstraintColorRange, ConstraintSchedule, JacobiSchedule, PreparedCloth,
    bending::{color_bending, prepare_bending},
    tether::prepare_tethers,
    vbd::VbdRestData,
};

/// Solver choice with the material already resolved for VBD.
pub(super) enum MeshSolver {
    Xpbd(ClothSolver),
    Vbd {
        material: ClothVbdMaterial,
        contact_stiffness: f32,
    },
}

impl From<ClothSolver> for MeshSolver {
    fn from(solver: ClothSolver) -> Self {
        Self::Xpbd(solver)
    }
}

/// Validate all inputs before preparing particles and constraints.
///
/// Returns readable positions and a validated triangle-index array.
fn validate_inputs<'a>(
    mesh: &'a Mesh,
    transform: &GlobalTransform,
    config: &Cloth,
) -> Result<(&'a [[f32; 3]], Vec<u32>), String> {
    if !config.particle_mass.is_finite() || config.particle_mass <= 0.0 {
        return Err("particle_mass must be positive and finite".into());
    }

    if !config.collision_margin.is_finite() || config.collision_margin < 0.0 {
        return Err("collision_margin must be nonnegative and finite".into());
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

    Ok((positions, indices))
}

fn validate_attachments(
    particle_count: usize,
    config: &Cloth,
    dynamic: &[DynamicAttachment],
) -> Result<BTreeSet<u32>, String> {
    let mut attached_particles = BTreeSet::new();
    let pins = config
        .attachments
        .iter()
        .map(|pin| (pin.particle, pin.position))
        .chain(dynamic.iter().map(|pin| (pin.particle, pin.local_offset)));

    for (particle, position_or_offset) in pins {
        if particle as usize >= particle_count {
            return Err(format!("attachment particle {particle} is out of range"));
        }
        if !position_or_offset.is_finite() {
            return Err(format!(
                "attachment {particle} position or local offset must be finite"
            ));
        }
        if !attached_particles.insert(particle) {
            return Err(format!(
                "particle {particle} has multiple attachment entries"
            ));
        }
    }
    Ok(attached_particles)
}

/// Color stretch constraints and arrange them into contiguous color ranges.
///
/// Constraints within one color share no particles.
/// Particle indices must already be validated.
fn color_stretch(
    particle_count: usize,
    stretch_constraints: Vec<StretchConstraint>,
) -> Result<(Vec<StretchConstraint>, Vec<ConstraintColorRange>), String> {
    // Color ranges use u32 indices.
    u32::try_from(stretch_constraints.len()).map_err(|_| "stretch count exceeds u32")?;

    let constraint_count = stretch_constraints.len();

    // used_colors[particle] records colors already touching this particle.
    let mut used_colors: Vec<Vec<usize>> = vec![Vec::new(); particle_count];

    // buckets[color] contains all constraints assigned to this color.
    let mut buckets: Vec<Vec<StretchConstraint>> = Vec::new();

    for constraint in stretch_constraints {
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

    // Join the buckets into one array and record each color's range.
    let mut reordered = Vec::with_capacity(constraint_count);
    let mut colors = Vec::with_capacity(buckets.len());

    for bucket in buckets {
        let start = reordered.len() as u32;
        let count = bucket.len() as u32;

        reordered.extend(bucket);

        colors.push(ConstraintColorRange { start, count });
    }

    Ok((reordered, colors))
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

/// Prepare particles, constraints and the selected solver schedule.
pub(crate) fn prepare_mesh(
    mesh: &Mesh,
    transform: &GlobalTransform,
    config: &Cloth,
    dynamic: &[DynamicAttachment],
    solver: impl Into<MeshSolver>,
    long_range: Option<ClothLongRangeAttachments>,
    self_collision: Option<ClothSelfCollision>,
) -> Result<PreparedCloth, String> {
    let solver = solver.into();
    let (positions, indices) = validate_inputs(mesh, transform, config)?;
    let anchors = validate_attachments(positions.len(), config, dynamic)?;

    // Preserve source vertex indices and convert positions to world space.
    let inv_mass = config.particle_mass.recip();

    let mut particles: Vec<Particle> = positions
        .iter()
        .map(|position| Particle {
            position: transform.transform_point(Vec3::from(*position)).to_array(),
            inv_mass,
        })
        .collect();

    let (stretch_constraints, bend_constraints, tethers, schedule, self_collision) = match solver {
        MeshSolver::Xpbd(solver) => {
            let self_collision = self_collision
                .map(|settings| PreparedSelfCollision::new(settings, &particles))
                .transpose()?;
            let (stretch, bend, tethers, schedule) =
                prepare_xpbd(&particles, &indices, &anchors, config, solver, long_range)?;
            (stretch, bend, tethers, schedule, self_collision)
        }
        MeshSolver::Vbd {
            material,
            contact_stiffness,
        } => {
            if !contact_stiffness.is_finite() || contact_stiffness < 0.0 {
                return Err("VBD contact stiffness must be finite and nonnegative".into());
            }
            if !material.is_valid() {
                return Err(
                    "VBD stiffnesses, membrane stiffness sum and damping must be finite and nonnegative".into(),
                );
            }
            if !inv_mass.is_finite() || inv_mass == 0.0 {
                return Err("VBD inverse mass must be positive and finite".into());
            }
            if long_range.is_some() || self_collision.is_some() {
                return Err("VBD does not yet support LRA or self-collision".into());
            }
            let rest = VbdRestData::new(&particles, &indices, &anchors, material)?;
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                ConstraintSchedule::Vbd {
                    rest,
                    material,
                    contact_stiffness,
                },
                None,
            )
        }
    };

    // Pin targets affect the initial state, not the rest geometry. A dynamic
    // pin starts at its rest position until a valid target has been sampled.
    for index in anchors {
        particles[index as usize].inv_mass = 0.0;
    }
    for attachment in &config.attachments {
        particles[attachment.particle as usize].position = attachment.position.to_array();
    }

    let (normal_offsets, normal_pairs) = build_normal_adjacency(particles.len(), &indices);
    let velocities = vec![[0.0; 4]; particles.len()];

    Ok(PreparedCloth {
        particles,
        velocities,
        stretch_constraints,
        bend_constraints,
        tethers,
        schedule,
        self_collision,
        normal_offsets,
        normal_pairs,
    })
}

fn prepare_xpbd(
    particles: &[Particle],
    indices: &[u32],
    anchors: &BTreeSet<u32>,
    config: &Cloth,
    solver: ClothSolver,
    long_range: Option<ClothLongRangeAttachments>,
) -> Result<
    (
        Vec<StretchConstraint>,
        Vec<BendConstraint>,
        Vec<TetherConstraint>,
        ConstraintSchedule,
    ),
    String,
> {
    if !config.stretch_compliance.is_finite() || config.stretch_compliance < 0.0 {
        return Err("stretch_compliance must be nonnegative and finite".into());
    }
    if !config.bend_compliance.is_finite() || config.bend_compliance < 0.0 {
        return Err("bend_compliance must be nonnegative and finite".into());
    }

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
    let stretch_constraints: Vec<_> = edges
        .into_iter()
        .map(|[i, j]| {
            let a = Vec3::from(particles[i as usize].position);
            let b = Vec3::from(particles[j as usize].position);

            StretchConstraint {
                particles: [i, j],
                rest_length: a.distance(b),
                compliance: config.stretch_compliance,
            }
        })
        .collect();

    let bend_constraints = prepare_bending(particles, indices, config.bend_compliance)?;
    let tethers = match long_range {
        Some(settings) => {
            prepare_tethers(particles.len(), &stretch_constraints, anchors, settings)?
        }
        None => Vec::new(),
    };

    let (stretch_constraints, bend_constraints, schedule) = match solver {
        ClothSolver::ColoredGs => {
            let (stretch, stretch_colors) = color_stretch(particles.len(), stretch_constraints)?;
            let (bend, bend_colors) = color_bending(particles.len(), bend_constraints)?;
            let schedule = ConstraintSchedule::ColoredGs(ColoredGsSchedule {
                stretch_colors,
                bend_colors,
            });
            (stretch, bend, schedule)
        }
        ClothSolver::Jacobi => {
            let schedule = JacobiSchedule::new(
                particles.len(),
                &stretch_constraints,
                &bend_constraints,
                &tethers,
            )?;
            (
                stretch_constraints,
                bend_constraints,
                ConstraintSchedule::Jacobi(schedule),
            )
        }
        ClothSolver::Vbd => unreachable!("VBD uses its own rest geometry"),
    };
    Ok((stretch_constraints, bend_constraints, tethers, schedule))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::FixedAttachment;
    use bevy::{asset::RenderAssetUsages, mesh::Indices};

    fn hinge_mesh() -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD,
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
        )
        .with_inserted_indices(Indices::U32(vec![2, 0, 1, 3, 1, 0]))
    }

    fn config() -> Cloth {
        Cloth {
            particle_mass: 1.0,
            stretch_compliance: 0.0,
            bend_compliance: 0.125,
            attachments: Vec::new(),
            collision_margin: 0.02,
            double_sided: false,
        }
    }

    #[test]
    fn vbd_mesh_measures_rest_geometry_before_applying_pins() {
        let mesh = hinge_mesh();
        let mut config = config();
        config.attachments.push(FixedAttachment {
            particle: 2,
            position: Vec3::new(0.0, 1.0, 5.0),
        });
        let prepared = prepare_mesh(
            &mesh,
            &GlobalTransform::IDENTITY,
            &config,
            &[],
            MeshSolver::Vbd {
                material: ClothVbdMaterial {
                    shear: 1.0,
                    area: 1.0,
                    bend: 1.0,
                    ..default()
                },
                contact_stiffness: 0.0,
            },
            None,
            None,
        )
        .unwrap();
        let ConstraintSchedule::Vbd { rest, .. } = &prepared.schedule else {
            panic!("expected VBD schedule");
        };
        assert_eq!(prepared.particles[2].position, [0.0, 1.0, 5.0]);
        assert_eq!(prepared.particles[2].inv_mass, 0.0);
        assert_eq!(rest.membranes.len(), 2);
        assert_eq!(rest.membranes[0].rest_area, 0.5);
        assert_eq!(rest.hinges[0].rest_angle, 0.0);
        assert!(!rest.vertices.contains(&2));
        assert!(prepared.stretch_constraints.is_empty());
        assert!(prepared.bend_constraints.is_empty());
        assert!(prepared.tethers.is_empty());
    }

    #[test]
    fn colored_gs_keeps_hinge_edges_in_first_fit_dispatch_order() {
        let prepared = prepare_mesh(
            &hinge_mesh(),
            &GlobalTransform::IDENTITY,
            &config(),
            &[],
            ClothSolver::default(),
            None,
            None,
        )
        .unwrap();
        assert!(prepared.tethers.is_empty());
        let edges: Vec<_> = prepared
            .stretch_constraints
            .iter()
            .map(|edge| edge.particles)
            .collect();
        let ConstraintSchedule::ColoredGs(schedule) = &prepared.schedule else {
            panic!("expected Colored GS by default");
        };
        let ranges: Vec<_> = schedule
            .stretch_colors
            .iter()
            .map(|range| (range.start, range.count))
            .collect();
        assert_eq!(edges, [[0, 1], [0, 2], [1, 3], [0, 3], [1, 2]]);
        assert_eq!(ranges, [(0, 1), (1, 2), (3, 2)]);
    }

    #[test]
    fn bend_rest_angle_uses_world_geometry_before_attachment_targets() {
        let mesh = hinge_mesh();
        let transform = GlobalTransform::from(Transform {
            translation: Vec3::new(2.0, 3.0, 4.0),
            rotation: Quat::from_rotation_y(0.7),
            scale: Vec3::splat(2.0),
        });
        let mut config = config();
        config.attachments.push(FixedAttachment {
            particle: 2,
            position: transform.transform_point(Vec3::new(0.0, 1.0, 1.0)),
        });
        let prepared = prepare_mesh(
            &mesh,
            &transform,
            &config,
            &[],
            ClothSolver::ColoredGs,
            None,
            None,
        )
        .unwrap();
        assert_eq!(prepared.bend_constraints.len(), 1);
        assert!(prepared.bend_constraints[0].rest_angle.abs() < 1e-6);
        assert_eq!(prepared.bend_constraints[0].compliance, 0.125);
        let ConstraintSchedule::ColoredGs(schedule) = &prepared.schedule else {
            panic!("expected Colored GS schedule");
        };
        assert_eq!(schedule.bend_colors.len(), 1);
        let bend_range = schedule.bend_colors[0];
        assert_eq!((bend_range.start, bend_range.count), (0, 1));
        assert_eq!(
            prepared.particles[2].position,
            config.attachments[0].position.to_array()
        );
        assert_eq!(prepared.particles[2].inv_mass, 0.0);

        // Preparing this changed shape as rest geometry yields a nonzero fold.
        let changed = prepare_bending(&prepared.particles, &[2, 0, 1, 3, 1, 0], 0.125).unwrap();
        assert!(changed[0].rest_angle.abs() > 0.1);
    }

    #[test]
    fn jacobi_uses_the_same_constraints_and_gathers_both_types() {
        let mesh = hinge_mesh();
        let mut config = config();
        config.attachments.push(FixedAttachment {
            particle: 2,
            position: Vec3::new(0.0, 1.0, 1.0),
        });
        let prepare = |solver| {
            prepare_mesh(
                &mesh,
                &GlobalTransform::IDENTITY,
                &config,
                &[],
                solver,
                None,
                None,
            )
            .unwrap()
        };
        let gs = prepare(ClothSolver::ColoredGs);
        let jacobi = prepare(ClothSolver::Jacobi);

        let sorted_stretches = |cloth: &PreparedCloth| {
            let mut edges = cloth.stretch_constraints.clone();
            edges.sort_by_key(|edge| edge.particles);
            edges
        };
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&sorted_stretches(&gs)),
            bytemuck::cast_slice::<_, u8>(&sorted_stretches(&jacobi))
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&gs.bend_constraints),
            bytemuck::cast_slice::<_, u8>(&jacobi.bend_constraints)
        );
        assert_eq!(jacobi.particles[2].position, [0.0, 1.0, 1.0]);
        assert_eq!(jacobi.particles[2].inv_mass, 0.0);

        let ConstraintSchedule::Jacobi(schedule) = &jacobi.schedule else {
            panic!("expected Jacobi schedule");
        };
        // Five stretch edges followed by hinge endpoints [2, 3, 0, 1].
        assert_eq!(schedule.offsets, [0, 4, 8, 11, 14]);
        assert_eq!(
            schedule.slots,
            [0, 2, 4, 12, 1, 6, 8, 13, 3, 7, 10, 5, 9, 11]
        );
    }

    #[test]
    fn both_solvers_use_the_same_rest_tethers_and_jacobi_gathers_only_free_ends() {
        let mesh = hinge_mesh();
        let mut config = config();
        config.attachments.push(FixedAttachment {
            particle: 2,
            position: Vec3::new(0.0, 1.0, 5.0),
        });
        let prepare = |solver, long_range| {
            prepare_mesh(
                &mesh,
                &GlobalTransform::IDENTITY,
                &config,
                &[],
                solver,
                long_range,
                None,
            )
            .unwrap()
        };
        let long_range = Some(ClothLongRangeAttachments { compliance: 0.25 });
        let gs = prepare(ClothSolver::ColoredGs, long_range);
        let jacobi = prepare(ClothSolver::Jacobi, long_range);
        assert!(prepare(ClothSolver::Jacobi, None).tethers.is_empty());
        assert_eq!(gs.tethers, jacobi.tethers);
        let paths: Vec<_> = jacobi
            .tethers
            .iter()
            .map(|tether| (tether.particle, tether.anchor, tether.max_distance))
            .collect();
        // Anchor target is out of the rest plane. Particle 3 reaches anchor 2
        // through particle 0; both rest edges have length 1.
        assert_eq!(jacobi.particles[2].position, [0.0, 1.0, 5.0]);
        assert_eq!(paths, [(0, 2, 1.0), (1, 2, 2.0_f32.sqrt()), (3, 2, 2.0)]);

        let ConstraintSchedule::Jacobi(schedule) = &jacobi.schedule else {
            panic!("expected Jacobi schedule");
        };
        assert_eq!(schedule.offsets, [0, 5, 10, 13, 17]);
        assert_eq!(
            schedule.slots,
            [0, 2, 4, 12, 14, 1, 6, 8, 13, 15, 3, 7, 10, 5, 9, 11, 16]
        );
    }

    #[test]
    fn dynamic_pin_is_a_hard_anchor_for_both_solvers() {
        let mesh = hinge_mesh();
        let mut config = config();
        config.attachments.push(FixedAttachment {
            particle: 3,
            position: Vec3::new(0.0, -1.0, 5.0),
        });
        let pins = [DynamicAttachment {
            particle: 2,
            target: Entity::PLACEHOLDER,
            local_offset: Vec3::Z * 100.0,
        }];

        for solver in [ClothSolver::ColoredGs, ClothSolver::Jacobi] {
            let prepared = prepare_mesh(
                &mesh,
                &GlobalTransform::IDENTITY,
                &config,
                &pins,
                solver,
                Some(ClothLongRangeAttachments { compliance: 0.0 }),
                None,
            )
            .unwrap();
            assert_eq!(prepared.particles[2].inv_mass, 0.0);
            assert_eq!(prepared.particles[2].position, Vec3::Y.to_array());
            assert_eq!(prepared.particles[3].inv_mass, 0.0);
            assert_eq!(prepared.particles[0].inv_mass, 1.0);
            assert_eq!(prepared.particles[1].inv_mass, 1.0);
            assert!(prepared.bend_constraints[0].rest_angle.abs() < 1e-6);
            let paths: Vec<_> = prepared
                .tethers
                .iter()
                .map(|tether| (tether.particle, tether.anchor, tether.max_distance))
                .collect();
            // Both paths tie in the rest mesh, so the lower (dynamic) anchor wins.
            assert_eq!(paths, [(0, 2, 1.0), (1, 2, 2.0_f32.sqrt())]);
        }
    }

    #[test]
    fn rejects_invalid_dynamic_pins() {
        let mut config = config();
        config.attachments.push(FixedAttachment {
            particle: 0,
            position: Vec3::ZERO,
        });
        let pin = DynamicAttachment {
            particle: 1,
            target: Entity::PLACEHOLDER,
            local_offset: Vec3::X,
        };
        let validate =
            |pins: &[DynamicAttachment]| validate_attachments(4, &config, pins).unwrap_err();

        assert!(validate(&[DynamicAttachment { particle: 4, ..pin }]).contains("out of range"));
        assert!(
            validate(&[DynamicAttachment {
                local_offset: Vec3::NAN,
                ..pin
            }])
            .contains("finite")
        );
        assert!(validate(&[DynamicAttachment { particle: 0, ..pin }]).contains("multiple"));
        assert!(validate(&[pin, pin]).contains("multiple"));
    }

    #[test]
    fn invalid_bend_compliance_is_rejected() {
        let mesh = hinge_mesh();
        for compliance in [-1.0, f32::NAN, f32::INFINITY] {
            let mut config = config();
            config.bend_compliance = compliance;
            let error = prepare_mesh(
                &mesh,
                &GlobalTransform::IDENTITY,
                &config,
                &[],
                ClothSolver::default(),
                None,
                None,
            )
            .unwrap_err();
            assert!(error.contains("bend_compliance"));
        }
    }

    #[test]
    fn normal_adjacency_groups_oriented_pairs_by_vertex() {
        // Vertex 4 is isolated; its CSR range must be empty.
        let (offsets, pairs) = build_normal_adjacency(5, &[0, 1, 2, 0, 2, 3]);

        assert_eq!(offsets, vec![0, 2, 3, 5, 6, 6]);
        assert_eq!(pairs, vec![[1, 2], [2, 3], [2, 0], [0, 1], [3, 0], [0, 2]]);
    }
}

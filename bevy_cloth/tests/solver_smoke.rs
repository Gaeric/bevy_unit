//! solver wiring (schedule, `add_cloth`) plus the M2b acceptance targets.

use bevy_cloth::sim::{
    ClothError,
    collide::SdfCollider,
    mesh_gen::generate_cloth_mesh,
    params::SimParams,
    solver::{Phase, Solver},
};
use glam::{Mat4, Quat, Vec3};

const DT: f32 = 1.0 / 60.0;

fn close(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() < tolerance
}

fn iterations(repeat: usize) -> Vec<Phase> {
    let mut phases = Vec::new();
    for _ in 0..repeat {
        phases.extend([
            Phase::SolveStretch,
            Phase::SolveAttachment,
            Phase::SolveBending,
            Phase::ApplyDeltas,
        ]);
    }
    phases
}

/// scene 1 pose: position (0, 1.5, 1) with euler (90, 0, 0).
fn scene_1_pose() -> Mat4 {
    Mat4::from_translation(Vec3::new(0.0, 1.5, 1.0))
        * Mat4::from_quat(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
}

fn cloth(res: u32, params: &mut SimParams, transform: Mat4, attached: &[u32]) -> Solver {
    let mut solver = Solver::default();
    solver
        .add_cloth(params, &generate_cloth_mesh(res), transform, attached, DT)
        .expect("generated meshes are source-layout grids");
    solver
}

#[test]
fn schedule_matches_the_source() {
    let params = SimParams {
        num_substeps: 2,
        num_iterations: 4,
        interleaved_hash: 3,
        ..Default::default()
    };

    let mut expected = vec![
        Phase::StabilizeSdf,
        // substep 0: the hash is due
        Phase::Predict,
        Phase::RebuildHash,
        Phase::CollideParticles,
        Phase::CollideSdf,
    ];
    expected.extend(iterations(4));
    expected.push(Phase::Finalize);
    // substep 1
    expected.extend([Phase::Predict, Phase::CollideParticles, Phase::CollideSdf]);
    expected.extend(iterations(4));
    expected.push(Phase::Finalize);
    expected.push(Phase::ComputeNormals);

    assert_eq!(Solver::schedule(&params), expected);
}

#[test]
fn hash_is_rebuilt_once_per_interleaved_substeps() {
    let params = SimParams {
        num_substeps: 6,
        interleaved_hash: 3,
        ..Default::default()
    };
    let schedule = Solver::schedule(&params);

    let rebuilds: Vec<usize> = schedule
        .iter()
        .enumerate()
        .filter(|(_, phase)| **phase == Phase::RebuildHash)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(rebuilds.len(), 2, "substeps 0 and 3");

    // every rebuild is followed by the collision pass it feeds
    for index in rebuilds {
        assert_eq!(schedule[index + 1], Phase::CollideParticles);
    }
}

#[test]
fn self_collision_is_skipped_when_disabled() {
    let params = SimParams {
        enable_self_collision: false,
        ..Default::default()
    };
    let schedule = Solver::schedule(&params);

    assert!(!schedule.contains(&Phase::RebuildHash));
    assert!(!schedule.contains(&Phase::CollideParticles));
}

#[test]
fn zero_interleaved_hash_does_not_divide_by_zero() {
    let params = SimParams {
        interleaved_hash: 0,
        ..Default::default()
    };

    assert!(Solver::schedule(&params).contains(&Phase::RebuildHash));
}

#[test]
fn add_cloth_bookkeeping() {
    let mesh = generate_cloth_mesh(2);
    let mut params = SimParams::default();
    let mut solver = Solver::default();

    let offset = solver.add_cloth(&mut params, &mesh, scene_1_pose(), &[0], DT);
    assert_eq!(offset, Ok(0));
    assert_eq!(solver.num_particles(), 9);
    assert_eq!(solver.indices, mesh.indices);

    // world space: the pose rotates x by 90 degrees and translates by (0, 1.5, 1)
    assert!(close(solver.positions[0].x, -1.0, 1e-6));
    assert!(close(solver.positions[0].y, 1.5, 1e-6));
    assert!(close(solver.positions[0].z, 1.0, 1e-6));
    assert!(close(solver.positions[3].z, 0.0, 1e-6));
    assert_eq!(solver.initial_positions, solver.positions);

    // the diameter is measured on the local mesh: |p0 - p1| = cloth_size / res
    assert!(close(params.particle_diameter, 1.0 * 1.5, 1e-6));
    assert_eq!(params.num_particles, 9);
    assert!(close(
        params.max_speed,
        2.0 * params.particle_diameter / DT * 2.0,
        1e-3
    ));
    assert_eq!(solver.hash.num_particles, 9);

    // rest lengths and slots live in world space
    let stretch = &solver.constraints.stretch;
    assert_eq!(stretch.lengths.len(), 20);
    for (pair, rest) in stretch.indices.iter().zip(&stretch.lengths) {
        let want = (solver.positions[pair[0] as usize] - solver.positions[pair[1] as usize]).length();
        assert!(close(*rest, want, 1e-6), "{pair:?}");
    }
    assert_eq!(solver.constraints.attach.slot_positions.len(), 1);
    assert_eq!(
        solver.constraints.attach.slot_positions[0],
        solver.positions[0]
    );
    assert_eq!(solver.constraints.inv_masses[0], 0.0);
    assert_eq!(solver.constraints.inv_masses[1], 1.0);
}

#[test]
fn a_second_cloth_is_shifted_into_solver_space() {
    let mesh = generate_cloth_mesh(2);
    let mut params = SimParams::default();
    let mut solver = Solver::default();

    assert_eq!(
        solver.add_cloth(&mut params, &mesh, Mat4::IDENTITY, &[0], DT),
        Ok(0)
    );
    // a scale makes an unshifted rest length disagree with the world distance
    let scaled = Mat4::from_scale(Vec3::splat(2.0));
    assert_eq!(solver.add_cloth(&mut params, &mesh, scaled, &[0, 4], DT), Ok(9));

    assert_eq!(solver.num_particles(), 18);
    assert_eq!(params.num_particles, 18);
    assert_eq!(solver.hash.num_particles, 18);
    assert_eq!(solver.indices.len(), 2 * mesh.indices.len());
    assert_eq!(solver.indices[mesh.indices.len()], mesh.indices[0] + 9);

    let stretch = &solver.constraints.stretch;
    assert_eq!(stretch.lengths.len(), 40);
    for (pair, rest) in stretch.indices.iter().zip(&stretch.lengths) {
        let want = (solver.positions[pair[0] as usize] - solver.positions[pair[1] as usize]).length();
        assert!(close(*rest, want, 1e-5), "{pair:?}");
    }

    // slots accumulate: 1 from the first cloth, 2 from the second, and the second cloth's slot ids
    // are shifted by the first cloth's slot count
    let attach = &solver.constraints.attach;
    assert_eq!(attach.slot_positions.len(), 3);
    assert_eq!(attach.slot_positions[1], solver.positions[9]);
    assert_eq!(attach.slot_positions[2], solver.positions[13]);
    assert_eq!(attach.particle_ids.len(), 27);
    for (row, (particle, slot)) in attach.particle_ids.iter().zip(&attach.slot_ids).enumerate() {
        let want_slot = (row / 9) as u32;
        let base = if row < 9 { 0 } else { 9 };
        assert_eq!(*slot, want_slot, "row {row}");
        assert_eq!(*particle, (base + row % 9) as u32, "row {row}");
    }

    for (i, mass) in solver.constraints.inv_masses.iter().enumerate() {
        let pinned = i == 0 || i == 9 || i == 13;
        assert_eq!(*mass, if pinned { 0.0 } else { 1.0 }, "particle {i}");
    }
}

#[test]
fn add_cloth_rejects_a_mesh_that_is_not_a_source_layout_grid() {
    let mut mesh = generate_cloth_mesh(1);
    mesh.indices = vec![0, 1, 2, 2, 1, 3];

    let mut params = SimParams::default();
    let mut solver = Solver::default();
    assert_eq!(
        solver.add_cloth(&mut params, &mesh, Mat4::IDENTITY, &[], DT),
        Err(ClothError::IndexLayout { first_mismatch: 1 })
    );

    // a rejected cloth leaves the solver untouched
    assert_eq!(solver.num_particles(), 0);
    assert_eq!(solver.indices.len(), 0);
    assert_eq!(params.num_particles, 0);
}

// ---------------------------------------------------------------------------
// M2b acceptance targets: these need the [USER] kernel bodies.
// run them with `cargo test -p bevy_cloth -- --ignored --nocapture` and record the numbers.
// ---------------------------------------------------------------------------

/// semi-implicit euler moves `g * dt^2 * n(n+1) / 2`, not `g * t^2 / 2`: the two differ by `g*dt*t/2`.
#[test]
#[ignore = "M2b: needs the user's kernels"]
fn free_fall_matches_the_discrete_integration() {
    let mut params = SimParams {
        num_substeps: 1,
        num_iterations: 0,
        gravity: Vec3::new(0.0, -9.8, 0.0),
        damping: 0.0,
        enable_self_collision: false,
        ..Default::default()
    };
    let mut solver = cloth(2, &mut params, Mat4::IDENTITY, &[]);
    let start = solver.positions.clone();

    let steps = 30;
    for _ in 0..steps {
        solver.step(&params, &[], DT);
    }

    let fell = start[0].y - solver.positions[0].y;
    let expected = 0.5 * params.gravity.y * DT * DT * (steps * (steps + 1)) as f32;
    println!("free fall: {fell} (expected {expected})");
    assert!(close(fell, expected, 1e-4));
}

#[test]
#[ignore = "M2b: needs the user's kernels"]
fn no_gravity_keeps_the_cloth_still() {
    let mut params = SimParams {
        gravity: Vec3::ZERO,
        damping: 0.0,
        enable_self_collision: false,
        ..Default::default()
    };
    let mut solver = cloth(2, &mut params, Mat4::IDENTITY, &[]);
    let start = solver.positions.clone();

    for _ in 0..10 {
        solver.step(&params, &[], DT);
    }

    for (i, (now, before)) in solver.positions.iter().zip(&start).enumerate() {
        assert!((*now - *before).length() < 1e-5, "particle {i}");
    }
    assert!(solver.velocities.iter().all(|v| v.length() < 1e-5));
}

#[test]
#[ignore = "M2b: needs the user's kernels"]
fn plane_holds_the_cloth_at_the_collision_margin() {
    let mut params = SimParams::default();
    let mut solver = cloth(4, &mut params, Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)), &[]);
    let plane = SdfCollider::plane(0.0, DT);

    for _ in 0..120 {
        solver.step(&params, &[plane], DT);
    }

    let lowest = solver.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    println!("lowest y {lowest} (margin {})", params.collision_margin);
    assert!(close(lowest, params.collision_margin, 1e-3));
    assert!(solver.positions.iter().all(|p| p.is_finite()));
}

#[test]
#[ignore = "M2b: needs the user's kernels"]
fn sphere_push_out_is_radial() {
    let mut params = SimParams {
        num_substeps: 1,
        gravity: Vec3::ZERO,
        damping: 0.0,
        friction: 0.0,
        enable_self_collision: false,
        ..Default::default()
    };
    let mut solver = cloth(2, &mut params, Mat4::IDENTITY, &[]);
    let center = Vec3::new(0.0, -0.5, 0.0);
    let radius = 0.5;
    let before = solver.positions.clone();

    solver.step(&params, &[SdfCollider::sphere(center, radius, DT)], DT);

    for (i, (now, was)) in solver.positions.iter().zip(&before).enumerate() {
        let distance = (*now - center).length();
        assert!(
            distance >= radius + params.collision_margin - 1e-4,
            "particle {i} is inside the sphere: {distance}"
        );

        let was_distance = (*was - center).length();
        if was_distance < radius + params.collision_margin {
            let radial = (*was - center).normalize();
            let moved = *now - *was;
            let tangential = moved - moved.dot(radial) * radial;
            println!("particle {i}: {was_distance} -> {distance}, tangential {tangential:?}");
            assert!(tangential.length() < 1e-3, "particle {i} drifted sideways");
        }
    }
}

#[test]
#[ignore = "M2b: needs the user's kernels"]
fn max_speed_clamps_the_velocity() {
    let mut params = SimParams {
        num_substeps: 1,
        num_iterations: 1,
        gravity: Vec3::new(0.0, -2000.0, 0.0),
        damping: 0.0,
        enable_self_collision: false,
        ..Default::default()
    };
    let mut solver = cloth(2, &mut params, Mat4::IDENTITY, &[]);

    for _ in 0..30 {
        solver.step(&params, &[], DT);
    }

    let fastest = solver.velocities.iter().map(|v| v.length()).fold(0.0, f32::max);
    println!("fastest {fastest} (clamp {})", params.max_speed);
    assert!(fastest <= params.max_speed + 1e-3);
    assert!(solver.positions.iter().all(|p| p.is_finite()));
}

/// the threshold is a guess: record the deviations and tighten it later.
#[test]
#[ignore = "M2b: needs the user's kernels"]
fn substeps_barely_change_the_result() {
    let mut results = Vec::new();

    for substeps in [1u32, 2, 4] {
        let mut params = SimParams {
            num_substeps: substeps,
            ..Default::default()
        };
        let mut solver = cloth(
            4,
            &mut params,
            Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            &[],
        );
        let plane = SdfCollider::plane(0.0, DT);
        for _ in 0..120 {
            solver.step(&params, &[plane], DT);
        }
        results.push((substeps, solver.positions));
    }

    let base = &results[1].1;
    for (substeps, positions) in &results {
        let deviation = positions
            .iter()
            .zip(base)
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0, f32::max);
        println!("substeps {substeps}: max deviation from substeps 2 is {deviation}");
        assert!(deviation < 0.05);
    }
}

/// the corner is pinned by a zero rest distance (`inv_mass == 0`), so the chain hangs on its
/// stretch lengths.
#[test]
#[ignore = "M2b: needs the user's kernels"]
fn a_pinned_corner_hangs_at_the_rest_length() {
    let mut params = SimParams::default();
    let mut solver = cloth(1, &mut params, Mat4::IDENTITY, &[0]);
    let pinned = solver.positions[0];
    let rest = solver.constraints.stretch.lengths[0];

    for _ in 0..240 {
        solver.step(&params, &[], DT);
    }

    let distance = (solver.positions[0] - solver.positions[1]).length();
    println!("|p0 - p1| {distance} (rest {rest})");
    assert!(close(distance, rest, 5e-2));
    assert!((solver.positions[0] - pinned).length() < 1e-6, "pinned corner moved");
    assert!(solver.positions.iter().all(|p| p.is_finite()));
}

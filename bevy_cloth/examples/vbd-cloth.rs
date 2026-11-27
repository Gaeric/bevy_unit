//! Headless GPU probe for VBD inertia, elasticity, static contact and material damping.

use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        mesh::{
            RenderMesh,
            allocator::{MeshAllocator, MeshAllocatorSettings},
        },
        pipelined_rendering::PipelinedRenderingPlugin,
        render_asset::RenderAssets,
        render_resource::{
            BufferDescriptor, BufferUsages, CachedPipelineState, CommandEncoderDescriptor, MapMode,
            PipelineCache, PipelineDescriptor, PollType, VertexFormat,
        },
        renderer::{RenderDevice, RenderQueue},
    },
    time::TimeUpdateStrategy,
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use bevy_cloth::{
    Cloth, ClothDynamicAttachments, ClothPlugin, ClothSolver, ClothStepSettings, ClothVbdContact,
    ClothVbdMaterial, DynamicAttachment, FixedAttachment, VbdStaticCollider,
};
use std::{
    thread,
    time::{Duration, Instant},
};

const STEP: Duration = Duration::from_micros(62_500); // h = 1/16 s, m/h² = 256
const EPSILON: f32 = 2e-5;
const POSITION_BYTES: u64 = 12;

fn main() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .set(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins(ClothPlugin)
    .insert_resource(TimeUpdateStrategy::FixedTimesteps(0));
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .set_timestep(STEP);
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<MeshAllocatorSettings>()
        .extra_buffer_usages |= BufferUsages::COPY_SRC;

    let deadline = Instant::now() + Duration::from_secs(10);
    while app.plugins_state() == PluginsState::Adding {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for render device"
        );
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();

    let world = app.world_mut();
    let triangle = [Vec3::ZERO, Vec3::X, Vec3::Y];
    let (_, inertia) = spawn_case(
        world,
        &triangle,
        &[0, 1, 2],
        &[(0, -Vec3::X)],
        Vec3::X * 64.0,
        ClothVbdMaterial::default(),
        2,
    );
    let (_, membrane) = spawn_case(
        world,
        &triangle,
        &[0, 1, 2],
        &[(0, Vec3::ZERO), (2, Vec3::Y)],
        Vec3::X * 64.0,
        ClothVbdMaterial {
            shear: 128.0,
            area: 128.0,
            ..default()
        },
        2,
    );
    let (_, hinge) = spawn_case(
        world,
        &[Vec3::ZERO, Vec3::X, Vec3::Y, -Vec3::Y],
        &[2, 0, 1, 3, 1, 0],
        &[(0, Vec3::ZERO), (1, Vec3::X)],
        -Vec3::Z * 64.0,
        ClothVbdMaterial {
            bend: 256.0,
            ..default()
        },
        1,
    );

    // A single triangle spanning both test vertices, with outward normal -X.
    let plane = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.875, -2.0, -2.0], [0.875, -2.0, 3.0], [0.875, 3.0, -2.0]],
    );
    let plane = world.resource_mut::<Assets<Mesh>>().add(plane);
    world.spawn((Mesh3d(plane), VbdStaticCollider));
    let (contact_entity, contact) = spawn_case(
        world,
        &[Vec3::ZERO, 0.75 * Vec3::X, Vec3::Y],
        &[0, 1, 2],
        &[(0, Vec3::ZERO), (2, Vec3::Y)],
        64.0 * Vec3::X,
        ClothVbdMaterial::default(),
        2,
    );
    let (coupled_entity, coupled) = spawn_case(
        world,
        &triangle,
        &[0, 1, 2],
        &[(0, Vec3::ZERO), (2, Vec3::Y)],
        -64.0 * Vec3::X,
        ClothVbdMaterial {
            shear: 128.0,
            area: 128.0,
            ..default()
        },
        2,
    );
    for entity in [contact_entity, coupled_entity] {
        let mut entity = world.entity_mut(entity);
        entity.get_mut::<Cloth>().unwrap().collision_margin = 0.0625;
        entity.insert(ClothVbdContact { stiffness: 256.0 });
    }

    // Pure damping must not resist a rigid translation or a finite rotation.
    let shell = [Vec3::ZERO, Vec3::X, Vec3::Y, -Vec3::Y];
    let shell_indices = [2, 0, 1, 3, 1, 0];
    let damping = ClothVbdMaterial {
        membrane_damping: 4.0,
        bend_damping: 16.0,
        ..default()
    };
    let (_, translation) = spawn_case(
        world,
        &shell,
        &shell_indices,
        &[],
        64.0 * Vec3::Z,
        damping,
        2,
    );
    let (rotation_entity, rotation) = spawn_case(
        world,
        &shell,
        &shell_indices,
        &[],
        256.0 * (Vec3::Z - Vec3::Y),
        damping,
        2,
    );
    let mut targets = Vec::new();
    for particle in [0, 1, 3] {
        let target = world
            .spawn(Transform::from_translation(shell[particle]))
            .id();
        targets.push((particle, target));
    }
    world
        .entity_mut(rotation_entity)
        .insert(ClothDynamicAttachments(
            targets
                .iter()
                .map(|&(particle, target)| DynamicAttachment {
                    particle: particle as u32,
                    target,
                    local_offset: Vec3::ZERO,
                })
                .collect(),
        ));

    let (_, bending_rate) = spawn_case(
        world,
        &shell,
        &shell_indices,
        &[(0, Vec3::ZERO), (1, Vec3::X), (3, -Vec3::Y)],
        -64.0 * Vec3::Z,
        ClothVbdMaterial {
            bend_damping: 16.0,
            ..default()
        },
        1,
    );

    // Both pins shift to x = 1/8, so equilibrium is at x = 9/8. The free
    // vertex starts compressed at x = 1, with zero velocity and no external load.
    let release = |world: &mut World, membrane_damping| {
        spawn_case(
            world,
            &triangle,
            &[0, 1, 2],
            &[(0, 0.125 * Vec3::X), (2, Vec3::new(0.125, 1.0, 0.0))],
            Vec3::ZERO,
            ClothVbdMaterial {
                shear: 128.0,
                area: 128.0,
                membrane_damping,
                ..default()
            },
            6,
        )
        .1
    };
    let undamped = release(world, 0.0);
    let damped = release(world, 4.0);

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        app.update();
        // Use the moved hard pin to distinguish actual Mesh writeback from upload.
        if pipelines_ready(&app)
            && read_positions(&app, &inertia, 3).is_some_and(|p| p[0] == -Vec3::X)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for VBD pipelines and Mesh output"
        );
        thread::sleep(Duration::from_millis(1));
    }

    let rotate = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    for (particle, target) in targets {
        app.world_mut()
            .get_mut::<Transform>(target)
            .unwrap()
            .translation = rotate * shell[particle];
    }
    // The pin samples are taken in PostUpdate. Submit them before the first tick.
    app.update();
    app.insert_resource(TimeUpdateStrategy::FixedTimesteps(1));
    app.update();
    check(
        &app,
        "inertia, tick 1",
        &inertia,
        &[-Vec3::X, 1.25 * Vec3::X, Vec3::Y + 0.25 * Vec3::X],
    );
    // At the predictor x = 1.25 the force is -48 and Hxx = 256 + 192.
    // The second sweep stays at the minimizer only if y stays at 1.25.
    check(
        &app,
        "membrane",
        &membrane,
        &[Vec3::ZERO, (8.0 / 7.0) * Vec3::X, Vec3::Y],
    );
    // Both free corners predict z = -0.25. The second corner's block uses
    // the angle after the first corner moved, not the original 2*atan(0.25).
    check(
        &app,
        "hinge",
        &hinge,
        &[
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(0.0, 1.0593888, -0.0124449),
            Vec3::new(0.0, -1.0311182, -0.1255271),
        ],
    );

    // h = 1/16, m/h² = k = 256. The target center is x = 0.875 - 0.0625.
    // Starting at predicted x = 1, half the compression remains after solving.
    check(
        &app,
        "contact",
        &contact,
        &[Vec3::ZERO, 0.90625 * Vec3::X, Vec3::Y],
    );
    // Predicted x = 0.75 is outside the thickness. The first membrane block
    // moves to x = 6/7, then the next sweep adds contact to the 448 N/m block.
    check(
        &app,
        "contact and membrane",
        &coupled,
        &[Vec3::ZERO, (37.0 / 44.0) * Vec3::X, Vec3::Y],
    );

    check(
        &app,
        "rigid translation",
        &translation,
        &shell.map(|x| x + 0.25 * Vec3::Z),
    );
    check(
        &app,
        "rigid rotation",
        &rotation,
        &shell.map(|x| rotate * x),
    );
    // With only hinge damping, c/h = 256. One Gauss–Newton block at
    // (0, 1, -1/4) corrects along the current dihedral gradient.
    check(
        &app,
        "dihedral rate",
        &bending_rate,
        &[
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(0.0, 1.0296944, -0.1312225),
            -Vec3::Y,
        ],
    );

    app.update();
    check(
        &app,
        "inertia, tick 2",
        &inertia,
        &[-Vec3::X, 1.75 * Vec3::X, Vec3::Y + 0.75 * Vec3::X],
    );
    for _ in 0..3 {
        app.update();
    }
    let undamped_fifth = free_x(&app, &undamped);
    let damped_fifth = free_x(&app, &damped);
    app.update();
    let undamped_energy = oscillation_energy(undamped_fifth, free_x(&app, &undamped));
    let damped_energy = oscillation_energy(damped_fifth, free_x(&app, &damped));
    assert!(
        (0.05..0.055).contains(&undamped_energy) && damped_energy < 0.5 * undamped_energy,
        "expected faster decay with damping, energies: {undamped_energy} and {damped_energy}"
    );
    println!("VBD cloth GPU probe passed");
}

fn free_x(app: &App, mesh: &Handle<Mesh>) -> f32 {
    read_positions(app, mesh, 3).expect("oscillation output mesh")[1].x
}

fn oscillation_energy(previous_x: f32, x: f32) -> f32 {
    let velocity = (x - previous_x) / STEP.as_secs_f32();
    // Along this axis the membrane is a spring with stiffness 192 N/m.
    0.5 * velocity * velocity + 96.0 * (x - 1.125).powi(2)
}

fn pipelines_ready(app: &App) -> bool {
    let cache = app.sub_app(RenderApp).world().resource::<PipelineCache>();
    [
        "predict_vbd",
        "solve_vbd_color",
        "reset_vbd_contacts",
        "gather_vbd_contacts",
        "xpbd collider transform_vertices",
        "xpbd collider refit_leaves",
        "xpbd collider refit_internal",
        "xpbd update_velocity",
        "xpbd output_mesh",
    ]
    .into_iter()
    .all(|label| {
        cache.pipelines().any(|entry| {
            let PipelineDescriptor::ComputePipelineDescriptor(descriptor) = &entry.descriptor
            else {
                return false;
            };
            descriptor.label.as_deref() == Some(label)
                && matches!(entry.state, CachedPipelineState::Ok(_))
        })
    })
}

fn spawn_case(
    world: &mut World,
    points: &[Vec3],
    indices: &[u32],
    pins: &[(u32, Vec3)],
    gravity: Vec3,
    material: ClothVbdMaterial,
    iterations: u32,
) -> (Entity, Handle<Mesh>) {
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        points.iter().map(|p| p.to_array()).collect::<Vec<_>>(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; points.len()])
    .with_inserted_indices(Indices::U32(indices.to_vec()));
    let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh);
    let entity = world
        .spawn((
            Mesh3d(mesh.clone()),
            Cloth {
                particle_mass: 1.0,
                stretch_compliance: 0.0,
                bend_compliance: 0.0,
                attachments: pins
                    .iter()
                    .map(|&(particle, position)| FixedAttachment { particle, position })
                    .collect(),
                collision_margin: 0.0,
                double_sided: false,
            },
            ClothSolver::Vbd,
            material,
            ClothStepSettings {
                gravity,
                substeps: 1,
                iterations,
            },
        ))
        .id();
    (entity, mesh)
}

fn check(app: &App, label: &str, mesh: &Handle<Mesh>, expected: &[Vec3]) {
    let actual = read_positions(app, mesh, expected.len()).expect("cloth output mesh");
    for (vertex, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.distance(expected) < EPSILON,
            "{label}, vertex {vertex}: expected {expected:?}, got {actual:?}"
        );
    }
}

fn read_positions(app: &App, mesh: &Handle<Mesh>, count: usize) -> Option<Vec<Vec3>> {
    let world = app.sub_app(RenderApp).world();
    let render_mesh = world
        .resource::<RenderAssets<RenderMesh>>()
        .get(mesh.id())?;
    let allocation = world
        .resource::<MeshAllocator>()
        .mesh_vertex_slice(&mesh.id())?;
    let layout = render_mesh
        .layout
        .0
        .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])
        .expect("probe position layout");
    assert_eq!(layout.attributes[0].format, VertexFormat::Float32x3);
    let first =
        u64::from(allocation.range.start) * layout.array_stride + layout.attributes[0].offset;

    let device = world.resource::<RenderDevice>();
    let queue = world.resource::<RenderQueue>();
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("VBD probe readback"),
        size: count as u64 * POSITION_BYTES,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    for vertex in 0..count as u64 {
        encoder.copy_buffer_to_buffer(
            allocation.buffer,
            first + vertex * layout.array_stride,
            &staging,
            vertex * POSITION_BYTES,
            POSITION_BYTES,
        );
    }
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(MapMode::Read, |result| result.expect("map cloth vertices"));
    device
        .poll(PollType::wait_indefinitely())
        .expect("wait for cloth vertices");
    let bytes = slice.get_mapped_range().expect("mapped cloth vertices");
    let positions = bytes
        .chunks_exact(POSITION_BYTES as usize)
        .map(|bytes| Vec3::from_array(bytemuck::pod_read_unaligned(bytes)))
        .collect();
    drop(bytes);
    staging.unmap();
    Some(positions)
}

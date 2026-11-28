//! Headless XPBD check: move a hidden collider's triangle with one joint.

use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    mesh::{
        PrimitiveTopology, VertexAttributeValues,
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    },
    prelude::*,
    render::{
        RenderApp,
        mesh::{
            RenderMesh,
            allocator::{MeshAllocator, MeshAllocatorSettings},
        },
        pipelined_rendering::PipelinedRenderingPlugin,
        render_asset::RenderAssets,
        render_resource::{
            BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode, PollType,
            VertexFormat,
        },
        renderer::{RenderDevice, RenderQueue},
    },
    time::TimeUpdateStrategy,
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use bevy_cloth::{
    Cloth, ClothExternalFriction, ClothPlugin, ClothStepSettings, CollisionGroup, MeshCollider,
    compute_skinning::{ComputeSkinning, ComputeSkinningPlugin, ComputeSkinningResults},
    debug::{
        ClothCapture, ClothCaptureId, ClothDebug, ClothDebugPlugin, ClothDiagnostics,
        xpbd::PositionTrace,
    },
};
use std::{
    thread,
    time::{Duration, Instant},
};

const START_Z: f32 = 0.02;
const MARGIN: f32 = 0.1;
const EPSILON: f32 = 1e-5;
const POSITION_BYTES: u64 = 12;

fn main() {
    let capture_enabled = std::env::args().any(|arg| arg == "--capture");
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins((ComputeSkinningPlugin, ClothPlugin))
    .insert_resource(TimeUpdateStrategy::FixedTimesteps(1));
    if capture_enabled {
        app.add_plugins(ClothDebugPlugin);
    }
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<MeshAllocatorSettings>()
        .extra_buffer_usages |= BufferUsages::COPY_SRC;

    while app.plugins_state() == PluginsState::Adding {
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();

    let (collider, tip) = spawn_collider(app.world_mut());
    let (_, initial_probe) = spawn_probe(app.world_mut(), 2.2, 0.2, 0);
    let (bent_probe_entity, bent_probe) = spawn_probe(app.world_mut(), 0.8, 1.2, 0);
    let (old_probe_entity, old_probe) = spawn_probe(app.world_mut(), 2.2, 0.2, 1);

    let deadline = Instant::now() + Duration::from_secs(10);
    let initial_version = loop {
        app.update();
        // Wait for both the simulation and the GPU mesh output to be ready.
        if let Some(version) = pose_version(&app, collider)
            && probe_z(&app, &initial_probe).is_some_and(|z| (z - MARGIN).abs() < EPSILON)
        {
            break version;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for bind-pose contact"
        );
        thread::sleep(Duration::from_millis(1));
    };
    check_z(&app, "bent location before rotation", &bent_probe, START_Z);
    check_z(&app, "isolated old-location probe", &old_probe, START_Z);

    // Keep the sample pose unchanged while the sampler pipeline warms up.
    if capture_enabled {
        let warmup = request_capture(&mut app, old_probe_entity);
        let capture = wait_capture(&mut app, warmup);
        let trace = xpbd_trace(&capture);
        assert_z(
            "isolated before constraints",
            trace.substeps[0].before_constraints[0].z,
            START_Z,
        );
        assert_z(
            "isolated after constraints",
            trace.substeps[0].after_constraints[0].z,
            START_Z,
        );
    }
    let rotation_capture = capture_enabled.then(|| request_capture(&mut app, bent_probe_entity));

    app.world_mut()
        .get_mut::<Transform>(tip)
        .expect("tip joint")
        .rotate_z(std::f32::consts::FRAC_PI_2);
    app.world_mut()
        .entity_mut(old_probe_entity)
        .insert(CollisionGroup(0));
    app.update();

    let bent_version = pose_version(&app, collider).expect("updated skinning output");
    assert!(
        bent_version > initial_version,
        "the child joint changed the pose"
    );
    check_z(
        &app,
        "bent location on the rotation frame",
        &bent_probe,
        MARGIN,
    );
    check_z(
        &app,
        "old location on the rotation frame",
        &old_probe,
        START_Z,
    );
    if let Some(request) = rotation_capture {
        let capture = wait_capture(&mut app, request);
        assert_eq!(capture.cloth, bent_probe_entity);
        assert_eq!(capture.particle_ids, [0]);
        let trace = xpbd_trace(&capture);
        let substep = &trace.substeps[0];
        assert_z(
            "before constraints",
            substep.before_constraints[0].z,
            MARGIN,
        );
        assert_z("after constraints", substep.after_constraints[0].z, MARGIN);
        println!("XPBD position capture passed (tick {})", capture.tick);
    }
    println!(
        "XPBD skinned collider probe passed (pose versions {initial_version} -> {bent_version})"
    );
}

fn request_capture(app: &mut App, entity: Entity) -> ClothCaptureId {
    app.world_mut()
        .resource_mut::<ClothDebug>()
        .capture_next_tick(entity, &[0])
        .expect("request one cloth tick")
}

fn wait_capture(app: &mut App, id: ClothCaptureId) -> ClothCapture {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(result) = app.world_mut().resource_mut::<ClothDebug>().take_result(id) {
            return result.expect("cloth tick capture");
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for cloth capture"
        );
        app.update();
        thread::sleep(Duration::from_millis(1));
    }
}

fn xpbd_trace(capture: &ClothCapture) -> &PositionTrace {
    match &capture.diagnostics {
        ClothDiagnostics::Xpbd(trace) => trace,
        other => panic!("expected XPBD diagnostics: {other:?}"),
    }
}

fn assert_z(label: &str, z: f32, expected: f32) {
    assert!(
        (z - expected).abs() < EPSILON,
        "{label}: expected z = {expected}, got {z}"
    );
}

fn spawn_collider(world: &mut World) -> (Entity, Entity) {
    let root = world.spawn(Transform::IDENTITY).id();
    let tip = world
        .spawn((Transform::from_translation(Vec3::X), ChildOf(root)))
        .id();
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            // Nearer than the moving triangle's old bounds, but outside contact
            // range. A stale BVH can choose this leaf and prune the moving one.
            [0.6, 1.0, -0.2],
            [1.3, 1.0, -0.2],
            [0.6, 1.7, -0.2],
        ],
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_JOINT_INDEX,
        VertexAttributeValues::Uint16x4(vec![
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, vec![[1.0, 0.0, 0.0, 0.0]; 6]);
    let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh);
    let inverse_bindposes = world
        .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
        .add(SkinnedMeshInverseBindposes::from(vec![
            Mat4::IDENTITY,
            Mat4::from_translation(-Vec3::X),
        ]));
    let collider = world
        .spawn((
            Mesh3d(mesh),
            SkinnedMesh {
                inverse_bindposes,
                joints: vec![root, tip],
            },
            // Positions are already in world space when passed to the collider.
            Transform::from_xyz(100.0, 0.0, 0.0),
            Visibility::Hidden,
            ComputeSkinning,
            MeshCollider,
        ))
        .id();
    (collider, tip)
}

fn spawn_probe(world: &mut World, x: f32, y: f32, group: u32) -> (Entity, Handle<Mesh>) {
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [x, y, START_Z],
            [x + 0.05, y, START_Z],
            [x, y + 0.05, START_Z],
        ],
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3]);
    let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh);
    let entity = world
        .spawn((
            Mesh3d(mesh.clone()),
            Transform::IDENTITY,
            CollisionGroup(group),
            Cloth {
                particle_mass: 1.0,
                stretch_compliance: 0.0,
                bend_compliance: 0.0,
                attachments: Vec::new(),
                collision_margin: MARGIN,
                double_sided: true,
            },
            ClothStepSettings {
                gravity: Vec3::ZERO,
                substeps: 1,
                iterations: 1,
            },
            // Skinned contact should use the normal-only pipeline despite this setting.
            ClothExternalFriction { coefficient: 0.4 },
        ))
        .id();
    (entity, mesh)
}

fn pose_version(app: &App, entity: Entity) -> Option<u64> {
    let world = app.sub_app(RenderApp).world();
    Some(
        world
            .resource::<ComputeSkinningResults>()
            .get(entity)?
            .pose_version(),
    )
}

fn check_z(app: &App, label: &str, mesh: &Handle<Mesh>, expected: f32) {
    let z = probe_z(app, mesh).expect("cloth output mesh");
    assert_z(label, z, expected);
}

// Read only the first vertex position from the GPU-written cloth mesh. The
// simulation itself never copies or reads back the skinned collider positions.
fn probe_z(app: &App, mesh: &Handle<Mesh>) -> Option<f32> {
    let world = app.sub_app(RenderApp).world();
    let allocator = world.resource::<MeshAllocator>();
    let meshes = world.resource::<RenderAssets<RenderMesh>>();
    let render_mesh = meshes.get(mesh.id())?;
    let slice = allocator.mesh_vertex_slice(&mesh.id())?;
    let layout = render_mesh
        .layout
        .0
        .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])
        .expect("probe position layout");
    let attribute = layout.attributes[0];
    assert_eq!(attribute.format, VertexFormat::Float32x3);
    let offset = u64::from(slice.range.start) * layout.array_stride + attribute.offset;

    let device = world.resource::<RenderDevice>();
    let queue = world.resource::<RenderQueue>();
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("XPBD skinned collider probe readback"),
        size: POSITION_BYTES,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(slice.buffer, offset, &staging, 0, POSITION_BYTES);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(MapMode::Read, |result| result.expect("map cloth vertex"));
    device
        .poll(PollType::wait_indefinitely())
        .expect("wait for cloth vertex");
    let bytes = slice.get_mapped_range().expect("mapped cloth vertex");
    let z = bytemuck::pod_read_unaligned::<[f32; 3]>(&bytes)[2];
    drop(bytes);
    staging.unmap();
    Some(z)
}

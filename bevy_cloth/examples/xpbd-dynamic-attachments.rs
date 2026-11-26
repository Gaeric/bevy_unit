//! Headless XPBD check: rotate a child joint driving one cloth pin.

use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    mesh::PrimitiveTopology,
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
    Cloth, ClothDynamicAttachments, ClothPlugin, ClothStepSettings, DynamicAttachment,
    FixedAttachment,
};
use std::{
    thread,
    time::{Duration, Instant},
};

const FIXED: Vec3 = Vec3::new(2.0, 0.0, 0.0);
const PIN_REST: Vec3 = Vec3::new(4.25, 0.0, 0.0);
const FREE_REST: Vec3 = Vec3::new(4.25, 0.0, 1.0);
const EPSILON: f32 = 1e-5;
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
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins(ClothPlugin)
    .insert_resource(TimeUpdateStrategy::FixedTimesteps(1));
    app.sub_app_mut(RenderApp)
        .world_mut()
        .resource_mut::<MeshAllocatorSettings>()
        .extra_buffer_usages |= BufferUsages::COPY_SRC;

    while app.plugins_state() == PluginsState::Adding {
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    let (tip, mesh) = spawn_patch(app.world_mut());

    let deadline = Instant::now() + Duration::from_secs(10);
    let before = loop {
        app.update();
        // The initial target is offset from the rest pin. The free vertex only
        // moves when XPBD has run, not when the source mesh is first uploaded.
        if let Some(positions) = read_positions(&app, &mesh)
            && positions[2].distance(FREE_REST) > EPSILON
        {
            break positions;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the first cloth tick"
        );
        thread::sleep(Duration::from_millis(1));
    };
    check_position("fixed corner before rotation", before[0], FIXED);
    check_position(
        "dynamic corner before rotation",
        before[1],
        Vec3::new(4.0, 0.0, 0.0),
    );
    assert!(before[2].y.abs() < EPSILON);

    let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    app.world_mut()
        .get_mut::<Transform>(tip)
        .expect("tip joint")
        .rotation = rotation;
    app.update();

    let bent_target = Vec3::new(3.0, 1.0, 0.0);
    let after = read_positions(&app, &mesh).expect("cloth output mesh");
    check_position("fixed corner on the rotation frame", after[0], FIXED);
    check_position(
        "dynamic corner on the rotation frame",
        after[1],
        bent_target,
    );
    let pivot = Vec3::new(3.0, 0.0, 0.0);
    let rigidly_rotated_neighbor = pivot + rotation * (before[2] - pivot);
    assert!(
        after[2].y > 0.01 && after[2].distance(rigidly_rotated_neighbor) > 0.01,
        "free neighbor should be pulled by XPBD, not rigidly driven: {:?} -> {:?}",
        before[2],
        after[2]
    );

    assert!(app.world_mut().despawn(tip));
    app.update();
    let missing = read_positions(&app, &mesh).expect("cloth output mesh");
    check_position(
        "dynamic corner after losing the target",
        missing[1],
        bent_target,
    );
    println!("XPBD dynamic attachment probe passed");
}

fn spawn_patch(world: &mut World) -> (Entity, Handle<Mesh>) {
    let root = world.spawn(Transform::from_translation(FIXED)).id();
    let tip = world
        .spawn((Transform::from_translation(Vec3::X), ChildOf(root)))
        .id();
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        [FIXED, PIN_REST, FREE_REST].map(|p| p.to_array()).to_vec(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, -1.0, 0.0]; 3]);
    let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh);
    world.spawn((
        Mesh3d(mesh.clone()),
        Cloth {
            particle_mass: 1.0,
            stretch_compliance: 0.0,
            bend_compliance: 0.0,
            attachments: vec![FixedAttachment {
                particle: 0,
                position: FIXED,
            }],
            collision_margin: 0.0,
            double_sided: false,
        },
        ClothDynamicAttachments(vec![DynamicAttachment {
            particle: 1,
            target: tip,
            local_offset: Vec3::X,
        }]),
        ClothStepSettings {
            gravity: Vec3::ZERO,
            substeps: 1,
            iterations: 2,
        },
    ));
    (tip, mesh)
}

fn check_position(label: &str, actual: Vec3, expected: Vec3) {
    assert!(
        actual.distance(expected) < EPSILON,
        "{label}: expected {expected:?}, got {actual:?}"
    );
}

fn read_positions(app: &App, mesh: &Handle<Mesh>) -> Option<[Vec3; 3]> {
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
        label: Some("XPBD dynamic attachment probe readback"),
        size: 3 * POSITION_BYTES,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    for vertex in 0..3 {
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
    let positions: [[f32; 3]; 3] = bytemuck::pod_read_unaligned(&bytes);
    drop(bytes);
    staging.unmap();
    Some(positions.map(Vec3::from_array))
}

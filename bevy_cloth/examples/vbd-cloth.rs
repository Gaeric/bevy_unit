//! Headless GPU probe for VBD inertia, elasticity and static normal contact.

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
    Cloth, ClothPlugin, ClothSolver, ClothStepSettings, ClothVbdContact, ClothVbdMaterial,
    FixedAttachment, VbdStaticCollider,
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

    app.update();
    check(
        &app,
        "inertia, tick 2",
        &inertia,
        &[-Vec3::X, 1.75 * Vec3::X, Vec3::Y + 0.75 * Vec3::X],
    );
    println!("VBD cloth GPU probe passed");
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

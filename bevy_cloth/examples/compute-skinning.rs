//! Headless validation: an independent compute pass directly reads the skinning output.

use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    mesh::{
        VertexAttributeValues,
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    },
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{
            BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferDescriptor, BufferUsages, CachedComputePipelineId, CommandEncoderDescriptor,
            ComputePassDescriptor, ComputePipelineDescriptor, MapMode, PipelineCache, PollType,
            PrimitiveTopology, ShaderStages,
            binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    },
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use bevy_cloth::compute_skinning::{
    ComputeSkinning, ComputeSkinningPlugin, ComputeSkinningResults, ComputeSkinningSystems,
};
use std::{
    thread,
    time::{Duration, Instant},
};

const PROBE_BYTES: u64 = 48;
const PROBE_SHADER: &str = "
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sample: array<vec4<f32>>;

@compute @workgroup_size(1)
fn probe() {
    sample[0] = positions[0];
    sample[1] = positions[1];
    sample[2] = positions[2];
}
";

#[derive(Resource)]
struct ProbeInput {
    source: Entity,
    shader: Handle<Shader>,
}

#[derive(Resource)]
struct Probe {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    sample: Buffer,
    pose_version: Option<u64>,
}

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
    .add_plugins(ComputeSkinningPlugin);
    while app.plugins_state() == PluginsState::Adding {
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();

    let world = app.world_mut();
    let root = world.spawn(Transform::from_xyz(3.0, 0.0, 0.0)).id();
    let tip = world
        .spawn((Transform::from_xyz(1.0, 0.0, 0.0), ChildOf(root)))
        .id();
    let inverse_bindposes = world
        .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
        .add(SkinnedMeshInverseBindposes::from(vec![
            Mat4::IDENTITY,
            Mat4::from_translation(-Vec3::X),
        ]));
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_JOINT_INDEX,
        VertexAttributeValues::Uint16x4(vec![[0, 1, 0, 0]; 3]),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_JOINT_WEIGHT,
        vec![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
    );
    let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh);
    // Neither the mesh entity transform nor its visibility should affect the result.
    let source = world
        .spawn((
            Mesh3d(mesh),
            SkinnedMesh {
                inverse_bindposes,
                joints: vec![root, tip],
            },
            Transform::from_xyz(100.0, 0.0, 0.0),
            Visibility::Hidden,
            ComputeSkinning,
        ))
        .id();
    let shader = world
        .resource_mut::<Assets<Shader>>()
        .add(Shader::from_wgsl(
            PROBE_SHADER,
            "compute_skinning_probe.wgsl",
        ));
    app.sub_app_mut(RenderApp)
        .insert_resource(ProbeInput { source, shader })
        .add_systems(RenderStartup, init_probe)
        .add_systems(
            RenderGraph,
            probe_positions
                .in_set(RenderGraphSystems::Render)
                .after(ComputeSkinningSystems),
        );

    let initial = wait_for_probe(&mut app, None);
    check_sample(&app, [[3.0, 1.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
    app.update();
    assert_eq!(
        app.sub_app(RenderApp)
            .world()
            .resource::<Probe>()
            .pose_version,
        Some(initial)
    );

    app.world_mut()
        .get_mut::<Transform>(tip)
        .expect("tip transform")
        .rotate_z(std::f32::consts::FRAC_PI_2);
    app.update();
    let bent = app
        .sub_app(RenderApp)
        .world()
        .resource::<Probe>()
        .pose_version
        .expect("bent pose");
    assert!(
        bent > initial,
        "the probe must see the new pose on the first update after bending"
    );
    check_sample(&app, [[3.0, 1.0, 0.0], [4.0, 1.0, 0.0], [4.5, 0.5, 0.0]]);
    app.world_mut()
        .entity_mut(source)
        .remove::<ComputeSkinning>();
    app.update();
    assert!(
        app.sub_app(RenderApp)
            .world()
            .resource::<ComputeSkinningResults>()
            .get(source)
            .is_none()
    );
    println!("Compute skinning GPU direct-read probe passed (pose versions {initial} -> {bent})");
}

fn init_probe(
    mut commands: Commands,
    input: Res<ProbeInput>,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "skinning probe layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("skinning probe".into()),
        layout: vec![layout.clone()],
        shader: input.shader.clone(),
        entry_point: Some("probe".into()),
        ..default()
    });
    let sample = device.create_buffer(&BufferDescriptor {
        label: Some("skinning probe sample"),
        size: PROBE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    commands.insert_resource(Probe {
        layout,
        pipeline,
        sample,
        pose_version: None,
    });
}

fn probe_positions(
    mut probe: ResMut<Probe>,
    input: Res<ProbeInput>,
    results: Res<ComputeSkinningResults>,
    cache: Res<PipelineCache>,
    mut context: RenderContext,
) {
    let Some(pipeline) = cache.get_compute_pipeline(probe.pipeline) else {
        return;
    };
    let Some(output) = results.get(input.source) else {
        return;
    };
    if probe.pose_version == Some(output.pose_version()) {
        return;
    }
    assert_eq!(output.vertex_count(), 3);
    let bindings = context.render_device().create_bind_group(
        "skinning probe bindings",
        &cache.get_bind_group_layout(&probe.layout),
        &BindGroupEntries::sequential((
            output.buffer().as_entire_binding(),
            probe.sample.as_entire_binding(),
        )),
    );
    let mut pass = context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("read skinning output directly on GPU"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bindings, &[]);
    pass.dispatch_workgroups(1, 1, 1);
    probe.pose_version = Some(output.pose_version());
}

fn wait_for_probe(app: &mut App, previous: Option<u64>) -> u64 {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        app.update();
        if let Some(version) = app
            .sub_app(RenderApp)
            .world()
            .resource::<Probe>()
            .pose_version
            && Some(version) != previous
        {
            return version;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the skinning probe"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

fn check_sample(app: &App, expected: [[f32; 3]; 3]) {
    let world = app.sub_app(RenderApp).world();
    let device = world.resource::<RenderDevice>();
    let queue = world.resource::<RenderQueue>();
    let probe = world.resource::<Probe>();
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("skinning probe readback"),
        size: PROBE_BYTES,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&probe.sample, 0, &staging, 0, PROBE_BYTES);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(MapMode::Read, |result| result.expect("map probe sample"));
    device
        .poll(PollType::wait_indefinitely())
        .expect("wait for probe sample");
    let bytes = slice.get_mapped_range().expect("mapped probe sample");
    for (i, (actual, xyz)) in bytemuck::cast_slice::<u8, [f32; 4]>(&bytes)
        .iter()
        .zip(expected)
        .enumerate()
    {
        let expected = Vec3::from_array(xyz).extend(1.0);
        assert!(
            Vec4::from_array(*actual).abs_diff_eq(expected, 1e-5),
            "vertex {i}: expected {expected:?}, got {actual:?}"
        );
    }
    drop(bytes);
    staging.unmap();
}

//! Compare a native Blender Cloth cache with three live Bevy solvers.

mod cache;

use bevy::{
    asset::RenderAssetUsages,
    camera::{
        CameraOutputMode, Viewport,
        visibility::{NoFrustumCulling, RenderLayers},
    },
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::allocator::MeshAllocator,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{
            BlendState, CachedPipelineState, Extent3d, PipelineCache, PipelineDescriptor,
            TextureDimension, TextureFormat,
        },
        view::Msaa,
    },
    shader::ShaderCacheError,
    time::{TimeSystems, TimeUpdateStrategy},
    window::PrimaryWindow,
};

use std::{
    path::PathBuf,
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use bevy_cloth::{
    Cloth, ClothPlugin, ClothSolver, ClothStepSettings, ClothVbdMaterial, FixedAttachment,
};
use cache::Reference;

const SOLVERS: [(ClothSolver, &str); 3] = [
    (ClothSolver::ColoredGs, "Colored GS"),
    (ClothSolver::Jacobi, "Jacobi"),
    (ClothSolver::Vbd, "VBD"),
];
const SUBSTEPS: u32 = 6;
const SWEEPS: u32 = 10;

#[derive(Resource, Default)]
struct Playback {
    tick: usize,
    playing: bool,
    single_step: bool,
    side_view: bool,
}

#[derive(Resource)]
struct StripMaterial(Handle<StandardMaterial>);

#[derive(Component)]
struct Strip;

#[derive(Component)]
struct ReferenceStrip;

#[derive(Component)]
struct Panel(u32);

#[derive(Component)]
struct Status;

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct SceneReady {
    expected_meshes: Vec<AssetId<Mesh>>,
    ready: Arc<AtomicBool>,
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(path) = args
        .next()
        .map(PathBuf::from)
        .filter(|_| args.next().is_none())
    else {
        eprintln!("Usage: cloth-visual-compare <Blender cache from bake_cloth_comparison.py>");
        return ExitCode::FAILURE;
    };
    let reference = match Reference::load(&path) {
        Ok(reference) => reference,
        Err(error) => {
            eprintln!("{}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let fps = reference.fps;
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Cloth visual comparison".into(),
                    resolution: (1440, 540).into(),
                    ..default()
                }),
                ..default()
            })
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins((ClothPlugin, ExtractResourcePlugin::<SceneReady>::default()))
    .insert_resource(reference)
    .insert_resource(TimeUpdateStrategy::FixedTimesteps(0))
    .insert_resource(GlobalAmbientLight {
        brightness: 500.0,
        ..default()
    })
    .init_resource::<Playback>()
    .add_systems(Startup, (setup_view, rebuild_strips).chain())
    .add_systems(First, set_tick_budget.before(TimeSystems))
    .add_systems(Update, (handle_keys, update_view).chain())
    .add_systems(FixedUpdate, advance_reference);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .set_timestep_hz(f64::from(fps));
    app.sub_app_mut(RenderApp)
        .add_systems(Render, check_scene_ready.after(RenderSystems::Render));
    std::process::Termination::report(app.run())
}

fn strip_mesh(reference: &Reference) -> Mesh {
    let rest = &reference.frames[0];
    let uv: Vec<_> = rest.iter().map(|&[x, _, z]| [x, 0.5 + z / 0.4]).collect();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, rest.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv)
    .with_inserted_indices(Indices::U32(reference.indices.clone()));
    mesh.compute_area_weighted_normals();
    mesh
}

fn rebuild_strips(
    mut commands: Commands,
    reference: Res<Reference>,
    material: Res<StripMaterial>,
    mut meshes: ResMut<Assets<Mesh>>,
    existing: Query<(Entity, &Mesh3d), With<Strip>>,
) {
    for (entity, mesh) in &existing {
        commands.entity(entity).despawn();
        meshes.remove(mesh.id());
    }
    let input = strip_mesh(&reference);
    commands.spawn((
        Strip,
        ReferenceStrip,
        Mesh3d(meshes.add(input.clone())),
        MeshMaterial3d(material.0.clone()),
        NoFrustumCulling,
        RenderLayers::layer(0),
    ));
    let mut expected_meshes = Vec::with_capacity(SOLVERS.len());
    for (index, (solver, label)) in SOLVERS.into_iter().enumerate() {
        let mesh = meshes.add(input.clone());
        expected_meshes.push(mesh.id());
        let mut entity = commands.spawn((
            Strip,
            Name::new(label),
            Mesh3d(mesh),
            MeshMaterial3d(material.0.clone()),
            NoFrustumCulling,
            RenderLayers::layer(index + 1),
            solver,
            Cloth {
                particle_mass: reference.vertex_mass,
                stretch_compliance: 0.001,
                bend_compliance: 0.1,
                attachments: reference
                    .pins
                    .iter()
                    .map(|&particle| FixedAttachment {
                        particle,
                        position: Vec3::from(reference.frames[0][particle as usize]),
                    })
                    .collect(),
                collision_margin: 0.0,
                double_sided: false,
            },
            ClothStepSettings {
                gravity: Vec3::NEG_Y * reference.gravity,
                substeps: SUBSTEPS,
                iterations: SWEEPS,
            },
        ));
        if solver == ClothSolver::Vbd {
            entity.insert(ClothVbdMaterial {
                shear: 500.0,
                area: 500.0,
                bend: 0.005,
                ..default()
            });
        }
    }
    commands.insert_resource(SceneReady {
        expected_meshes,
        ready: Arc::new(AtomicBool::new(false)),
    });
}

fn check_scene_ready(
    scene: Option<Res<SceneReady>>,
    cache: Res<PipelineCache>,
    allocator: Res<MeshAllocator>,
    mut waiting_since: Local<Option<std::time::Instant>>,
) {
    let Some(scene) = scene else { return };
    if scene.ready.load(Ordering::Acquire) {
        *waiting_since = None;
        return;
    }
    // These are the pipelines the three elastic-only paths and Mesh output wait on.
    let required = [
        "xpbd predict",
        "xpbd update_velocity",
        "xpbd solve_stretch",
        "xpbd solve_bending",
        "solve_stretch_jacobi",
        "solve_bend_jacobi",
        "apply_jacobi",
        "predict_vbd",
        "solve_vbd_color",
        "stabilize_contact",
        "xpbd collider transform_vertices",
        "xpbd collider refit_leaves",
        "xpbd collider refit_internal",
        "xpbd output_mesh",
    ];
    let mut pending = Vec::new();
    for label in required {
        let state = cache.pipelines().find_map(|pipeline| {
            let PipelineDescriptor::ComputePipelineDescriptor(descriptor) = &pipeline.descriptor
            else {
                return None;
            };
            (descriptor.label.as_deref() == Some(label)).then_some(&pipeline.state)
        });
        match state {
            Some(CachedPipelineState::Ok(_)) => (),
            Some(CachedPipelineState::Err(
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable,
            )) => pending.push(label),
            Some(CachedPipelineState::Err(error)) => panic!("{label}: {error}"),
            _ => pending.push(label),
        }
    }
    let has_meshes = scene
        .expected_meshes
        .iter()
        .all(|id| allocator.mesh_vertex_slice(id).is_some());
    if pending.is_empty() && has_meshes {
        // Checked after the render graph: each new instance can initialize its tick
        // cursor at zero budget before the example starts sending ticks.
        scene.ready.store(true, Ordering::Release);
    } else if waiting_since
        .get_or_insert_with(std::time::Instant::now)
        .elapsed()
        .as_secs()
        >= 30
    {
        panic!(
            "Timed out preparing comparison: pending pipelines {pending:?}; meshes ready: {has_meshes}"
        );
    }
}

fn set_tick_budget(
    scene: Option<Res<SceneReady>>,
    reference: Res<Reference>,
    mut playback: ResMut<Playback>,
    mut strategy: ResMut<TimeUpdateStrategy>,
    mut next_tick: Local<Option<Instant>>,
) {
    let ready = scene.is_some_and(|scene| scene.ready.load(Ordering::Acquire));
    let now = Instant::now();
    let period = Duration::from_secs_f64(1.0 / f64::from(reference.fps));
    let due = if playback.playing && ready {
        let deadline = next_tick.get_or_insert(now);
        if now >= *deadline {
            // Advance one sample at most per displayed frame. Discard excess lag.
            *deadline = (*deadline + period).max(now);
            true
        } else {
            false
        }
    } else {
        *next_tick = None;
        false
    };
    let step = ready && playback.tick + 1 < reference.frames.len() && (due || playback.single_step);
    playback.single_step = false;
    *strategy = TimeUpdateStrategy::FixedTimesteps(u32::from(step));
}

fn advance_reference(
    mut playback: ResMut<Playback>,
    reference: Res<Reference>,
    strip: Single<&Mesh3d, With<ReferenceStrip>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // FixedUpdate also requests exactly one live solve from ClothPlugin.
    playback.tick += 1;
    if playback.tick + 1 == reference.frames.len() {
        playback.playing = false;
    }
    if let Some(mut mesh) = meshes.get_mut(strip.id()) {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            reference.frames[playback.tick].clone(),
        );
        mesh.compute_area_weighted_normals();
    }
}

fn handle_keys(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut playback: ResMut<Playback>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        playback.tick = 0;
        playback.playing = false;
        playback.single_step = false;
        commands.run_system_cached(rebuild_strips);
    } else if keys.just_pressed(KeyCode::KeyS) {
        playback.playing = false;
        playback.single_step = true;
    } else if keys.just_pressed(KeyCode::Space) {
        playback.playing = !playback.playing;
    }
    if keys.just_pressed(KeyCode::KeyV) {
        playback.side_view = !playback.side_view;
    }
}

fn camera_pose(side_view: bool) -> Transform {
    // The free edge can swing to either side of the fixed edge at x = 0.
    let focus = Vec3::new(0.0, -0.4, 0.0);
    let offset = if side_view {
        Vec3::new(0.0, 0.0, 3.0)
    } else {
        Vec3::new(0.9, 1.1, 3.0)
    };
    Transform::from_translation(focus + offset).looking_at(focus, Vec3::Y)
}

fn setup_view(
    mut commands: Commands,
    reference: Res<Reference>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let size = 80;
    let mut pixels = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let color = if ((x / 4) + (y / 10)) % 2 == 0 {
                [235, 231, 207, 255]
            } else {
                [45, 101, 140, 255]
            };
            pixels.extend(color);
        }
    }
    let texture = images.add(Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ));
    commands.insert_resource(StripMaterial(materials.add(StandardMaterial {
        base_color_texture: Some(texture),
        cull_mode: None,
        double_sided: true,
        perceptual_roughness: 1.0,
        ..default()
    })));
    commands.spawn((
        DirectionalLight {
            illuminance: 9_000.0,
            ..default()
        },
        Transform::from_xyz(-2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        RenderLayers::from_layers(&[0, 1, 2, 3]),
    ));
    for (index, title) in ["Blender", SOLVERS[0].1, SOLVERS[1].1, SOLVERS[2].1]
        .into_iter()
        .enumerate()
    {
        commands.spawn((
            Panel(index as u32),
            Camera3d::default(),
            Camera {
                order: index as isize,
                ..default()
            },
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: bevy::camera::ScalingMode::AutoMin {
                    min_width: 2.25,
                    min_height: 1.5,
                },
                ..OrthographicProjection::default_3d()
            }),
            camera_pose(false),
            RenderLayers::layer(index),
        ));
        let title = if index == 0 {
            format!("Blender {} (Q{})", reference.version, reference.quality)
        } else {
            title.to_owned()
        };
        commands.spawn((
            Text::new(title),
            TextFont {
                font_size: FontSize::Px(18.0),
                ..default()
            },
            Node {
                position_type: PositionType::Absolute,
                top: px(12),
                left: percent(index as f32 * 25.0 + 1.0),
                ..default()
            },
        ));
    }
    commands.spawn((
        Camera2d,
        Camera {
            order: 4,
            clear_color: ClearColorConfig::Custom(Color::NONE),
            output_mode: CameraOutputMode::Write {
                blend_state: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            ..default()
        },
        Msaa::Off,
        IsDefaultUiCamera,
    ));
    commands.spawn((
        Status,
        Text::new("Preparing GPU"),
        TextFont {
            font_size: FontSize::Px(16.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12),
            left: px(12),
            ..default()
        },
    ));
}

fn update_view(
    window: Single<&Window, With<PrimaryWindow>>,
    mut cameras: Query<(&Panel, &mut Camera, &mut Transform)>,
    mut status: Single<&mut Text, With<Status>>,
    scene: Option<Res<SceneReady>>,
    reference: Res<Reference>,
    playback: Res<Playback>,
) {
    let width = window.physical_width();
    let height = window.physical_height();
    for (panel, mut camera, mut transform) in &mut cameras {
        camera.is_active = width >= 4 && height > 0;
        if camera.is_active {
            let start = panel.0 * width / 4;
            let end = (panel.0 + 1) * width / 4;
            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(start, 0),
                physical_size: UVec2::new(end - start, height),
                ..default()
            });
        }
        *transform = camera_pose(playback.side_view);
    }
    let state = if !scene.is_some_and(|scene| scene.ready.load(Ordering::Acquire)) {
        "Preparing GPU"
    } else if playback.playing {
        "Playing"
    } else {
        "Paused"
    };
    status.0 = format!(
        "{state} | Tick {}/{} ({:.2} s)  |  Space play/pause  S step  R reset  V side/oblique",
        playback.tick,
        reference.frames.len() - 1,
        playback.tick as f64 / f64::from(reference.fps)
    );
}

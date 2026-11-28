//! View a native Blender Cloth cache without running the live solvers.

mod cache;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    time::{TimeSystems, TimeUpdateStrategy},
};

use std::{
    path::PathBuf,
    process::ExitCode,
    time::{Duration, Instant},
};

use cache::Reference;

#[derive(Resource, Default)]
struct Playback {
    tick: usize,
    playing: bool,
    single_step: bool,
    side_view: bool,
}

#[derive(Component)]
struct ReferenceStrip;

#[derive(Component)]
struct ViewCamera;

#[derive(Component)]
struct Status;

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
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Blender Cloth reference".into(),
            resolution: (960, 600).into(),
            ..default()
        }),
        ..default()
    }))
    .insert_resource(reference)
    .insert_resource(TimeUpdateStrategy::FixedTimesteps(0))
    .insert_resource(GlobalAmbientLight {
        brightness: 500.0,
        ..default()
    })
    .init_resource::<Playback>()
    .add_systems(Startup, setup_view)
    .add_systems(First, set_tick_budget.before(TimeSystems))
    .add_systems(Update, (handle_keys, update_view).chain())
    .add_systems(FixedUpdate, advance_reference);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .set_timestep_hz(f64::from(fps));
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

fn set_tick_budget(
    reference: Res<Reference>,
    mut playback: ResMut<Playback>,
    mut strategy: ResMut<TimeUpdateStrategy>,
    mut next_tick: Local<Option<Instant>>,
) {
    let now = Instant::now();
    let period = Duration::from_secs_f64(1.0 / f64::from(reference.fps));
    let due = if playback.playing {
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
    let step = playback.tick + 1 < reference.frames.len() && (due || playback.single_step);
    playback.single_step = false;
    *strategy = TimeUpdateStrategy::FixedTimesteps(u32::from(step));
}

fn advance_reference(
    mut playback: ResMut<Playback>,
    reference: Res<Reference>,
    strip: Single<&Mesh3d, With<ReferenceStrip>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    playback.tick += 1;
    if playback.tick + 1 == reference.frames.len() {
        playback.playing = false;
    }
    apply_reference_pose(&mut meshes, strip.id(), &reference.frames[playback.tick]);
}

fn apply_reference_pose(meshes: &mut Assets<Mesh>, id: AssetId<Mesh>, positions: &[[f32; 3]]) {
    if let Some(mut mesh) = meshes.get_mut(id) {
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
        mesh.compute_area_weighted_normals();
    }
}

fn handle_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut playback: ResMut<Playback>,
    reference: Res<Reference>,
    strip: Single<&Mesh3d, With<ReferenceStrip>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        playback.tick = 0;
        playback.playing = false;
        playback.single_step = false;
        apply_reference_pose(&mut meshes, strip.id(), &reference.frames[0]);
    } else if keys.just_pressed(KeyCode::KeyS) {
        playback.playing = false;
        playback.single_step = true;
    } else if keys.just_pressed(KeyCode::Space) {
        playback.playing = !playback.playing && playback.tick + 1 < reference.frames.len();
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
    mut meshes: ResMut<Assets<Mesh>>,
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
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(texture),
        cull_mode: None,
        double_sided: true,
        perceptual_roughness: 1.0,
        ..default()
    });
    commands.spawn((
        ReferenceStrip,
        Mesh3d(meshes.add(strip_mesh(&reference))),
        MeshMaterial3d(material),
        NoFrustumCulling,
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 9_000.0,
            ..default()
        },
        Transform::from_xyz(-2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        ViewCamera,
        Camera3d::default(),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::camera::ScalingMode::AutoMin {
                min_width: 2.25,
                min_height: 1.5,
            },
            ..OrthographicProjection::default_3d()
        }),
        camera_pose(false),
        IsDefaultUiCamera,
    ));
    commands.spawn((
        Text::new(format!(
            "Blender {} (Q{})",
            reference.version, reference.quality
        )),
        TextFont::from_font_size(18.0),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            ..default()
        },
    ));
    commands.spawn((
        Status,
        Text::new("Paused"),
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
    mut camera: Single<&mut Transform, With<ViewCamera>>,
    mut status: Single<&mut Text, With<Status>>,
    reference: Res<Reference>,
    playback: Res<Playback>,
) {
    **camera = camera_pose(playback.side_view);
    let state = if playback.playing {
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

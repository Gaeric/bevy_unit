use bevy::{light::CascadeShadowConfigBuilder, prelude::*};
use bevy_scene::SceneInstanceReady;
use bone_attachments::{BoneAttachmentsPlugin, scene::SceneAttachmentExt};
use std::f32::consts::PI;

const GLTF_PATH: &str = "Fox.glb";
const ATTACHMENT_PATH: &str = "FoxAttachment.glb";

fn main() {
    App::new()
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 2000.,
            ..default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(BoneAttachmentsPlugin)
        .add_systems(Startup, setup_mesh_and_animation)
        .add_systems(Startup, setup_camera_and_environment)
        .add_systems(Update, move_fox)
        .run();
}

// A component that stores a reference to an animation we want to play.
// This is created when we sart loading the mesh (see `setup_mesh_and_animation`) and
// read when the mesh has spawned (see `paly_animation_once_load`).
#[derive(Component)]
struct AnimationToPlay {
    graph_handle: Handle<AnimationGraph>,
    index: AnimationNodeIndex,
}

fn setup_mesh_and_animation(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
) {
    let (graph, index) = AnimationGraph::from_clip(
        asset_server.load(GltfAssetLabel::Animation(2).from_asset(GLTF_PATH)),
    );

    // Store the animation graph as an asset
    let graph_handle = graphs.add(graph);

    // Create a component that stores a reference to our animation.
    let animation_to_play = AnimationToPlay {
        graph_handle,
        index,
    };

    let mesh_scene = SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(GLTF_PATH)));

    let mut entity = commands.spawn((animation_to_play, mesh_scene));

    info!("setup mesh entity is {:?}", entity.id());

    entity
        .observe(play_animation_when_ready)
        .observe(attach_helm);
}

fn play_animation_when_ready(
    trigger: On<SceneInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    animations_to_play: Query<&AnimationToPlay>,
    mut players: Query<&mut AnimationPlayer>,
) {
    if let Ok(animation_to_play) = animations_to_play.get(trigger.entity) {
        for child in children.iter_descendants(trigger.entity) {
            if let Ok(mut player) = players.get_mut(child) {
                player.play(animation_to_play.index).repeat();

                commands
                    .entity(child)
                    .insert(AnimationGraphHandle(animation_to_play.graph_handle.clone()));
            }
        }
    }
}

fn attach_helm(
    trigger: On<SceneInstanceReady>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    // Start loading if the attachment.
    // let attachment_scene = asset_server.load_with_settings(
    //     GltfAssetLabel::Scene(0).from_asset(ATTACHMENT_PATH),
    //     |settings: &mut GltfLoaderSettings| {
    //         settings.include_animation_target_ids = true;
    //     },
    // );

    let attachment_scene = asset_server.load(GltfAssetLabel::Scene(0).from_asset(ATTACHMENT_PATH));

    commands
        .entity(trigger.entity)
        .attach_scene(attachment_scene);
}

fn setup_camera_and_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(200.0, 200.0, 300.0).looking_at(Vec3::new(0.0, 20.0, 0.0), Vec3::Y),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(500000.0, 500000.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
    ));

    // Light and Shadow
    commands.spawn((
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, 1.0, -PI / 4.0)),
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 200.0,
            maximum_distance: 400.0,
            ..default()
        }
        .build(),
    ));
}

fn move_fox(mut query: Single<&mut Transform, With<AnimationToPlay>>, time: Res<Time>) {
    let elapsed = time.elapsed_secs_f64() % 10.0;

    if elapsed < 2.5 {
        query.translation.z -= 1.0;
    } else if elapsed < 5.0 {
        query.translation.z += 1.0;
    } else if elapsed < 7.5 {
        query.translation.x += 1.0;
    } else {
        query.translation.x -= 1.0;
    }
}

use std::time::Duration;

use bevy::{animation::animate_targets, prelude::*};

#[allow(dead_code)]
pub(crate) fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 5000.,
            ..default()
        })
        .insert_resource(ClearColor(Color::srgb_u8(70, 112, 216)))
        .add_systems(Startup, setup)
        .add_systems(Startup, assets_setup)
        .add_systems(Update, setup_scene_once_loaded.before(animate_targets))
        .add_systems(Update, keyboard_animation_control)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::new(0.0, 0.0, 0.), Vec3::Y),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        MeshMaterial3d(materials.add(Color::srgba_u8(0, 0, 0, 255))),
    ));
}

#[derive(Resource)]
struct Animations {
    animations: Vec<AnimationNodeIndex>,
    #[allow(dead_code)]
    graph: Handle<AnimationGraph>,
}

fn assets_setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
) {
    let mut graph = AnimationGraph::new();
    let animations: Vec<AnimationNodeIndex> = graph
        .add_clips(
            [
                // GltfAssetLabel::Animation(4).from_asset("female_base/ani-model4.gltf"),
                // GltfAssetLabel::Animation(3).from_asset("female_base/ani-model4.gltf"),
                // GltfAssetLabel::Animation(2).from_asset("female_base/ani-model4.gltf"),
                // GltfAssetLabel::Animation(1).from_asset("female_base/ani-model4.gltf"),
                // GltfAssetLabel::Animation(0).from_asset("female_base/ani-model4.gltf"),
                GltfAssetLabel::Animation(0).from_asset("female_base/untitled.glb"),
            ]
            .into_iter()
            .map(|path| asset_server.load(path)),
            1.0,
            graph.root,
        )
        .collect();

    let graph = graphs.add(graph);
    commands.insert_resource(Animations {
        animations,
        graph: graph.clone(),
    });

    commands.spawn(SceneRoot(
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("female_base/untitled.glb")),
        // asset_server.load(GltfAssetLabel::Scene(0).from_asset("female_base/ani-model4.gltf")),
    ));
}

fn setup_scene_once_loaded(
    mut commands: Commands,
    animations: Res<Animations>,
    mut players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
) {
    for (entity, mut player) in &mut players {
        let mut transitions = AnimationTransitions::new();

        transitions
            .play(&mut player, animations.animations[0], Duration::ZERO)
            .repeat();

        commands
            .entity(entity)
            .insert(AnimationGraphHandle(animations.graph.clone()))
            .insert(transitions);
    }
}

fn keyboard_animation_control(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut animation_players: Query<(&mut AnimationPlayer, &mut AnimationTransitions)>,
    animations: Res<Animations>,
    mut current_animation: Local<usize>,
) {
    for (mut player, mut transitions) in &mut animation_players {
        let Some((&playering_animation_index, _)) = player.playing_animations().next() else {
            continue;
        };

        let playing_animation = player.animation_mut(playering_animation_index).unwrap();
        let mut speed = playing_animation.speed();
        if keyboard_input.just_pressed(KeyCode::Space) {
            if playing_animation.is_paused() {
                playing_animation.resume();
            } else {
                playing_animation.pause();
            }
        }

        if playing_animation.is_paused() {
            return;
        }

        if keyboard_input.just_pressed(KeyCode::ArrowUp) {
            speed = speed * 1.2;
        } else if keyboard_input.just_pressed(KeyCode::ArrowDown) {
            speed = speed * 0.8;
        }

        playing_animation.set_speed(speed);

        if keyboard_input.just_pressed(KeyCode::Enter) {
            *current_animation = (*current_animation + 1) % animations.animations.len();

            transitions
                .play(
                    &mut player,
                    animations.animations[*current_animation],
                    Duration::from_millis(250),
                )
                .repeat();
        }
    }
}

use std::time::Duration;

use bevy::{animation::animate_targets, prelude::*};

mod dev;

fn main() {
    App::new()
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 200.,
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(dev::plugin)
        .add_systems(Startup, (setup, assets_setup))
        .add_systems(Update, setup_scene_once_loaded.before(animate_targets))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(5.0, 5.0, 7.5)
            .looking_at(Vec3::new(0.0, 1., 0.), Vec3::Y),
        ..default()
    });

    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(50.0, 50.0)),
        material: materials.add(Color::srgb(0.027, 0.245, 1.000)),
        ..default()
    });
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
                GltfAssetLabel::Animation(2).from_asset("female_base/ani-model4.gltf"),
                GltfAssetLabel::Animation(1).from_asset("female_base/ani-model4.gltf"),
                GltfAssetLabel::Animation(0).from_asset("female_base/ani-model4.gltf"),
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

    commands.spawn(SceneBundle {
        scene: asset_server
            .load(GltfAssetLabel::Scene(0).from_asset("female_base/ani-model4.gltf")),
        ..default()
    });
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
            .insert(animations.graph.clone())
            .insert(transitions);
    }
}

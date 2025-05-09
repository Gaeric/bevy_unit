mod modular;

use std::{
    collections::{btree_map::Entry, BTreeMap},
    time::Duration,
};

use bevy::prelude::*;
use modular::*;

#[cfg(feature = "with-inspector")]
use bevy_inspector_egui::quick::WorldInspectorPlugin;

pub struct ModularCharacterPlugin;

impl Plugin for ModularCharacterPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        // plugins
        app.add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()));
        // .add_plugins(ModularPlugin);

        #[cfg(feature = "with-inspector")]
        app.add_plugins(WorldInspectorPlugin::new());

        // AmbientLight Resource
        app.insert_resource(AmbientLight {
            color: Color::default(),
            brightness: 1000.0,
        });

        // Systems
        app.add_systems(Startup, spawn_camera)
            .add_systems(Startup, spawn_text)
            .add_systems(Startup, spawn_models)
            .add_systems(Startup, setup_animation_graph)
            .add_systems(Update, cycle_through_animations);

        // Observers
        app.add_observer(animation_player_added);
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.5, 5.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
    ));
}

fn spawn_text(mut commands: Commands, asset_server: Res<AssetServer>) {
    let text_font = TextFont {
        font: asset_server.load("modular_character/FiraSans-Regular.ttf"),
        font_size: 16.0,
        ..Default::default()
    };

    let text_color = TextColor(Color::WHITE);
    commands
        .spawn(Node {
            top: Val::Px(3.0),
            left: Val::Px(3.0),
            flex_direction: FlexDirection::Column,
            ..Default::default()
        })
        .with_children(|parent| {
            parent.spawn((
                Text::new("Cycle through heads with Q and W"),
                text_font.clone(),
                text_color,
            ));
            parent.spawn((
                Text::new("Cycle through bodies with E and R"),
                text_font.clone(),
                text_color,
            ));
            parent.spawn((
                Text::new("Cycle through legs with T and Y"),
                text_font.clone(),
                text_color,
            ));
            parent.spawn((
                Text::new("Cycle through feet with U and I"),
                text_font.clone(),
                text_color,
            ));
        });
}

fn spawn_models(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        SceneRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("modular_character/Witch.gltf")),
        ),
        Transform::from_xyz(1.0, 0.0, 0.0),
        Name::new("Witch"),
    ));

    commands.spawn((
        SceneRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("modular_character/SciFi.gltf")),
        ),
        Transform::from_xyz(-1.0, 0., 0.0),
        Name::new("SciFi"),
    ));
}

#[derive(Debug, Resource)]
struct AnimationGraphCache {
    animations: Vec<AnimationNodeIndex>,
    graph: Handle<AnimationGraph>,
}

fn setup_animation_graph(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
) {
    let mut graph = AnimationGraph::new();
    let animations = graph
        .add_clips(
            (0..24).map(|index| {
                asset_server.load(
                    GltfAssetLabel::Animation(index).from_asset("modular_character/Witch.gltf"),
                )
            }),
            1.0,
            graph.root,
        )
        .collect();

    let graph_handle = graphs.add(graph);
    commands.insert_resource(AnimationGraphCache {
        animations,
        graph: graph_handle,
    });
}

// todo: analyzer the animation process
fn animation_player_added(
    trigger: Trigger<OnAdd, AnimationPlayer>,
    mut commands: Commands,
    graph_cache: Res<AnimationGraphCache>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let mut transitions = AnimationTransitions::new();

    transitions
        .play(
            &mut players.get_mut(trigger.entity()).unwrap(),
            graph_cache.animations[0],
            Duration::ZERO,
        )
        .resume();

    commands
        .entity(trigger.entity())
        .insert(transitions)
        .insert(AnimationGraphHandle(graph_cache.graph.clone()));
}

fn cycle_through_animations(
    mut players: Query<(Entity, &mut AnimationPlayer, &mut AnimationTransitions)>,
    mut animation_id: Local<BTreeMap<Entity, usize>>,
    graph_cache: Res<AnimationGraphCache>,
) {
    for (entity, mut player, mut transition) in &mut players {
        let next_to_play = match animation_id.entry(entity) {
            Entry::Vacant(e) => {
                e.insert(0);
                Some(0)
            }
            Entry::Occupied(mut e) => {
                if player.all_finished() | player.all_paused() {
                    *e.get_mut() = (e.get() + 1) % 24;
                    Some(*e.get())
                } else {
                    None
                }
            }
        };

        if let Some(next_ani) = next_to_play {
            transition
                .play(
                    &mut player,
                    graph_cache.animations[next_ani],
                    Duration::from_millis(250),
                )
                .resume();
        }
    }
}

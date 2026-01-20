use std::{
    collections::{BTreeMap, btree_map::Entry},
    time::Duration,
};

use bevy::prelude::*;
use bevy::scene::InstanceId;
use modular_character::{
    ModularAppExt, ModularCharacter, NewModularAsset, ResetChanged, create_modular_segment,
};

pub fn mc_model_path(path: &str) -> String {
    format!("modular_character/origin/{path}")
}

pub const HEADS: [&str; 3] = [
    // mc_model_path("Witch.gltf#Scene2").into(),
    "Witch.gltf#Scene2",
    // "SciFi.gltf#Scene2",
    "Soldier.gltf#Scene2",
    "Adventurer.gltf#Scene2",
];

pub const BODIES: [&str; 4] = [
    "Witch.gltf#Scene3",
    // "SciFi.gltf#Scene3",
    "Soldier.gltf#Scene3",
    "Adventurer.gltf#Scene3",
    "scifi_torso.glb#Scene0",
];

pub const LEGS: [&str; 4] = [
    "Witch.gltf#Scene4",
    // "SciFi.gltf#Scene4",
    "Soldier.gltf#Scene4",
    "Adventurer.gltf#Scene4",
    // todo: model incorrect, remove the feet from this scene
    "witch_legs.glb#Scene0",
];

pub const FEETS: [&str; 3] = [
    "Witch.gltf#Scene5",
    // "SciFi.gltf#Scene5",
    "Soldier.gltf#Scene5",
    "Adventurer.gltf#Scene5",
];

create_modular_segment!(Head, 0, HEADS);
create_modular_segment!(Body, 1, BODIES);
create_modular_segment!(Legs, 2, LEGS);
create_modular_segment!(Feet, 3, FEETS);

impl Plugin for DemoModularPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        // plugins
        app.add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()));
        app.add_message::<ResetChanged>();
        app.register_modular_component::<ModularCharacterHead>();
        app.register_modular_component::<ModularCharacterBody>();
        app.register_modular_component::<ModularCharacterLegs>();
        app.register_modular_component::<ModularCharacterFeet>();

        #[cfg(feature = "with-inspector")]
        {
            use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};

            app.add_plugins(EguiPlugin::default())
                .add_plugins(WorldInspectorPlugin::new());
        }

        // AmbientLight Resource
        app.insert_resource(AmbientLight {
            color: Color::default(),
            brightness: 1000.0,
            ..default()
        });

        // Systems
        app.add_systems(Startup, spawn_camera)
            .add_systems(Startup, spawn_text)
            .add_systems(Startup, spawn_models)
            .add_systems(Startup, spawn_modular)
            .add_systems(Startup, setup_animation_graph)
            .add_systems(Update, cycle_through_animations);

        // app.add_systems(Update, (cycle_head, cycle_body, cycle_leg, cycle_feet));

        app.add_systems(
            Update,
            (
                move |c: Commands,
                      q: Single<&ModularCharacterHead>,
                      k: Res<ButtonInput<KeyCode>>| {
                    cycle_modular::<ModularCharacterHead>(c, q, k, (KeyCode::KeyQ, KeyCode::KeyW))
                },
                move |c: Commands,
                      q: Single<&ModularCharacterBody>,
                      k: Res<ButtonInput<KeyCode>>| {
                    cycle_modular::<ModularCharacterBody>(c, q, k, (KeyCode::KeyE, KeyCode::KeyR))
                },
                move |c: Commands,
                      q: Single<&ModularCharacterLegs>,
                      k: Res<ButtonInput<KeyCode>>| {
                    cycle_modular::<ModularCharacterLegs>(c, q, k, (KeyCode::KeyT, KeyCode::KeyY))
                },
                move |c: Commands,
                      q: Single<&ModularCharacterFeet>,
                      k: Res<ButtonInput<KeyCode>>| {
                    cycle_modular::<ModularCharacterFeet>(c, q, k, (KeyCode::KeyU, KeyCode::KeyI))
                },
            ),
        );

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
            asset_server.load(GltfAssetLabel::Scene(0).from_asset(mc_model_path("Witch.gltf"))),
        ),
        Transform::from_xyz(1.0, 0.0, 0.0),
        Name::new("Witch"),
    ));

    // commands.spawn((
    //     SceneRoot(
    //         asset_server.load(GltfAssetLabel::Scene(0).from_asset(mc_model_path("SciFi.gltf"))),
    //     ),
    //     Transform::from_xyz(-1.0, 0., 0.0),
    //     Name::new("SciFi"),
    // ));
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
                asset_server
                    .load(GltfAssetLabel::Animation(index).from_asset(mc_model_path("Witch.gltf")))
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
    trigger: On<Add, AnimationPlayer>,
    mut commands: Commands,
    graph_cache: Res<AnimationGraphCache>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let mut transitions = AnimationTransitions::new();

    transitions
        .play(
            &mut players.get_mut(trigger.entity).unwrap(),
            graph_cache.animations[0],
            Duration::ZERO,
        )
        .resume();

    commands
        .entity(trigger.entity)
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

fn cycle_modular<T: ModularCharacter>(
    mut commands: Commands,
    modular: Single<&T>,
    key_input: Res<ButtonInput<KeyCode>>,
    keys: (KeyCode, KeyCode),
) {
    let (add_key, sub_key) = keys;

    let assets = modular.assets();
    let new_id = if key_input.just_pressed(add_key) {
        modular.id().wrapping_sub(1).min(assets.len() - 1)
    } else if key_input.just_pressed(sub_key) {
        (modular.id() + 1) % assets.len()
    } else {
        return;
    };

    let path = format!("modular_character/origin/{}", assets[new_id]);

    commands.trigger(NewModularAsset {
        // entity,
        component_id: modular.component_id(),
        id: new_id,
        path,
    });
}

fn spawn_modular(
    mut commands: Commands,
    mut scene_spawner: ResMut<SceneSpawner>,
    asset_server: Res<AssetServer>,
) {
    let entity = commands
        .spawn((
            Transform::default(),
            Visibility::default(),
            Name::new("Modular"),
            ModularCharacterHead {
                id: 0,
                instance_id: Some(scene_spawner.spawn(asset_server.load(mc_model_path(HEADS[0])))),
                entities: vec![],
            },
            ModularCharacterBody {
                id: 0,
                instance_id: Some(scene_spawner.spawn(asset_server.load(mc_model_path(BODIES[0])))),
                entities: vec![],
            },
            ModularCharacterLegs {
                id: 0,
                instance_id: Some(scene_spawner.spawn(asset_server.load(mc_model_path(LEGS[0])))),
                entities: vec![],
            },
            ModularCharacterFeet {
                id: 0,
                instance_id: Some(scene_spawner.spawn(asset_server.load(mc_model_path(FEETS[0])))),
                entities: vec![],
            },
        ))
        .id();

    // Armature
    scene_spawner.spawn_as_child(
        asset_server.load(GltfAssetLabel::Scene(1).from_asset(mc_model_path("Witch.gltf"))),
        entity,
    );
}

pub struct DemoModularPlugin;

fn main() -> AppExit {
    App::new().add_plugins(DemoModularPlugin).run()
}

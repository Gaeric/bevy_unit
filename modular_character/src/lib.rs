mod modular;

use std::time::Duration;

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
            .add_systems(Startup, spawn_text);

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

#[derive(Debug, Resource)]
struct AnimationGraphCache {
    animations: Vec<AnimationNodeIndex>,
    graph: Handle<AnimationGraph>,
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

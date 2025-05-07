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
        app.add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
            .add_plugins(ModularPlugin);

        #[cfg(feature = "with-inspector")]
        app.add_plugins(WorldInspectorPlugin::new());

        // AmbientLight Resource
        app.insert_resource(AmbientLight {
            color: Color::default(),
            brightness: 1000.0,
        });

        // Observers
        app.add_observer(animation_player_added);
    }
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

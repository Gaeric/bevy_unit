use bevy::prelude::*;
// use blenvy::*;

use modular_character::ModularCharacterPlugin;

// mod animation_demo;
// mod bone_demo;
// mod cloth_demo;
// mod dev;
// mod ui;

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        // .add_plugins(animation_demo::plugin)
        // .add_plugins(bone_demo::plugin)
        // .add_plugins(ui::plugin)
        .add_plugins(ModularCharacterPlugin)
        // .add_plugins(dev::plugin)
        // .add_plugins(cloth_demo::plugin)
        // .add_plugins(BlenvyPlugin::default())
        // .register_type::<Player>()
        // .add_plugins(PhysicsPlugins::default())
        // .add_systems(Startup, setup)
        .run()
}

// #[derive(Component, Reflect)]
// #[reflect(Component)]
// struct Player {
//     strength: f32,
//     perception: f32,
//     endurance: f32,
//     charisma: f32,
//     intelligence: f32,
//     agility: f32,
//     luck: f32,
// }

// fn setup(mut commands: Commands) {
//     commands.spawn((
//         BlueprintInfo::from_path("levels/World.glb"),
//         SpawnBlueprint,
//         HideUntilReady,
//         GameWorldTag,
//     ));
// }

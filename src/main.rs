use bevy::prelude::*;

// use bevy_shine::ShinePlugin;

use bevy_waltz::WaltzPlugin;

// mod animation_demo;
// mod dev;
// mod ui;

fn main() -> AppExit {
    App::new()
        // .add_plugins(DefaultPlugins)
        // .add_plugins(animation_demo::plugin)
        // .add_plugins(bone_demo::plugin)
        // .add_plugins(ui::plugin)
        .add_plugins(WaltzPlugin)
        // .add_plugins(dev::plugin)
        // .add_plugins(ShinePlugin)
        .run()
}

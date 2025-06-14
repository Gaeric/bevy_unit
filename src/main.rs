use bevy::prelude::*;

// use bevy_shine::ShinePlugin;

use bevy_waltz::WaltzPlugin;

// mod ui;

fn main() -> AppExit {
    App::new()
        // .add_plugins(DefaultPlugins)
        // .add_plugins(ui::plugin)
        .add_plugins(WaltzPlugin)
        // .add_plugins(dev::plugin)
        // .add_plugins(ShinePlugin)
        .run()
}
